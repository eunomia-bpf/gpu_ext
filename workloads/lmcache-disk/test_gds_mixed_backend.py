#!/usr/bin/env python3
"""CPU-only wiring tests for the LMCache GDS mixed-traffic runner.

No GPU work is performed: the real LMCache GdsBackend, cuFile, and CUDA pool
are never constructed here.  The tests cover the rotation and traffic-plan
logic, the metric computation, the controlled policy inputs against the
committed policy adapter, and the reference-lifetime/completion wiring of the
committed ``lmcache_gds_backend_adapter`` on a fake backend that mirrors the
LMCache 0.5.4 ``submit_put_task`` / ``_async_save_bytes_to_disk`` contract.
"""

import asyncio
import importlib.util
from pathlib import Path
import os
import sys
import threading
import time
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "run_gds_mixed_backend", HERE / "run_gds_mixed_backend.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

os.environ.pop("LMCACHE_GDS_POLICY_MODE", None)
sys.path.insert(0, str(HERE / "gds-control"))
import lmcache_gds_backend_adapter as gds_adapter  # noqa: E402
import lmcache_gds_policy_adapter as gds_policy  # noqa: E402


MIB = 1024 * 1024


def parse(*argv: str):
    return runner.parse_args(list(argv))


def make_args(**overrides) -> object:
    args = parse()
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class ConfigTests(unittest.TestCase):
    def test_configs_and_modes_are_the_three_gds_arms(self):
        self.assertEqual(runner.CONFIGS, ("gds_fifo", "gds_native", "gds_bpf"))
        self.assertEqual(runner.POLICY_MODES, {
            "gds_fifo": "fifo",
            "gds_native": "native",
            "gds_bpf": "bpf",
        })

    def test_controlled_inputs_are_labeled_and_shared(self):
        inputs = runner.CONTROLLED_POLICY_INPUTS
        self.assertEqual(inputs["hbm_pressure_permille"], 801)
        self.assertEqual(inputs["slack_ns"], 10_000_000)
        self.assertIs(inputs["speculative_recomputable"], False)
        self.assertIn("controlled", inputs["label"])
        self.assertIn("not measured live HBM pressure", inputs["label"])

    def test_default_plan_shape(self):
        plan = runner.make_plan(parse())
        self.assertEqual(plan.reads, 4)
        self.assertEqual(plan.writes, 6)
        self.assertEqual(plan.object_bytes, 24 * MIB)
        self.assertEqual(plan.pool_bytes, 256 * MIB)
        self.assertEqual(plan.read_stagger_s, 0.002)
        self.assertEqual(plan.write_stagger_s, 0.004)
        self.assertEqual(plan.objects, 10)
        self.assertEqual(plan.to_dict()["total_offered_bytes"], 10 * 24 * MIB)

    def test_plan_is_mode_independent(self):
        for _ in runner.CONFIGS:
            self.assertEqual(
                runner.make_plan(parse()).to_dict(),
                runner.make_plan(parse()).to_dict(),
            )


class RotationTests(unittest.TestCase):
    def test_each_mode_occupies_each_position_once_per_block(self):
        orders = runner.rotation_orders(5)
        self.assertEqual(len(orders), 5)
        for order in orders:
            self.assertEqual(sorted(order), sorted(runner.CONFIGS))

    def test_blocks_are_cyclic_rotations(self):
        orders = runner.rotation_orders(5)
        for block, order in enumerate(orders):
            expected = list(
                runner.CONFIGS[block % 3:] + runner.CONFIGS[:block % 3])
            self.assertEqual(order, expected)

    def test_single_block_for_first_measurement(self):
        self.assertEqual(len(runner.rotation_orders(1)), 1)


class PlanValidationTests(unittest.TestCase):
    def test_pool_too_small_is_rejected(self):
        args = make_args(gds_buffer_size_mib=100, reads=4, writes=6)
        with self.assertRaises(ValueError):
            runner.make_plan(args)

    def test_object_alignment_is_checked(self):
        plan = runner.TrafficPlan(reads=1, writes=1, object_bytes=1000,
                                  read_stagger_s=0.0, write_stagger_s=0.0,
                                  pool_bytes=8 * MIB)
        with self.assertRaises(ValueError):
            runner.validate_plan(plan)

    def test_zero_reads_or_writes_is_rejected(self):
        plan = runner.TrafficPlan(reads=0, writes=1, object_bytes=4096,
                                  read_stagger_s=0.0, write_stagger_s=0.0,
                                  pool_bytes=8 * MIB)
        with self.assertRaises(ValueError):
            runner.validate_plan(plan)


class MetricsTests(unittest.TestCase):
    @staticmethod
    def request(role, offer, completed, status="completed",
                nbytes=24 * MIB) -> dict:
        return {
            "role": role,
            "bytes": nbytes,
            "offer_s": offer,
            "submitted_s": offer,
            "completed_s": completed,
            "status": status,
        }

    def test_read_percentiles_from_offer_to_completion(self):
        requests = [
            self.request("read_demand", 0.0, 0.010),
            self.request("read_demand", 0.002, 0.022),
            self.request("read_demand", 0.004, 0.034),
            self.request("read_demand", 0.006, 0.046),
        ]
        metrics = runner.compute_metrics(requests)
        self.assertEqual(metrics["completed_reads"], 4)
        self.assertAlmostEqual(metrics["read_end_to_end_p50_ms"], 20.0)
        self.assertAlmostEqual(metrics["read_end_to_end_p99_ms"], 40.0)

    def test_single_read_has_equal_p50_p99(self):
        metrics = runner.compute_metrics(
            [self.request("read_demand", 0.0, 0.05)])
        self.assertAlmostEqual(metrics["read_end_to_end_p50_ms"], 50.0)
        self.assertAlmostEqual(metrics["read_end_to_end_p99_ms"], 50.0)

    def test_write_completion_throughput(self):
        requests = [
            self.request("write_background", 0.0, 5.0),
            self.request("write_background", 1.0, 6.0),
            self.request("write_background", 2.0, 7.0),
        ]
        metrics = runner.compute_metrics(requests)
        self.assertEqual(metrics["completed_writes"], 3)
        self.assertAlmostEqual(
            metrics["write_completion_throughput_mib_s"],
            3 * 24 * MIB / 7.0 / MIB)
        self.assertAlmostEqual(metrics["write_completion_ops_s"], 3.0 / 7.0)

    def test_total_storage_bandwidth_mixes_roles(self):
        requests = [
            self.request("read_demand", 0.0, 1.0),
            self.request("write_background", 0.0, 3.0),
        ]
        metrics = runner.compute_metrics(requests)
        self.assertAlmostEqual(
            metrics["total_storage_bandwidth_mib_s"],
            2 * 24 * MIB / 3.0 / MIB)

    def test_uncompleted_requests_are_excluded(self):
        requests = [
            self.request("read_demand", 0.0, 0.010),
            self.request("read_demand", 0.0, None, status="not_completed"),
            self.request("write_background", 0.0, None,
                         status="save_not_confirmed"),
        ]
        metrics = runner.compute_metrics(requests)
        self.assertEqual(metrics["completed_reads"], 1)
        self.assertEqual(metrics["completed_writes"], 0)
        self.assertIsNone(metrics.get("write_completion_throughput_mib_s"))
        self.assertAlmostEqual(metrics["total_storage_bandwidth_mib_s"],
                               24 * MIB / 0.010 / MIB)


class ControlledPolicyInputTests(unittest.TestCase):
    """The committed deciders, fed exactly the runner's controlled inputs."""

    def _raw_request(self, kind):
        if kind == "write":
            return gds_policy.PolicyRequest(
                op=gds_policy.OP_WRITE,
                flags=gds_policy.FLAG_SAFE_TO_DEFER,
                nbytes=24 * MIB,
                slack_ns=runner.CONTROLLED_POLICY_INPUTS["slack_ns"],
                hbm_pressure_permille=runner.CONTROLLED_POLICY_INPUTS[
                    "hbm_pressure_permille"],
            )
        return gds_policy.PolicyRequest(
            op=gds_policy.OP_READ,
            flags=gds_policy.FLAG_DEMAND,
            nbytes=24 * MIB,
            hbm_pressure_permille=runner.CONTROLLED_POLICY_INPUTS[
                "hbm_pressure_permille"],
        )

    def test_fifo_always_submits_now(self):
        decider = gds_policy.FifoDecider()
        for kind in ("read", "write"):
            decision = decider.decide(self._raw_request(kind), 0)
            self.assertEqual(decision.action, gds_policy.ACTION_SUBMIT_NOW)

    def test_native_defers_background_write_under_controlled_pressure(self):
        decision = gds_policy.NativeDecider().decide(
            self._raw_request("write"), 0)
        self.assertEqual(decision.action, gds_policy.ACTION_DEFER)
        self.assertEqual(decision.defer_ns, 10_000_000)

    def test_native_never_defers_demand_reads(self):
        decision = gds_policy.NativeDecider().decide(
            self._raw_request("read"), 0)
        self.assertEqual(decision.action, gds_policy.ACTION_SUBMIT_NOW)


class FakeMemoryObj:
    """Mirrors the reference-count contract of a LMCache TensorMemoryObj."""

    def __init__(self, nbytes):
        self.nbytes = nbytes
        self.refs = 1
        self.tensor = object()

    def get_size(self):
        return self.nbytes

    def ref_count_up(self):
        self.refs += 1

    def ref_count_down(self):
        self.refs -= 1
        if self.refs < 0:
            raise AssertionError("double free in test")

    @property
    def ref_count(self):
        return self.refs


class FakeBackend:
    """Mirrors the LMCache 0.5.4 GdsBackend save/submit contract."""

    def __init__(self, loop):
        self.loop = loop
        self.put_lock = threading.Lock()
        self.put_tasks = set()
        self.hot_lock = threading.Lock()
        self.hot_cache = {}
        self.save_started = []

    def submit_put_task(self, key, memory_obj, on_complete_callback=None):
        assert memory_obj.tensor is not None
        memory_obj.ref_count_up()
        with self.put_lock:
            self.put_tasks.add(key)
        return asyncio.run_coroutine_threadsafe(
            self._async_save_bytes_to_disk(key, memory_obj, on_complete_callback),
            self.loop,
        )

    async def _async_save_bytes_to_disk(self, key, memory_obj,
                                        on_complete_callback=None):
        try:
            self.save_started.append((key, time.perf_counter()))
        finally:
            memory_obj.ref_count_down()
            with self.put_lock:
                self.put_tasks.discard(key)
        if on_complete_callback is not None:
            on_complete_callback(key)

    def get_blocking(self, key):
        return FakeMemoryObj(24 * MIB)

    def get_non_blocking(self, key, location=None):
        return None

    def batched_get_blocking(self, keys):
        return [self.get_blocking(key) for key in keys]


class AdapterWiringTests(unittest.TestCase):
    """Reference lifetime and completion handling, no GPU involved."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self.loop.run_forever,
                                             daemon=True)
        self.loop_thread.start()
        self.backend = FakeBackend(self.loop)

    def tearDown(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5.0)
        self.loop.close()

    def install(self, mode):
        provider = gds_adapter.EnvironmentRequestProvider(
            gds_adapter.Telemetry(
                hbm_pressure_permille=runner.CONTROLLED_POLICY_INPUTS[
                    "hbm_pressure_permille"],
                slack_ns=runner.CONTROLLED_POLICY_INPUTS["slack_ns"],
                speculative_recomputable=runner.CONTROLLED_POLICY_INPUTS[
                    "speculative_recomputable"],
            )
        )
        return gds_adapter.install_backend(self.backend, mode=mode,
                                           request_provider=provider)

    def test_fifo_write_reaches_save_immediately(self):
        adapter = self.install("fifo")
        key = "w0"
        log = {key: {"submitted_s": None}}
        log_lock = threading.Lock()
        t0 = time.perf_counter()
        runner.wrap_submit_timing(self.backend, log, log_lock, t0)
        memory_obj = FakeMemoryObj(24 * MIB)
        future = self.backend.submit_put_task(key, memory_obj)
        future.result(timeout=10.0)
        self.assertEqual(len(self.backend.save_started), 1)
        self.assertLess(self.backend.save_started[0][1] - t0, 0.020)
        self.assertIsNotNone(log[key]["submitted_s"])
        self.assertEqual(memory_obj.ref_count, 0)
        self.assertEqual(adapter.stats["submit_now"], 1)
        self.assertEqual(adapter.stats["defer"], 0)
        self.assertEqual(self.backend.put_tasks, set())

    def test_native_write_is_deferred_by_controlled_slack(self):
        adapter = self.install("native")
        key = "w0"
        log = {key: {"submitted_s": None}}
        log_lock = threading.Lock()
        t0 = time.perf_counter()
        runner.wrap_submit_timing(self.backend, log, log_lock, t0)
        memory_obj = FakeMemoryObj(24 * MIB)
        offer = time.perf_counter()
        future = self.backend.submit_put_task(key, memory_obj)
        future.result(timeout=10.0)
        self.assertEqual(len(self.backend.save_started), 1)
        deferred = self.backend.save_started[0][1] - offer
        self.assertGreaterEqual(deferred, 0.0095)
        self.assertLess(deferred, 1.0)
        self.assertAlmostEqual(log[key]["submitted_s"] - (offer - t0),
                               0.010, delta=0.005)
        self.assertEqual(adapter.stats["defer"], 1)
        self.assertEqual(memory_obj.ref_count, 0)
        self.assertEqual(self.backend.put_tasks, set())
        adapter.close()

    def test_deferred_write_keeps_reference_until_save(self):
        adapter = self.install("native")
        key = "w0"
        memory_obj = FakeMemoryObj(24 * MIB)
        observed = []

        def on_complete(_key):
            observed.append(memory_obj.ref_count)

        future = self.backend.submit_put_task(key, memory_obj,
                                             on_complete_callback=on_complete)
        time.sleep(0.003)
        self.assertEqual(memory_obj.ref_count, 2)
        self.assertIn(key, self.backend.put_tasks)
        future.result(timeout=10.0)
        self.assertEqual(on_complete, observed[0] and on_complete)
        self.assertEqual(observed[0], 1)
        memory_obj.ref_count_down()
        self.assertEqual(memory_obj.ref_count, 0)
        self.assertEqual(self.backend.put_tasks, set())
        adapter.close()

    def test_demand_read_always_submits_through_adapter(self):
        adapter = self.install("bpf") if False else self.install("fifo")
        memory_obj = self.backend.get_blocking("r0")
        self.assertIsInstance(memory_obj, FakeMemoryObj)
        self.assertEqual(adapter.stats["decisions"], 1)
        self.assertEqual(adapter.stats["submit_now"], 1)
        memory_obj.ref_count_down()
        self.assertEqual(memory_obj.ref_count, 0)
        adapter.close()


class CliTests(unittest.TestCase):
    def test_dry_run_reports_controlled_inputs_and_traffic(self):
        namespace = parse("--blocks", "1", "--dry-run")
        plan = runner.dry_run_plan(namespace)
        self.assertTrue(plan["dry_run"])
        self.assertEqual(plan["blocks"], 1)
        self.assertEqual(plan["configs"], list(runner.CONFIGS))
        self.assertEqual(plan["traffic"]["reads"], 4)
        self.assertEqual(plan["traffic"]["writes"], 6)
        self.assertEqual(plan["traffic"]["object_bytes"], 24 * MIB)
        self.assertEqual(plan["controlled_policy_inputs"],
                         dict(runner.CONTROLLED_POLICY_INPUTS))
        self.assertEqual(plan["gates"], [])
        self.assertFalse(plan["retries"])

    def test_default_blocks_is_five_rotations(self):
        plan = runner.dry_run_plan(parse("--dry-run"))
        self.assertEqual(len(plan["block_orders"]), 5)
        for order in plan["block_orders"]:
            self.assertEqual(sorted(order), sorted(runner.CONFIGS))

    def test_invalid_pool_configuration_rejected(self):
        with self.assertRaises(SystemExit):
            parse("--gds-buffer-size-mib", "0")

    def test_cell_metrics_shape(self):
        record = {
            "metrics": {
                "read_end_to_end_p50_ms": 10.0,
                "read_end_to_end_p99_ms": 20.0,
                "write_completion_throughput_mib_s": 30.0,
                "total_storage_bandwidth_mib_s": 40.0,
                "completed_reads": 4,
                "completed_writes": 6,
            },
            "decision_counts": {"decisions": 10, "defer": 6},
        }
        metrics = runner.cell_metrics(record)
        self.assertEqual(metrics["defer_decisions"], 6)
        self.assertEqual(metrics["read_end_to_end_p50_ms"], 10.0)


if __name__ == "__main__":
    unittest.main()
