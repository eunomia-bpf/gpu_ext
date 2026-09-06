#!/usr/bin/env python3
"""CPU-only tests for the LMCache GDS policy-vs-mechanism ablation runner."""

import importlib.util
import io
import json
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("run_gds_policy_ablation",
                                             HERE / "run_gds_policy_ablation.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

os.environ.pop("LMCACHE_GDS_POLICY_MODE", None)
sys.path.insert(0, str(HERE / "gds-control"))
import lmcache_gds_backend_adapter as gds_adapter  # noqa: E402
import lmcache_gds_policy_adapter as gds_policy  # noqa: E402


BASE_ENV = {
    "LMCACHE_LOCAL_DISK": "file:///old",
    "LMCACHE_MAX_LOCAL_DISK_SIZE": "16.0",
    "LMCACHE_EXTRA_CONFIG": '{"use_odirect":true}',
    "PYTHONPATH": "/existing",
}


class _FakeBackend:
    class _Lock:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    hot_lock = _Lock()
    hot_cache = {}


class ArmMappingTests(unittest.TestCase):
    def test_configs_are_exactly_the_five_arms(self):
        self.assertEqual(runner.CONFIGS, (
            "gds_fifo", "gds_bpf_floor", "gds_defer_native",
            "gds_full_native", "gds_full_bpf",
        ))

    def test_arm_to_policy_mode_mapping(self):
        self.assertEqual(runner.POLICY_MODES, {
            "gds_fifo": "fifo",
            "gds_bpf_floor": "bpf",
            "gds_defer_native": "native",
            "gds_full_native": "native",
            "gds_full_bpf": "bpf",
        })

    def test_mechanism_pairs_share_policy_inputs_and_floor_arms_share_pressure_zero(self):
        self.assertEqual(runner.POLICY_ENVIRONMENTS["gds_fifo"],
                         runner.POLICY_ENVIRONMENTS["gds_bpf_floor"])
        self.assertEqual(runner.POLICY_ENVIRONMENTS["gds_full_native"],
                         runner.POLICY_ENVIRONMENTS["gds_full_bpf"])
        self.assertNotEqual(runner.POLICY_ENVIRONMENTS["gds_defer_native"],
                            runner.POLICY_ENVIRONMENTS["gds_full_native"])

    def test_policy_environment_only_holds_policy_variables(self):
        for arm, variables in runner.POLICY_ENVIRONMENTS.items():
            self.assertEqual(set(variables),
                             {key for key in variables
                              if key.startswith("LMCACHE_GDS_POLICY_")},
                             arm)
            for value in variables.values():
                self.assertIsInstance(value, str, arm)


class PolicyEnvironmentTests(unittest.TestCase):
    def test_exact_policy_variables_per_arm(self):
        expected = {
            "gds_fifo": {"LMCACHE_GDS_POLICY_HBM_PRESSURE_PERMILLE": "0"},
            "gds_bpf_floor": {"LMCACHE_GDS_POLICY_HBM_PRESSURE_PERMILLE": "0"},
            "gds_defer_native": {
                "LMCACHE_GDS_POLICY_HBM_PRESSURE_PERMILLE": "801",
                "LMCACHE_GDS_POLICY_SLACK_NS": "10000000",
                "LMCACHE_GDS_POLICY_SPECULATIVE_RECOMPUTABLE": "False",
            },
            "gds_full_native": {
                "LMCACHE_GDS_POLICY_HBM_PRESSURE_PERMILLE": "801",
                "LMCACHE_GDS_POLICY_SLACK_NS": "10000000",
                "LMCACHE_GDS_POLICY_SPECULATIVE_RECOMPUTABLE": "True",
                "LMCACHE_GDS_POLICY_ESTIMATED_TRANSFER_NS": "5000000",
                "LMCACHE_GDS_POLICY_RECOMPUTE_NS": "1000000",
            },
        }
        self.assertEqual(runner.POLICY_ENVIRONMENTS,
                         {**expected, "gds_full_bpf": expected["gds_full_native"]})

    def test_environment_builds_gds_backend_and_exact_policy_variables(self):
        for arm in runner.CONFIGS:
            with self.subTest(arm=arm):
                with mock.patch.object(runner, "_BASE_SERVER_ENVIRONMENT",
                                       return_value=BASE_ENV.copy()) as fake:
                    env = runner.gds_server_environment(
                        arm, Path("/cache"), "575.57.08", 256)
                fake.assert_called_once_with("lmcache_disk", Path("/cache"), "575.57.08")
                self.assertNotIn("LMCACHE_LOCAL_DISK", env)
                self.assertNotIn("LMCACHE_MAX_LOCAL_DISK_SIZE", env)
                self.assertEqual(env["LMCACHE_GDS_PATH"], "/cache")
                self.assertEqual(env["LMCACHE_GDS_BUFFER_SIZE"], "256")
                self.assertEqual(env["LMCACHE_USE_GDS"], "True")
                self.assertEqual(env["LMCACHE_GDS_BACKEND"], "cufile")
                self.assertEqual(env["LMCACHE_GDS_POLICY_MODE"],
                                 runner.POLICY_MODES[arm])
                self.assertEqual(
                    {key: value for key, value in env.items()
                     if key.startswith("LMCACHE_GDS_POLICY_")},
                    {"LMCACHE_GDS_POLICY_MODE": runner.POLICY_MODES[arm],
                     **runner.POLICY_ENVIRONMENTS[arm]})
                self.assertEqual(env["LMCACHE_EXTRA_CONFIG"],
                                 runner.ops.canonical({"use_direct_io": True}))
                self.assertEqual(env["PYTHONPATH"].split(":"),
                                 [str(runner.BOOTSTRAP), str(runner.GDS_CONTROL),
                                  "/existing"])

    def test_unknown_arm_is_rejected(self):
        with self.assertRaises(ValueError):
            runner.gds_server_environment("lmcache_cpu", Path("/cache"),
                                          "575.57.08", 256)

    def test_injected_policy_environment_parses_to_intended_telemetry(self):
        intended = {
            "gds_fifo": dict(hbm_pressure_permille=0, slack_ns=0,
                             estimated_transfer_ns=0, recompute_ns=0,
                             speculative_recomputable=False),
            "gds_bpf_floor": dict(hbm_pressure_permille=0, slack_ns=0,
                                  estimated_transfer_ns=0, recompute_ns=0,
                                  speculative_recomputable=False),
            "gds_defer_native": dict(hbm_pressure_permille=801, slack_ns=10_000_000,
                                     estimated_transfer_ns=0, recompute_ns=0,
                                     speculative_recomputable=False),
            "gds_full_native": dict(hbm_pressure_permille=801, slack_ns=10_000_000,
                                    estimated_transfer_ns=5_000_000,
                                    recompute_ns=1_000_000,
                                    speculative_recomputable=True),
            "gds_full_bpf": dict(hbm_pressure_permille=801, slack_ns=10_000_000,
                                 estimated_transfer_ns=5_000_000,
                                 recompute_ns=1_000_000,
                                 speculative_recomputable=True),
        }
        for arm, fields in intended.items():
            with self.subTest(arm=arm):
                with mock.patch.object(runner, "_BASE_SERVER_ENVIRONMENT",
                                       return_value=BASE_ENV.copy()):
                    env = runner.gds_server_environment(
                        arm, Path("/cache"), "575.57.08", 256)
                telemetry = gds_adapter.EnvironmentRequestProvider.from_environ(
                    env).telemetry
                for name, value in fields.items():
                    self.assertEqual(getattr(telemetry, name), value, name)

    def test_native_decisions_match_each_arm_intent(self):
        backend = _FakeBackend()

        def env_for(arm):
            with mock.patch.object(runner, "_BASE_SERVER_ENVIRONMENT",
                                   return_value=BASE_ENV.copy()):
                return runner.gds_server_environment(
                    arm, Path("/cache"), "575.57.08", 256)

        for arm in ("gds_fifo", "gds_bpf_floor"):
            with self.subTest(arm=arm):
                provider = gds_adapter.EnvironmentRequestProvider.from_environ(
                    env_for(arm))
                read = provider(gds_adapter.READ_SPECULATIVE, "k", None, backend)
                write = provider(gds_adapter.WRITE, "k", None, backend)
                for request in (read, write):
                    decision = gds_policy.native_decide(request)
                    self.assertEqual(decision.action,
                                     gds_policy.ACTION_SUBMIT_NOW, arm)
                    self.assertEqual(decision.defer_ns, 0, arm)

        provider = gds_adapter.EnvironmentRequestProvider.from_environ(
            env_for("gds_defer_native"))
        defer_read = provider(gds_adapter.READ_SPECULATIVE, "k", None, backend)
        self.assertFalse(defer_read.flags & gds_policy.FLAG_RECOMPUTABLE)
        read_decision = gds_policy.native_decide(defer_read)
        self.assertEqual(read_decision.action, gds_policy.ACTION_DEFER)
        self.assertEqual(read_decision.defer_ns, 10_000_000)
        write_decision = gds_policy.native_decide(
            provider(gds_adapter.WRITE, "k", None, backend))
        self.assertEqual(write_decision.action, gds_policy.ACTION_DEFER)
        demand = gds_policy.native_decide(
            provider(gds_adapter.READ_DEMAND, "k", None, backend))
        self.assertEqual(demand.action, gds_policy.ACTION_SUBMIT_NOW)

        provider = gds_adapter.EnvironmentRequestProvider.from_environ(
            env_for("gds_full_native"))
        full_read = provider(gds_adapter.READ_SPECULATIVE, "k", None, backend)
        self.assertTrue(full_read.flags & gds_policy.FLAG_RECOMPUTABLE)
        full_read_decision = gds_policy.native_decide(full_read)
        self.assertEqual(full_read_decision.action, gds_policy.ACTION_RECOMPUTE)
        full_write_decision = gds_policy.native_decide(
            provider(gds_adapter.WRITE, "k", None, backend))
        self.assertEqual(full_write_decision.action, gds_policy.ACTION_DEFER)


class RotationTests(unittest.TestCase):
    def test_five_rotations_are_complete_and_position_balanced(self):
        orders = runner.rotation_orders(5)
        self.assertEqual(len(orders), 5)
        for order in orders:
            self.assertEqual(sorted(order), sorted(runner.CONFIGS))
        for position in range(5):
            self.assertEqual({order[position] for order in orders},
                             set(runner.CONFIGS))

    def test_single_block_is_the_reference_order(self):
        self.assertEqual(runner.rotation_orders(1), [list(runner.CONFIGS)])

    def test_blocks_below_one_are_rejected(self):
        with self.assertRaises(ValueError):
            runner.rotation_orders(0)


class SharedResourceTests(unittest.TestCase):
    def test_run_cell_injects_shared_kv_bytes_into_every_arm(self):
        calls: list[tuple[str, int | None]] = []

        def fake_start_server(config, model_path, cache_dir, port, log_path,
                              trace_dir=None,
                              expected_driver=runner.ops.EXPECTED_DRIVER,
                              kv_cache_memory_bytes=None, **options):
            calls.append((config, kv_cache_memory_bytes))
            return ("proc", "log-file", ["argv"], ["launch"])

        def fake_perf_run_cell(config, block, position, run_dir, port, model_path,
                               prefixes, expected_driver, store_barrier_timeout_s):
            runner.ops.start_server(config, model_path, run_dir / "cache", port,
                                    run_dir / "server.log",
                                    expected_driver=expected_driver)
            return {"config": config}

        with tempfile.TemporaryDirectory() as tmp:
            for index, arm in enumerate(runner.CONFIGS):
                run_dir = Path(tmp) / f"position-{index}-{arm}"
                run_dir.mkdir()
                with mock.patch.object(runner.ops, "start_server",
                                       fake_start_server), \
                        mock.patch.object(runner.perf, "run_cell",
                                          fake_perf_run_cell):
                    runner.run_cell(arm, 0, index, run_dir, 18080, Path("/model"),
                                    [], "575.57.08", 120.0, 256, 805306368)
        self.assertEqual([entry[0] for entry in calls], list(runner.CONFIGS))
        for _arm, kv_bytes in calls:
            self.assertEqual(kv_bytes, 805306368)


class CampaignMachineryTests(unittest.TestCase):
    def _run_single_block(self, tmp, failing_arm=None):
        calls: list[tuple[str, int | None, dict[str, str]]] = []

        def fake_start_server(config, model_path, cache_dir, port, log_path,
                              trace_dir=None,
                              expected_driver=runner.ops.EXPECTED_DRIVER,
                              kv_cache_memory_bytes=None, **options):
            calls.append((config, kv_cache_memory_bytes,
                          runner.ops.server_environment(config, cache_dir,
                                                        expected_driver)))
            return ("proc", "log-file", ["argv"], ["launch"])

        def fake_perf_run_cell(config, block, position, run_dir, port, model_path,
                               prefixes, expected_driver, store_barrier_timeout_s):
            runner.ops.start_server(config, model_path, run_dir / "cache", port,
                                    run_dir / "server.log",
                                    expected_driver=expected_driver)
            if failing_arm is not None and config == failing_arm:
                raise RuntimeError("simulated warm-phase failure")
            return {
                "schema": 1, "kind": runner.KIND, "config": config, "block": block,
                "position": position, "port": port, "ready": True,
                "requests": [], "barriers": [], "cleanup_errors": [],
                "server_returncode": 0, "error": None,
                "warm_phase": {
                    "warm_ttft_median_ms": 2.0, "requests_per_s": 10.0,
                    "output_tokens_per_s": 100.0, "requests": 8, "failures": 0,
                },
            }

        args = runner.parse_args(["--blocks", "1",
                                  "--output", str(Path(tmp) / "campaign")])
        stdout = io.StringIO()
        with mock.patch.object(runner.ops, "resolve_model",
                               return_value=Path("/model")), \
                mock.patch.object(runner.ops, "start_server", fake_start_server), \
                mock.patch.object(runner.perf, "run_cell", fake_perf_run_cell), \
                mock.patch("sys.stdout", stdout):
            status = runner.run_campaign(args)
        self.assertEqual(status, 0 if failing_arm is None else 2)
        return calls, stdout

    def test_campaign_records_raw_and_metrics_through_existing_machinery(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, stdout = self._run_single_block(tmp)
            root = Path(tmp) / "campaign"
            self.assertEqual([entry[0] for entry in calls], list(runner.CONFIGS))
            for config, kv_bytes, env in calls:
                self.assertEqual(kv_bytes, 805306368)
                self.assertEqual(env["LMCACHE_GDS_BUFFER_SIZE"], "256")
                self.assertEqual(env["LMCACHE_GDS_BACKEND"], "cufile")
                self.assertEqual(env["LMCACHE_GDS_POLICY_MODE"],
                                 runner.POLICY_MODES[config])
                self.assertEqual(
                    {key: value for key, value in env.items()
                     if key.startswith("LMCACHE_GDS_POLICY_")},
                    {"LMCACHE_GDS_POLICY_MODE": runner.POLICY_MODES[config],
                     **runner.POLICY_ENVIRONMENTS[config]})
            raw_lines = [json.loads(line)
                         for line in (root / "raw.jsonl").read_text().splitlines()]
            self.assertEqual([line["config"] for line in raw_lines],
                             list(runner.CONFIGS))
            for line in raw_lines:
                self.assertEqual(line["kind"], runner.KIND)
                self.assertEqual(line["port"], 18080)
            campaign = json.loads((root / "campaign.json").read_text())
            self.assertEqual(campaign["kind"], runner.KIND)
            self.assertEqual(campaign["params"]["kv_cache_memory_bytes"],
                             805306368)
            self.assertEqual(campaign["params"]["gds_buffer_size_mib"], 256)
            self.assertEqual(campaign["params"]["expected_driver"], "575.57.08")
            self.assertEqual(campaign["params"]["retry"], False)
            self.assertEqual(campaign["params"]["attempts_per_cell"], 1)
            self.assertEqual([cell["config"] for cell in campaign["cells"]],
                             list(runner.CONFIGS))
            self.assertEqual(
                [cell["metrics"]["warm_requests_per_s"]
                 for cell in campaign["cells"]],
                [10.0] * len(runner.CONFIGS))
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["cells_attempted"], len(runner.CONFIGS))
            for arm in runner.CONFIGS:
                self.assertEqual(summary["per_arm"][arm]["cells_attempted"], 1)
                self.assertEqual(
                    summary["per_arm"][arm]["medians"]["warm_requests_per_s"], 10.0)
            for arm in runner.CONFIGS:
                result = json.loads((root / "block-00"
                                     / f"position-{runner.CONFIGS.index(arm)}-{arm}"
                                     / "result.json").read_text())
                self.assertEqual(result["kind"], runner.KIND)
                self.assertEqual(result["config"], arm)
            self.assertIn(f'"kind": "{runner.KIND}"', stdout.getvalue())

    def test_campaign_preserves_a_failed_arm_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls, _ = self._run_single_block(tmp, failing_arm="gds_defer_native")
            root = Path(tmp) / "campaign"
            self.assertEqual([entry[0] for entry in calls], list(runner.CONFIGS))
            raw_lines = [json.loads(line)
                         for line in (root / "raw.jsonl").read_text().splitlines()]
            self.assertEqual(len(raw_lines), len(runner.CONFIGS))
            failed = next(line for line in raw_lines
                          if line["config"] == "gds_defer_native")
            self.assertIn("simulated warm-phase failure", failed["error"])
            self.assertFalse(failed["ready"])
            campaign = json.loads((root / "campaign.json").read_text())
            self.assertIsNone(
                campaign["cells"][runner.CONFIGS.index("gds_defer_native")]["metrics"]
                ["warm_requests_per_s"])


class DryRunAndGateTests(unittest.TestCase):
    def test_dry_run_does_not_create_output_and_reports_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "unused"
            stdout = io.StringIO()
            with mock.patch("sys.stdout", stdout):
                status = runner.main(["--dry-run", "--output", str(output)])
            self.assertEqual(status, 0)
            self.assertFalse(output.exists())
            plan = json.loads(stdout.getvalue())
            self.assertEqual(plan["kind"], runner.KIND)
            self.assertEqual(plan["configs"], list(runner.CONFIGS))
            self.assertEqual(plan["blocks"], 5)
            self.assertEqual(plan["block_orders"], runner.rotation_orders(5))
            self.assertEqual(plan["expected_driver_parameter"], "575.57.08")
            self.assertEqual(plan["port"], 18080)
            self.assertEqual(plan["kv_cache_memory_bytes"], 805306368)
            self.assertEqual(plan["gds"]["buffer_size_mib"], 256)
            self.assertEqual(plan["gds"]["backend"], "cufile")
            self.assertEqual(plan["gds"]["policy_modes"], runner.POLICY_MODES)
            self.assertEqual(plan["gds"]["policy_environment"],
                             runner.POLICY_ENVIRONMENTS)

    def test_plan_declares_no_gates_no_retries_single_attempt(self):
        plan = runner.dry_run_plan(runner.parse_args(["--dry-run"]))
        self.assertEqual(plan["gates"], [])
        self.assertIs(plan["retries"], False)
        self.assertEqual(plan["attempts_per_cell"], 1)

    def test_module_exposes_no_gate_mechanisms(self):
        for name in ("validate_log", "compare_outputs", "wait_for_cold_store",
                     "sync_and_verify_disk", "wait_gpu_idle", "admission",
                     "validate_driver", "run_preflight"):
            self.assertFalse(callable(getattr(runner, name, None)), name)

    def test_summary_uses_numeric_cell_medians(self):
        cells = []
        for index, value in enumerate((3.0, 1.0, 2.0)):
            cells.append({
                "config": "gds_full_native",
                "metrics": {
                    "warm_ttft_median_ms": value,
                    "warm_requests_per_s": 10.0 + value,
                    "warm_output_tokens_per_s": 100.0 + value,
                },
            })
        summary = runner.median_summary(cells)
        full_native = summary["per_arm"]["gds_full_native"]
        self.assertEqual(full_native["cells_attempted"], 3)
        self.assertEqual(full_native["cells_measured"], 3)
        self.assertEqual(full_native["medians"]["warm_ttft_median_ms"], 2.0)
        self.assertIsNone(
            summary["per_arm"]["gds_fifo"]["medians"]["warm_requests_per_s"])

    def test_jsonl_is_one_raw_object_per_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.jsonl"
            with path.open("x", encoding="utf-8") as stream:
                runner.write_jsonl_record(stream,
                                          {"config": "gds_fifo", "value": 1})
                runner.write_jsonl_record(stream,
                                          {"config": "gds_full_bpf", "value": 2})
            records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([record["config"] for record in records],
                         ["gds_fifo", "gds_full_bpf"])


class DefaultsTests(unittest.TestCase):
    def test_defaults_match_the_five_arm_campaign(self):
        args = runner.parse_args([])
        self.assertEqual(args.expected_driver, "575.57.08")
        self.assertEqual(args.kv_cache_memory_bytes, 805306368)
        self.assertEqual(args.gds_buffer_size_mib, 256)
        self.assertEqual(args.blocks, 5)
        self.assertEqual(args.port, 18080)
        self.assertEqual(args.store_barrier_timeout_s, 120.0)

    def test_kv_cache_memory_bytes_must_be_positive(self):
        with self.assertRaises(SystemExit):
            runner.parse_args(["--kv-cache-memory-bytes", "0"])

    def test_gds_buffer_size_mib_must_be_positive(self):
        with self.assertRaises(SystemExit):
            runner.parse_args(["--gds-buffer-size-mib", "0"])


if __name__ == "__main__":
    unittest.main()
