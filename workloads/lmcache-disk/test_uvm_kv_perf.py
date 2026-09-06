#!/usr/bin/env python3
"""CPU-only tests for the UVM KV performance-only runner (no GPU, no server)."""

import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from unittest import mock
import subprocess
import tempfile
import time
import unittest


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE / "run_uvm_kv_perf.py"
SPEC = importlib.util.spec_from_file_location("run_uvm_kv_perf", MODULE_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
ops = runner.ops

EXPECTED_DRIVER = "610.43.02"
CACHE_DIR = "/tmp/uvm-kv-cache-fixture"


class RunnerShapeTests(unittest.TestCase):
    def test_module_constants(self):
        self.assertEqual(runner.KIND, "lmcache_uvm_kv_perf")
        self.assertEqual(runner.CONFIGS,
                         ("recompute", "lmcache_disk", "lmcache_disk_uvm_kv",
                          "lmcache_disk_uvm_kv_gpubpf_debt"))
        self.assertEqual(runner.LOADER_ARM, "lmcache_disk_uvm_kv_gpubpf_debt")
        self.assertEqual(runner.CONFIGS[-1], runner.LOADER_ARM)
        self.assertEqual(runner.DEFAULT_BLOCKS, 1)
        self.assertEqual(runner.DEFAULT_PORT, 18080)
        self.assertEqual(runner.DEFAULT_STORE_BARRIER_TIMEOUT_S, 120.0)
        self.assertEqual(runner.REQUEST_LABELS, ("cold", "warm"))

    def test_parse_args_defaults(self):
        args = runner.parse_args([])
        self.assertEqual(args.blocks, 1)
        self.assertEqual(args.port, 18080)
        self.assertEqual(args.store_barrier_timeout_s, 120.0)
        self.assertEqual(args.expected_driver, EXPECTED_DRIVER)
        self.assertIs(args.dry_run, False)
        self.assertIsNone(args.output)
        self.assertEqual(args.eviction_loader, ops.EVICTION_LOADER)
        self.assertEqual(args.gpu_memory_utilization, 0.98)
        self.assertEqual(args.cpu_offload_gb, 0.0)

    def test_parse_args_eviction_loader_override(self):
        args = runner.parse_args(["--eviction-loader", "/tmp/eviction_debt"])
        self.assertEqual(args.eviction_loader, Path("/tmp/eviction_debt"))

    def test_no_schedule_or_legacy_arm_in_source(self):
        src = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("SCHEDULE", src)
        self.assertNotIn("schedule.json", src)
        self.assertNotIn("ops.CONFIGS", src)
        self.assertNotIn("lmcache_cpu", src)


class RotationTests(unittest.TestCase):
    def test_default_one_block_is_the_base_order(self):
        self.assertEqual(runner.rotation_orders(1), [list(runner.CONFIGS)])

    def test_every_order_is_a_complete_four_arm_permutation(self):
        for order in runner.rotation_orders(4):
            self.assertEqual(sorted(order), sorted(runner.CONFIGS))
            self.assertEqual(len(order), 4)

    def test_four_blocks_cover_each_arm_in_each_position_once(self):
        orders = runner.rotation_orders(4)
        for position in range(4):
            column = [order[position] for order in orders]
            self.assertEqual(sorted(column), sorted(runner.CONFIGS))

    def test_rotation_is_deterministic_and_periodic(self):
        self.assertEqual(runner.rotation_orders(4), runner.rotation_orders(4))
        self.assertEqual(runner.rotation_orders(8)[4], runner.rotation_orders(4)[0])
        self.assertEqual(runner.rotation_orders(8)[5], runner.rotation_orders(4)[1])

    def test_blocks_below_one_rejected(self):
        with self.assertRaises(ValueError):
            runner.rotation_orders(0)


class ConfigSelectionTests(unittest.TestCase):
    def test_parse_args_configs_default_is_all_four_in_base_order(self):
        args = runner.parse_args([])
        self.assertEqual(args.configs, runner.CONFIGS)

    def test_parse_args_selects_the_loader_arm_alone(self):
        args = runner.parse_args(["--configs", "lmcache_disk_uvm_kv_gpubpf_debt"])
        self.assertEqual(args.configs, ("lmcache_disk_uvm_kv_gpubpf_debt",))

    def test_parse_args_rejects_unknown_config(self):
        with self.assertRaises(ValueError):
            runner.parse_args(["--configs", "recompute,nosuch_arm"])

    def test_parse_args_rejects_duplicate_config(self):
        with self.assertRaises(ValueError):
            runner.parse_args(["--configs", "recompute,lmcache_disk,recompute"])

    def test_parse_args_rejects_empty_config_names(self):
        for value in ("", "recompute,", ",recompute", "recompute,,lmcache_disk"):
            with self.assertRaises(ValueError):
                runner.parse_args(["--configs", value])

    def test_rotation_orders_one_arm_repeats_it_every_block(self):
        configs = ("lmcache_disk_uvm_kv_gpubpf_debt",)
        orders = runner.rotation_orders(3, configs)
        self.assertEqual(orders, [["lmcache_disk_uvm_kv_gpubpf_debt"]] * 3)
        self.assertEqual(len(orders) * len(configs), 3)
        for order in orders:
            self.assertEqual(order, list(configs))

    def test_rotation_orders_subset_rotates_only_selected_arms(self):
        subset = ("lmcache_disk_uvm_kv_gpubpf_debt", "recompute")
        self.assertEqual(runner.rotation_orders(2, subset),
                         [["lmcache_disk_uvm_kv_gpubpf_debt", "recompute"],
                          ["recompute", "lmcache_disk_uvm_kv_gpubpf_debt"]])

    def test_rotation_orders_configs_default_matches_all_four(self):
        self.assertEqual(runner.rotation_orders(3),
                         runner.rotation_orders(3, runner.CONFIGS))

    def test_arm_summary_iterates_only_selected_configs(self):
        cells = [{"config": "lmcache_disk_uvm_kv_gpubpf_debt",
                  "warm": {"warm_ttft_median_ms": 12.5}}]
        summary = runner.arm_summary(cells, ("lmcache_disk_uvm_kv_gpubpf_debt",))
        self.assertEqual(list(summary), ["lmcache_disk_uvm_kv_gpubpf_debt"])
        self.assertEqual(summary["lmcache_disk_uvm_kv_gpubpf_debt"]["cell_count"], 1)
        self.assertEqual(summary["lmcache_disk_uvm_kv_gpubpf_debt"]["warm_ttft_median_ms"],
                         [12.5])


class EnvironmentTests(unittest.TestCase):
    def test_uvm_arm_is_disk_arm_plus_plugin_vars(self):
        disk = ops.server_environment("lmcache_disk", Path(CACHE_DIR))
        uvm = ops.server_environment("lmcache_disk_uvm_kv", Path(CACHE_DIR))
        lmc_keys = [key for key in disk if key.startswith("LMCACHE_")]
        self.assertTrue(lmc_keys)
        for key in lmc_keys:
            self.assertEqual(uvm[key], disk[key])
        self.assertEqual(set(uvm) - set(disk), {"UVM_KV_PLUGIN", "UVM_KV_PLUGIN_SO"})
        self.assertEqual(uvm["UVM_KV_PLUGIN"], "1")

    def test_uvm_plugin_so_is_absolute_prepared_allocator(self):
        uvm = ops.server_environment("lmcache_disk_uvm_kv", Path(CACHE_DIR))
        so = uvm["UVM_KV_PLUGIN_SO"]
        self.assertTrue(so.startswith("/"))
        self.assertEqual(Path(so), ops.UVM_KV_ALLOCATOR_SO)

    def test_allocator_constant_points_at_prepared_symlink_target(self):
        self.assertEqual(ops.UVM_KV_ALLOCATOR_SO.name, "uvm_allocator.so")
        self.assertTrue(ops.UVM_KV_ALLOCATOR_SO.is_absolute())
        self.assertEqual(ops.UVM_KV_ALLOCATOR_SO.parent,
                         ops.VLLM_WORKLOAD / "vllm" / "uvm_test")
        self.assertTrue(ops.UVM_KV_ALLOCATOR_SO.is_file())

    def test_other_arms_do_not_enable_plugin(self):
        for config in ("recompute", "lmcache_disk"):
            env = ops.server_environment(config, Path(CACHE_DIR))
            self.assertNotIn("UVM_KV_PLUGIN", env)
            self.assertNotIn("UVM_KV_PLUGIN_SO", env)

    def test_recompute_has_no_lmcache_or_uvm_env(self):
        env = ops.server_environment("recompute", Path(CACHE_DIR))
        self.assertFalse([key for key in env if key.startswith("LMCACHE_")])
        self.assertNotIn("UVM_KV_PLUGIN", env)

    def test_disk_arm_env_matches_lmcache_disk_runner_expectations(self):
        env = ops.server_environment("lmcache_disk_uvm_kv", Path(CACHE_DIR))
        self.assertEqual(env["LMCACHE_LOCAL_CPU"], "False")
        self.assertEqual(env["LMCACHE_LOCAL_DISK"], "file://" + CACHE_DIR)
        self.assertEqual(env["LMCACHE_USE_GPU_CONNECTOR_V3"], "True")
        self.assertIn("use_odirect", env["LMCACHE_EXTRA_CONFIG"])

    def test_debt_arm_env_is_identical_to_the_uvm_arm(self):
        uvm = ops.server_environment("lmcache_disk_uvm_kv", Path(CACHE_DIR))
        debt = ops.server_environment("lmcache_disk_uvm_kv_gpubpf_debt", Path(CACHE_DIR))
        self.assertEqual(debt, uvm)
        self.assertEqual(debt["UVM_KV_PLUGIN"], "1")
        self.assertEqual(Path(debt["UVM_KV_PLUGIN_SO"]), ops.UVM_KV_ALLOCATOR_SO)


class LoaderPrimitiveTests(unittest.TestCase):
    def test_eviction_loader_default_is_the_repo_extension_path(self):
        self.assertEqual(ops.EVICTION_LOADER, ops.GPU_EXT / "extension" / "eviction_debt")
        self.assertEqual(ops.EVICTION_LOADER.name, "eviction_debt")
        self.assertEqual(ops.EVICTION_LOADER.parent, ops.GPU_EXT / "extension")
        self.assertTrue(ops.EVICTION_LOADER.is_absolute())

    def test_server_argv_gives_the_debt_arm_the_lmcache_connector(self):
        argv = ops.server_argv("lmcache_disk_uvm_kv_gpubpf_debt", Path("/model"), 18080)
        joined = " ".join(argv)
        self.assertIn("LMCacheConnectorV1", joined)
        self.assertIn("kv_both", joined)

    def test_start_eviction_loader_runs_sudo_n_with_allocator_and_w0_new_session(self):
        captured = {}

        class FakeProc:
            pid = 4242

            def poll(self):
                return None

        def fake_popen(launch, **kwargs):
            captured["launch"] = launch
            captured["kwargs"] = kwargs
            return FakeProc()

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "loader.log"
            with mock.patch.object(Path, "is_file", return_value=True), \
                    mock.patch.object(ops.subprocess, "Popen", fake_popen):
                proc, log_file, argv, launch = ops.start_eviction_loader(
                    ops.EVICTION_LOADER, log_path)
            self.assertEqual(argv, [str(ops.EVICTION_LOADER), "-a",
                                    str(ops.UVM_KV_ALLOCATOR_SO), "-w", "0"])
            self.assertEqual(launch, ["/usr/bin/sudo", "-n", "/usr/bin/stdbuf", "-oL", "-eL",
                                      str(ops.EVICTION_LOADER), "-a",
                                      str(ops.UVM_KV_ALLOCATOR_SO), "-w", "0"])
            self.assertIs(captured["kwargs"]["stdin"], subprocess.PIPE)
            self.assertIs(captured["kwargs"]["stderr"], subprocess.STDOUT)
            self.assertIsInstance(captured["kwargs"]["stdout"], io.TextIOBase)
            self.assertIs(captured["kwargs"]["start_new_session"], True)
            self.assertEqual(captured["kwargs"]["env"]["PATH"],
                             "/usr/local/cuda-12.9/bin:/usr/bin:/bin")
            self.assertEqual(captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"], "")
            log_file.close()
            self.assertTrue(log_path.is_file())

    def test_start_eviction_loader_accepts_explicit_allocator(self):
        captured = {}

        class FakeProc:
            pid = 4242

            def poll(self):
                return None

        def fake_popen(launch, **kwargs):
            captured["launch"] = launch
            return FakeProc()

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "loader.log"
            allocator = Path("/tmp/other-uvm-allocator.so")
            with mock.patch.object(Path, "is_file", return_value=True), \
                    mock.patch.object(ops.subprocess, "Popen", fake_popen):
                proc, log_file, argv, launch = ops.start_eviction_loader(
                    ops.EVICTION_LOADER, log_path, allocator)
            self.assertEqual(argv, [str(ops.EVICTION_LOADER), "-a",
                                    str(allocator), "-w", "0"])
            self.assertEqual(captured["launch"][-4:], ["-a", str(allocator), "-w", "0"])
            log_file.close()

    def test_start_eviction_loader_missing_binary_is_a_recordable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "loader.log"
            with mock.patch.object(Path, "is_file", return_value=False):
                with self.assertRaises(FileNotFoundError):
                    ops.start_eviction_loader(ops.EVICTION_LOADER, log_path)
        self.assertFalse(log_path.exists())

    def test_start_eviction_loader_missing_allocator_is_a_recordable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            loader = Path(tmp) / "eviction_debt"
            loader.write_bytes(b"")
            log_path = Path(tmp) / "loader.log"
            missing_allocator = Path(tmp) / "missing-uvm-allocator.so"
            with self.assertRaises(FileNotFoundError):
                ops.start_eviction_loader(loader, log_path, missing_allocator)
        self.assertFalse(log_path.exists())

    def test_stop_eviction_loader_is_a_bounded_process_group_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "loader.log"
            log_file = log_path.open("w")
            proc = subprocess.Popen(["sleep", "30"], start_new_session=True,
                                    stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            try:
                started = time.monotonic()
                returncode, errors = ops.stop_eviction_loader(proc, log_file)
                elapsed = time.monotonic() - started
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            self.assertEqual(errors, [])
            self.assertIsNotNone(returncode)
            self.assertIsNotNone(proc.poll())
            self.assertLess(elapsed, 20)

    def test_stop_eviction_loader_closes_the_log_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "loader.log"
            log_file = log_path.open("w")
            proc = subprocess.Popen(["true"], start_new_session=True,
                                    stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            proc.wait()
            returncode, errors = ops.stop_eviction_loader(proc, log_file)
            self.assertEqual(errors, [])
            self.assertEqual(returncode, 0)
            self.assertTrue(log_file.closed)


class LoaderWarmSignalTests(unittest.TestCase):
    class FakeLoader:
        def __init__(self, returncode=None, stdin=None):
            self.returncode = returncode
            self.stdin = stdin

        def poll(self):
            return self.returncode

    def test_warm_key_written_and_flushed_while_loader_alive(self):
        buf = io.StringIO()
        outcome = runner.send_loader_warm_signal(self.FakeLoader(stdin=buf))
        self.assertEqual(outcome, {"key": "w", "written": "w\n", "sent": True})
        self.assertEqual(buf.getvalue(), "w\n")

    def test_warm_key_not_sent_when_loader_has_exited(self):
        outcome = runner.send_loader_warm_signal(self.FakeLoader(returncode=0))
        self.assertIs(outcome["sent"], False)
        self.assertIn("return code 0", outcome["error"])

    def test_warm_key_write_failure_is_recorded_not_raised(self):
        class BrokenStdin:
            def write(self, text):
                raise OSError("broken pipe")

            def flush(self):
                pass

        outcome = runner.send_loader_warm_signal(self.FakeLoader(stdin=BrokenStdin()))
        self.assertIs(outcome["sent"], False)
        self.assertEqual(outcome["error"], "OSError: broken pipe")

    def test_warm_key_not_sent_when_stdin_is_none(self):
        outcome = runner.send_loader_warm_signal(self.FakeLoader(stdin=None))
        self.assertIs(outcome["sent"], False)
        self.assertIn("stdin", outcome["error"])


class LoaderReadinessTests(unittest.TestCase):
    MARKER = "Successfully loaded migration-debt eviction policy!"

    class FakeProc:
        def __init__(self, returncode=None):
            self.returncode = returncode

        def poll(self):
            return self.returncode

    def test_marker_constant_is_exact(self):
        self.assertEqual(ops.EVICTION_LOADER_READY_MARKER, self.MARKER)

    def test_returns_once_the_exact_marker_is_in_the_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "loader.log"
            log_path.write_text(f"boot lines\n{self.MARKER}\n", encoding="utf-8")
            started = time.monotonic()
            ops.wait_eviction_loader_ready(self.FakeProc(), log_path, timeout=5)
            self.assertLess(time.monotonic() - started, 5)

    def test_partial_marker_does_not_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "loader.log"
            log_path.write_text("Successfully loaded migration-debt eviction polic\n",
                                encoding="utf-8")
            with self.assertRaises(ops.GateError):
                ops.wait_eviction_loader_ready(self.FakeProc(), log_path, timeout=0.5)

    def test_marker_written_after_a_delay_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "loader.log"
            log_path.write_text("starting\n", encoding="utf-8")

            def late_marker():
                time.sleep(0.3)
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(self.MARKER + "\n")
                return None

            proc = self.FakeProc()
            with mock.patch.object(proc, "poll", side_effect=late_marker):
                ops.wait_eviction_loader_ready(proc, log_path, timeout=10)
            self.assertIn(self.MARKER, log_path.read_text(encoding="utf-8"))

    def test_loader_exit_before_marker_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "loader.log"
            log_path.write_text("starting\n", encoding="utf-8")
            with self.assertRaises(ops.GateError) as ctx:
                ops.wait_eviction_loader_ready(self.FakeProc(returncode=1), log_path,
                                               timeout=5)
            self.assertIn("exited 1", str(ctx.exception))

    def test_readiness_timeout_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "loader.log"
            log_path.write_text("starting\n", encoding="utf-8")
            with self.assertRaises(ops.GateError) as ctx:
                ops.wait_eviction_loader_ready(self.FakeProc(), log_path, timeout=0.5)
            self.assertIn("timeout", str(ctx.exception))


class LoaderFailureSkipsServerTests(unittest.TestCase):
    PREFIXES = [{"index": 0, "cold_token_ids": [1], "warm_token_ids": [2],
                 "expected_store_tokens": 256}]

    def test_run_cell_records_loader_error_and_skips_server_when_start_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "cell"
            with mock.patch.object(ops, "start_eviction_loader",
                                   side_effect=FileNotFoundError("missing loader")), \
                    mock.patch.object(ops, "start_server") as start_server:
                record = runner.run_cell(runner.LOADER_ARM, 0, 0, run_dir, 18080,
                                         Path("/model"), self.PREFIXES,
                                         EXPECTED_DRIVER, 120.0)
            start_server.assert_not_called()
            self.assertIs(record["ready"], False)
            self.assertIs(record["loader"]["enabled"], True)
            self.assertIs(record["loader"]["ready"], False)
            self.assertIn("FileNotFoundError", record["loader"]["error"])
            self.assertIn("eviction loader failed", record["error"])
            self.assertEqual(record["requests"], [])
            self.assertTrue((run_dir / "result.json").is_file())

    def test_run_cell_records_loader_error_and_skips_server_when_readiness_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "cell"
            run_dir.mkdir()
            log_file = (run_dir / "loader.log").open("w")
            try:
                with mock.patch.object(
                        ops, "start_eviction_loader",
                        return_value=(mock.Mock(), log_file,
                                       ["loader", "-a", str(ops.UVM_KV_ALLOCATOR_SO),
                                        "-w", "0"],
                                       ["/usr/bin/sudo", "-n", "/usr/bin/stdbuf",
                                        "-oL", "-eL", "loader", "-a",
                                        str(ops.UVM_KV_ALLOCATOR_SO), "-w", "0"])), \
                        mock.patch.object(ops, "wait_eviction_loader_ready",
                                          side_effect=ops.GateError("readiness timeout")), \
                        mock.patch.object(ops, "start_server") as start_server, \
                        mock.patch.object(ops, "stop_eviction_loader",
                                          return_value=(1, [])) as stop_loader:
                    record = runner.run_cell(runner.LOADER_ARM, 0, 0, run_dir, 18080,
                                             Path("/model"), self.PREFIXES,
                                             EXPECTED_DRIVER, 120.0)
                start_server.assert_not_called()
                stop_loader.assert_called_once()
                self.assertIs(record["ready"], False)
                self.assertIs(record["loader"]["ready"], False)
                self.assertIn("GateError", record["loader"]["error"])
                self.assertIn("eviction loader failed", record["error"])
                self.assertEqual(record["requests"], [])
            finally:
                log_file.close()


class PressureCliTests(unittest.TestCase):
    def test_parse_args_pressure_defaults_disabled(self):
        args = runner.parse_args([])
        self.assertIsNone(args.kv_cache_memory_bytes)
        self.assertEqual(args.pressure_gib, 0)
        self.assertEqual(args.pressure_passes, 1)
        self.assertEqual(args.pressure_pause_ms, 0)
        self.assertEqual(args.pressure_binary, ops.PRESSURE_TENANT)

    def test_parse_args_pressure_options(self):
        args = runner.parse_args(["--kv-cache-memory-bytes", "48318382080",
                                  "--pressure-gib", "2", "--pressure-passes", "3",
                                  "--pressure-pause-ms", "250",
                                  "--pressure-binary", "/tmp/uvm_fault_stream"])
        self.assertEqual(args.kv_cache_memory_bytes, 48318382080)
        self.assertEqual(args.pressure_gib, 2)
        self.assertEqual(args.pressure_passes, 3)
        self.assertEqual(args.pressure_pause_ms, 250)
        self.assertEqual(args.pressure_binary, Path("/tmp/uvm_fault_stream"))

    def test_kv_cache_memory_bytes_must_be_positive(self):
        for value in ("0", "-8"):
            with self.assertRaises(ValueError):
                runner.parse_args(["--kv-cache-memory-bytes", value])

    def test_pressure_gib_must_be_nonnegative(self):
        with self.assertRaises(ValueError):
            runner.parse_args(["--pressure-gib", "-1"])

    def test_pressure_passes_must_be_positive_when_enabled(self):
        with self.assertRaises(ValueError):
            runner.parse_args(["--pressure-gib", "2", "--pressure-passes", "0"])

    def test_pressure_pause_ms_must_be_nonnegative_when_enabled(self):
        with self.assertRaises(ValueError):
            runner.parse_args(["--pressure-gib", "2", "--pressure-pause-ms", "-1"])

    def test_disabled_pressure_ignores_passes_and_pause(self):
        args = runner.parse_args(["--pressure-gib", "0", "--pressure-passes", "0",
                                  "--pressure-pause-ms", "-5"])
        self.assertEqual(args.pressure_gib, 0)
        self.assertEqual(args.pressure_passes, 0)


class KvPoolArgvTests(unittest.TestCase):
    def test_server_argv_without_pool_bytes_is_unchanged(self):
        argv = ops.server_argv("lmcache_disk", Path("/model"), 18080)
        self.assertNotIn("--kv-cache-memory-bytes", argv)
        self.assertEqual(argv[argv.index("--port") + 1], "18080")

    def test_server_argv_appends_pool_bytes_when_set(self):
        argv = ops.server_argv("lmcache_disk_uvm_kv", Path("/model"), 18080,
                               kv_cache_memory_bytes=48 * 1024**3)
        self.assertEqual(argv[-2:], ["--kv-cache-memory-bytes", str(48 * 1024**3)])

    def test_server_argv_appends_pool_bytes_for_recompute_arm(self):
        argv = ops.server_argv("recompute", Path("/model"), 18080,
                               kv_cache_memory_bytes=1)
        self.assertEqual(argv[-2:], ["--kv-cache-memory-bytes", "1"])

    def test_start_server_forwards_pool_bytes_into_argv(self):
        captured = {}

        class FakeProc:
            pid = 1

            def poll(self):
                return None

        def fake_popen(launch, **kwargs):
            captured["launch"] = launch
            return FakeProc()

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "server.log"
            with mock.patch.object(ops.subprocess, "Popen", fake_popen):
                proc, log_file, argv, launch = ops.start_server(
                    "lmcache_disk", Path("/model"), Path(tmp), 18080, log_path,
                    kv_cache_memory_bytes=4096)
            self.assertEqual(argv[-2:], ["--kv-cache-memory-bytes", "4096"])
            self.assertEqual(launch[-2:], ["--kv-cache-memory-bytes", "4096"])
            log_file.close()


class GpuMemoryUtilizationTests(unittest.TestCase):
    def test_default_constant_is_098(self):
        self.assertEqual(ops.DEFAULT_GPU_MEMORY_UTILIZATION, 0.98)

    def test_parse_args_default_is_098(self):
        args = runner.parse_args([])
        self.assertEqual(args.gpu_memory_utilization, 0.98)

    def test_parse_args_override(self):
        args = runner.parse_args(["--gpu-memory-utilization", "0.9"])
        self.assertEqual(args.gpu_memory_utilization, 0.9)

    def test_parse_args_accepts_upper_boundary_one(self):
        args = runner.parse_args(["--gpu-memory-utilization", "1"])
        self.assertEqual(args.gpu_memory_utilization, 1.0)

    def test_parse_args_rejects_values_outside_the_open_closed_range(self):
        for value in ("0", "0.0", "-0.5", "1.01", "1.5", "2", "nan"):
            with self.assertRaises(ValueError):
                runner.parse_args(["--gpu-memory-utilization", value])

    def test_server_argv_default_gpu_memory_utilization_is_098(self):
        argv = ops.server_argv("lmcache_disk", Path("/model"), 18080)
        self.assertEqual(argv[argv.index("--gpu-memory-utilization") + 1], "0.98")

    def test_server_argv_overrides_gpu_memory_utilization(self):
        argv = ops.server_argv("lmcache_disk_uvm_kv", Path("/model"), 18080,
                               gpu_memory_utilization=0.95)
        self.assertEqual(argv[argv.index("--gpu-memory-utilization") + 1], "0.95")

    def test_server_argv_gpu_memory_utilization_for_recompute_arm(self):
        argv = ops.server_argv("recompute", Path("/model"), 18080,
                               gpu_memory_utilization=0.5)
        self.assertEqual(argv[argv.index("--gpu-memory-utilization") + 1], "0.5")

    def test_start_server_forwards_gpu_memory_utilization_into_argv(self):
        class FakeProc:
            pid = 1

            def poll(self):
                return None

        def fake_popen(launch, **kwargs):
            return FakeProc()

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "server.log"
            with mock.patch.object(ops.subprocess, "Popen", fake_popen):
                proc, log_file, argv, launch = ops.start_server(
                    "lmcache_disk", Path("/model"), Path(tmp), 18080, log_path,
                    gpu_memory_utilization=0.7)
            self.assertEqual(argv[argv.index("--gpu-memory-utilization") + 1], "0.7")
            self.assertEqual(launch[launch.index("--gpu-memory-utilization") + 1], "0.7")
            log_file.close()


class CpuOffloadGbTests(unittest.TestCase):
    def test_default_constant_is_zero(self):
        self.assertEqual(ops.DEFAULT_CPU_OFFLOAD_GB, 0.0)

    def test_parse_args_default_is_zero(self):
        args = runner.parse_args([])
        self.assertEqual(args.cpu_offload_gb, 0.0)

    def test_parse_args_override(self):
        args = runner.parse_args(["--cpu-offload-gb", "8"])
        self.assertEqual(args.cpu_offload_gb, 8.0)

    def test_parse_args_accepts_explicit_zero(self):
        args = runner.parse_args(["--cpu-offload-gb", "0"])
        self.assertEqual(args.cpu_offload_gb, 0.0)

    def test_parse_args_rejects_negative_and_non_finite_values(self):
        for value in ("-0.5", "-1", "inf", "nan"):
            with self.assertRaises(ValueError):
                runner.parse_args(["--cpu-offload-gb", value])
        with self.assertRaises(ValueError):
            runner.parse_args(["--cpu-offload-gb=-inf"])

    def test_server_argv_default_does_not_append_cpu_offload(self):
        argv = ops.server_argv("lmcache_disk", Path("/model"), 18080)
        self.assertNotIn("--cpu-offload-gb", argv)

    def test_server_argv_zero_does_not_append_cpu_offload(self):
        argv = ops.server_argv("lmcache_disk", Path("/model"), 18080,
                               cpu_offload_gb=0.0)
        self.assertNotIn("--cpu-offload-gb", argv)

    def test_server_argv_appends_cpu_offload_when_positive(self):
        argv = ops.server_argv("lmcache_disk_uvm_kv", Path("/model"), 18080,
                               cpu_offload_gb=8.0)
        self.assertEqual(argv[-2:], ["--cpu-offload-gb", "8.0"])

    def test_server_argv_cpu_offload_for_recompute_arm(self):
        argv = ops.server_argv("recompute", Path("/model"), 18080,
                               cpu_offload_gb=4.5)
        self.assertEqual(argv[argv.index("--cpu-offload-gb") + 1], "4.5")

    def test_start_server_forwards_cpu_offload_into_argv(self):
        class FakeProc:
            pid = 1

            def poll(self):
                return None

        def fake_popen(launch, **kwargs):
            return FakeProc()

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "server.log"
            with mock.patch.object(ops.subprocess, "Popen", fake_popen):
                proc, log_file, argv, launch = ops.start_server(
                    "lmcache_disk", Path("/model"), Path(tmp), 18080, log_path,
                    cpu_offload_gb=8.0)
            self.assertEqual(argv[-2:], ["--cpu-offload-gb", "8.0"])
            self.assertEqual(launch[-2:], ["--cpu-offload-gb", "8.0"])
            log_file.close()

    def test_start_server_default_cpu_offload_leaves_argv_unchanged(self):
        class FakeProc:
            pid = 1

            def poll(self):
                return None

        def fake_popen(launch, **kwargs):
            return FakeProc()

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "server.log"
            with mock.patch.object(ops.subprocess, "Popen", fake_popen):
                proc, log_file, argv, launch = ops.start_server(
                    "lmcache_disk", Path("/model"), Path(tmp), 18080, log_path)
            self.assertNotIn("--cpu-offload-gb", argv)
            self.assertNotIn("--cpu-offload-gb", launch)
            log_file.close()


class PressurePrimitiveTests(unittest.TestCase):
    def test_pressure_tenant_default_is_the_repo_workload_binary(self):
        self.assertEqual(ops.PRESSURE_TENANT,
                         ops.GPU_EXT / "workloads" / "uvm-policy-mechanism" /
                         "uvm_fault_stream")
        self.assertEqual(ops.PRESSURE_TENANT.name, "uvm_fault_stream")
        self.assertTrue(ops.PRESSURE_TENANT.is_absolute())

    def test_pressure_readiness_line_constants_are_exact(self):
        self.assertEqual(ops.PRESSURE_READY_LINE, "READY pid=")
        self.assertEqual(ops.PRESSURE_MONITOR_LINE, "MONITOR_PID:")

    def test_start_pressure_tenant_argv_and_process_flags(self):
        captured = {}

        class FakeProc:
            pid = 777

            def poll(self):
                return None

        def fake_popen(launch, **kwargs):
            captured["launch"] = launch
            captured["kwargs"] = kwargs
            return FakeProc()

        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "uvm_fault_stream"
            binary.write_bytes(b"\x7fELF-fixture")
            log_path = Path(tmp) / "pressure.log"
            result_path = Path(tmp) / "pressure-result.json"
            with mock.patch.object(ops.subprocess, "Popen", fake_popen):
                proc, log_file, argv = ops.start_pressure_tenant(
                    binary, 2, 3, 250, log_path, result_path)
            self.assertEqual(argv, [str(binary), "--gib", "2", "--passes", "3",
                                    "--pause-ms", "250", "--wait-for-monitor",
                                    "--output", str(result_path)])
            self.assertEqual(captured["launch"], argv)
            self.assertIs(captured["kwargs"]["stdin"], subprocess.PIPE)
            self.assertIs(captured["kwargs"]["stderr"], subprocess.STDOUT)
            self.assertIsInstance(captured["kwargs"]["stdout"], io.TextIOBase)
            self.assertIs(captured["kwargs"]["start_new_session"], True)
            self.assertEqual(captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(captured["kwargs"]["env"]["PATH"],
                             "/usr/local/cuda-12.9/bin:/usr/bin:/bin")
            log_file.close()
            self.assertTrue(log_path.is_file())

    def test_start_pressure_tenant_missing_binary_is_a_recordable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "uvm_fault_stream"
            log_path = Path(tmp) / "pressure.log"
            with self.assertRaises(FileNotFoundError):
                ops.start_pressure_tenant(missing, 2, 1, 0, log_path,
                                          Path(tmp) / "pressure-result.json")
        self.assertFalse(log_path.exists())

    def test_stop_pressure_tenant_is_a_bounded_process_group_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "pressure.log"
            log_file = log_path.open("w")
            proc = subprocess.Popen(["sleep", "30"], start_new_session=True,
                                    stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            try:
                started = time.monotonic()
                returncode, errors = ops.stop_pressure_tenant(proc, log_file)
                elapsed = time.monotonic() - started
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            self.assertEqual(errors, [])
            self.assertIsNotNone(returncode)
            self.assertIsNotNone(proc.poll())
            self.assertLess(elapsed, 20)

    def test_stop_pressure_tenant_closes_the_log_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "pressure.log"
            log_file = log_path.open("w")
            proc = subprocess.Popen(["true"], start_new_session=True,
                                    stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            proc.wait()
            returncode, errors = ops.stop_pressure_tenant(proc, log_file)
            self.assertEqual(errors, [])
            self.assertEqual(returncode, 0)
            self.assertTrue(log_file.closed)


class PressureReadinessTests(unittest.TestCase):
    class FakeProc:
        def __init__(self, returncode=None):
            self.returncode = returncode

        def poll(self):
            return self.returncode

    def test_returns_once_both_exact_lines_are_in_the_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "pressure.log"
            log_path.write_text(
                "READY pid=123 gib=2 regions=4 passes=3 pause_ms=250\n"
                "MONITOR_PID: 123\n"
                "Press Enter after the UVM monitor is ready...\n", encoding="utf-8")
            started = time.monotonic()
            ops.wait_pressure_tenant_ready(self.FakeProc(), log_path, timeout=5)
            self.assertLess(time.monotonic() - started, 5)

    def test_ready_line_alone_does_not_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "pressure.log"
            log_path.write_text(
                "READY pid=123 gib=2 regions=4 passes=1 pause_ms=0\n",
                encoding="utf-8")
            with self.assertRaises(ops.GateError):
                ops.wait_pressure_tenant_ready(self.FakeProc(), log_path, timeout=0.5)

    def test_monitor_line_alone_does_not_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "pressure.log"
            log_path.write_text("MONITOR_PID: 123\n", encoding="utf-8")
            with self.assertRaises(ops.GateError):
                ops.wait_pressure_tenant_ready(self.FakeProc(), log_path, timeout=0.5)

    def test_partial_ready_line_does_not_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "pressure.log"
            log_path.write_text("xREADY pid=123\nMONITOR_PID: 123\n",
                                encoding="utf-8")
            with self.assertRaises(ops.GateError):
                ops.wait_pressure_tenant_ready(self.FakeProc(), log_path, timeout=0.5)

    def test_tenant_exit_before_lines_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "pressure.log"
            log_path.write_text("starting\n", encoding="utf-8")
            with self.assertRaises(ops.GateError) as ctx:
                ops.wait_pressure_tenant_ready(self.FakeProc(returncode=1), log_path,
                                               timeout=5)
            self.assertIn("exited 1", str(ctx.exception))

    def test_readiness_timeout_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "pressure.log"
            log_path.write_text("starting\n", encoding="utf-8")
            with self.assertRaises(ops.GateError) as ctx:
                ops.wait_pressure_tenant_ready(self.FakeProc(), log_path, timeout=0.5)
            self.assertIn("timeout", str(ctx.exception))


class PressureReleaseTests(unittest.TestCase):
    class FakeTenant:
        def __init__(self, returncode=None, stdin=None):
            self.returncode = returncode
            self.stdin = stdin

        def poll(self):
            return self.returncode

    def test_release_newline_written_and_flushed_while_tenant_alive(self):
        buf = io.StringIO()
        outcome = runner.release_pressure_tenant(self.FakeTenant(stdin=buf))
        self.assertEqual(outcome, {"written": "\n", "released": True})
        self.assertEqual(buf.getvalue(), "\n")

    def test_release_not_sent_when_tenant_has_exited(self):
        outcome = runner.release_pressure_tenant(self.FakeTenant(returncode=0))
        self.assertIs(outcome["released"], False)
        self.assertIn("return code 0", outcome["error"])

    def test_release_write_failure_is_recorded_not_raised(self):
        class BrokenStdin:
            def write(self, text):
                raise OSError("broken pipe")

            def flush(self):
                pass

        outcome = runner.release_pressure_tenant(self.FakeTenant(stdin=BrokenStdin()))
        self.assertIs(outcome["released"], False)
        self.assertEqual(outcome["error"], "OSError: broken pipe")

    def test_release_not_sent_when_stdin_is_none(self):
        outcome = runner.release_pressure_tenant(self.FakeTenant(stdin=None))
        self.assertIs(outcome["released"], False)
        self.assertIn("stdin", outcome["error"])


class PressureCellTests(unittest.TestCase):
    PREFIXES = [{"index": 0, "cold_token_ids": [1], "warm_token_ids": [2],
                 "expected_store_tokens": 256}]

    @staticmethod
    def fake_response(request_id):
        phase = request_id.rsplit("-", 1)[1]
        index = request_id.rsplit("-", 2)[1]
        return {"request_header": request_id,
                "engine_request_id": f"cmpl-{request_id}",
                "input_tokens": 1, "status": 200, "ttft_ms": 1.0, "e2e_ms": 2.0,
                "usage": {"prompt_tokens": 1, "completion_tokens": 16},
                "text": "x", "generated_token_ids": [3] * 16,
                "_phase": phase, "_index": index}

    def run_cell_with_fakes(self, tmp, pressure=None, kv_bytes=None, gpu_util=None,
                            cpu_offload_gb=None, tenant_readiness_error=None):
        events = []
        start_server_kwargs = {}
        run_dir = Path(tmp) / "cell"
        extra = {}
        if kv_bytes is not None:
            extra["kv_cache_memory_bytes"] = kv_bytes
        if gpu_util is not None:
            extra["gpu_memory_utilization"] = gpu_util
        if cpu_offload_gb is not None:
            extra["cpu_offload_gb"] = cpu_offload_gb

        fake_server = mock.Mock()
        fake_server.pid = 1
        fake_server.returncode = None
        fake_server.poll.return_value = None

        def fake_start_server(*args, **kwargs):
            start_server_kwargs.update(kwargs)
            events.append("start_server")
            return fake_server, (run_dir / "server.log").open("w"), \
                ["vllm", "serve", "/model"], ["taskset", "-c", "8-15",
                                               "vllm", "serve", "/model"]

        def fake_streamed_completion(port, token_ids, request_id):
            events.append(f"request:{request_id}")
            return self.fake_response(request_id)

        def fake_stop_server(proc, log_file):
            events.append("stop_server")
            proc.returncode = 0
            log_file.close()

        fake_tenant = mock.Mock()
        fake_tenant.pid = 2
        fake_tenant.returncode = None
        fake_tenant.poll.return_value = None
        fake_tenant_stdin = io.StringIO()
        fake_tenant.stdin = fake_tenant_stdin
        tenant_log_file = {"handle": None}

        def fake_start_tenant(binary, gib, passes, pause_ms, log_path, result_path):
            events.append("start_tenant")
            handle = log_path.open("w")
            tenant_log_file["handle"] = handle
            handle.write("READY pid=2 gib=%d regions=4 passes=%d pause_ms=%d\n"
                         "MONITOR_PID: 2\n" % (gib, passes, pause_ms))
            return fake_tenant, handle, [str(binary), "--gib", str(gib),
                                         "--passes", str(passes),
                                         "--pause-ms", str(pause_ms),
                                         "--wait-for-monitor",
                                         "--output", str(result_path)]

        def fake_wait_tenant(proc, log_path):
            events.append("wait_tenant")
            if tenant_readiness_error is not None:
                raise tenant_readiness_error

        def fake_stop_tenant(proc, log_file):
            events.append("stop_tenant")
            (run_dir / "pressure-result.json").write_text(
                '{"bytes": 2147483648, "passes": 1, "kernel_ms": 0.5, '
                '"completed_passes": 1, "mismatches": 0, '
                '"first_mismatch": null}\n', encoding="utf-8")
            proc.returncode = 0
            log_file.close()
            return 0, []

        with mock.patch.object(ops, "start_server", side_effect=fake_start_server), \
                mock.patch.object(ops, "wait_ready",
                                  side_effect=lambda *a, **k: events.append(
                                      "wait_ready")), \
                mock.patch.object(ops, "streamed_completion",
                                  side_effect=fake_streamed_completion), \
                mock.patch.object(ops, "stop_owned_server",
                                  side_effect=fake_stop_server), \
                mock.patch.object(ops, "start_pressure_tenant",
                                  side_effect=fake_start_tenant) as start_tenant, \
                mock.patch.object(ops, "wait_pressure_tenant_ready",
                                  side_effect=fake_wait_tenant) as wait_tenant, \
                mock.patch.object(ops, "stop_pressure_tenant",
                                  side_effect=fake_stop_tenant) as stop_tenant:
            record = runner.run_cell("recompute", 0, 0, run_dir, 18080,
                                     Path("/model"), self.PREFIXES,
                                     EXPECTED_DRIVER, 120.0,
                                     pressure=pressure, **extra)
        return record, events, fake_tenant_stdin, start_server_kwargs, \
            start_tenant, wait_tenant, stop_tenant, run_dir

    def test_no_pressure_cell_keeps_the_old_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            record, events, tenant_stdin, start_kwargs, start_tenant, \
                wait_tenant, stop_tenant, run_dir = self.run_cell_with_fakes(tmp)
            self.assertEqual(events, ["start_server", "wait_ready",
                                      "request:lmc-p0-cold",
                                      "request:lmc-p0-warm", "stop_server"])
            start_tenant.assert_not_called()
            wait_tenant.assert_not_called()
            stop_tenant.assert_not_called()
            self.assertEqual(record["pressure"], {"enabled": False})
            self.assertIsNone(record["kv_cache_memory_bytes"])
            self.assertFalse((run_dir / "pressure.log").exists())
            self.assertFalse((run_dir / "pressure-result.json").exists())
            self.assertTrue((run_dir / "result.json").is_file())
            self.assertEqual(tenant_stdin.getvalue(), "")

    def test_pressure_cell_ordering_release_and_cleanup(self):
        pressure = {"gib": 2, "passes": 3, "pause_ms": 250, "binary": "/bin/tenant"}
        with tempfile.TemporaryDirectory() as tmp:
            record, events, tenant_stdin, start_kwargs, *_ = \
                self.run_cell_with_fakes(tmp, pressure=pressure,
                                         kv_bytes=48 * 1024**3)
            self.assertEqual(events, ["start_tenant", "wait_tenant",
                                      "start_server", "wait_ready",
                                      "request:lmc-p0-cold",
                                      "request:lmc-p0-warm",
                                      "stop_tenant", "stop_server"])
            self.assertEqual(start_kwargs.get("kv_cache_memory_bytes"), 48 * 1024**3)
            self.assertEqual(tenant_stdin.getvalue(), "\n")
            p = record["pressure"]
            self.assertIs(p["enabled"], True)
            self.assertEqual(p["command"], ["/bin/tenant", "--gib", "2",
                                            "--passes", "3", "--pause-ms", "250",
                                            "--wait-for-monitor",
                                            "--output",
                                            str(Path(tmp) / "cell" /
                                                "pressure-result.json")])
            self.assertEqual(p["log_path"], str(Path(tmp) / "cell" / "pressure.log"))
            self.assertEqual(p["result_path"], str(Path(tmp) / "cell" /
                                                    "pressure-result.json"))
            self.assertIs(p["ready"], True)
            self.assertEqual(p["release"], {"written": "\n", "released": True})
            self.assertEqual(p["returncode"], 0)
            self.assertEqual(p["cleanup_errors"], [])
            self.assertEqual(p["result_json"]["mismatches"], 0)
            self.assertEqual(record["kv_cache_memory_bytes"], 48 * 1024**3)
            self.assertEqual(record["warm_phase"]["requests"], 1)

    def test_gpu_memory_utilization_forwarded_to_start_server_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            record, events, tenant_stdin, start_kwargs, *_ = \
                self.run_cell_with_fakes(tmp, gpu_util=0.75)
            self.assertEqual(start_kwargs.get("gpu_memory_utilization"), 0.75)
            self.assertEqual(record["gpu_memory_utilization"], 0.75)

    def test_gpu_memory_utilization_defaults_to_098(self):
        with tempfile.TemporaryDirectory() as tmp:
            record, events, tenant_stdin, start_kwargs, *_ = \
                self.run_cell_with_fakes(tmp)
            self.assertEqual(start_kwargs.get("gpu_memory_utilization"), 0.98)
            self.assertEqual(record["gpu_memory_utilization"], 0.98)

    def test_cpu_offload_gb_forwarded_to_start_server_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            record, events, tenant_stdin, start_kwargs, *_ = \
                self.run_cell_with_fakes(tmp, cpu_offload_gb=8.0)
            self.assertEqual(start_kwargs.get("cpu_offload_gb"), 8.0)
            self.assertEqual(record["cpu_offload_gb"], 8.0)

    def test_cpu_offload_gb_defaults_to_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            record, events, tenant_stdin, start_kwargs, *_ = \
                self.run_cell_with_fakes(tmp)
            self.assertEqual(start_kwargs.get("cpu_offload_gb"), 0.0)
            self.assertEqual(record["cpu_offload_gb"], 0.0)

    def test_pressure_launch_error_is_recorded_and_cell_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "cell"
            events = []
            fake_server = mock.Mock()
            fake_server.pid = 1
            fake_server.returncode = None
            fake_server.poll.return_value = None

            def fake_streamed_completion(port, token_ids, request_id):
                events.append(f"request:{request_id}")
                return self.fake_response(request_id)

            def fake_stop_server(proc, log_file):
                events.append("stop_server")
                proc.returncode = 0
                log_file.close()

            def fake_start_server(*args, **kwargs):
                return fake_server, (run_dir / "server.log").open("w"), \
                    ["vllm"], ["vllm"]

            with mock.patch.object(ops, "start_server",
                                   side_effect=fake_start_server), \
                    mock.patch.object(ops, "wait_ready"), \
                    mock.patch.object(ops, "streamed_completion",
                                      side_effect=fake_streamed_completion), \
                    mock.patch.object(ops, "stop_owned_server",
                                      side_effect=fake_stop_server), \
                    mock.patch.object(ops, "start_pressure_tenant",
                                      side_effect=FileNotFoundError(
                                          "pressure tenant binary not found: /bin/missing")) \
                        as start_tenant, \
                    mock.patch.object(ops, "stop_pressure_tenant") as stop_tenant:
                record = runner.run_cell("recompute", 0, 0, run_dir, 18080,
                                         Path("/model"), self.PREFIXES,
                                         EXPECTED_DRIVER, 120.0,
                                         pressure={"gib": 2, "passes": 1,
                                                   "pause_ms": 0,
                                                   "binary": "/bin/missing"})
            start_tenant.assert_called_once()
            stop_tenant.assert_not_called()
            p = record["pressure"]
            self.assertIs(p["enabled"], True)
            self.assertIn("FileNotFoundError", p["error"])
            self.assertIs(p["ready"], False)
            self.assertIsNone(p["command"])
            self.assertIsNone(p["log_path"])
            self.assertIsNone(p["result_path"])
            self.assertIsNone(p["release"])
            self.assertIsNone(p["returncode"])
            self.assertEqual(p["cleanup_errors"], [])
            self.assertEqual(events, ["request:lmc-p0-cold",
                                      "request:lmc-p0-warm", "stop_server"])
            self.assertTrue((run_dir / "result.json").is_file())

    def test_pressure_readiness_failure_still_stops_the_tenant(self):
        pressure = {"gib": 2, "passes": 1, "pause_ms": 0, "binary": "/bin/tenant"}
        with tempfile.TemporaryDirectory() as tmp:
            record, events, tenant_stdin, *_ = self.run_cell_with_fakes(
                tmp, pressure=pressure,
                tenant_readiness_error=ops.GateError("readiness timeout"))
            self.assertEqual(events, ["start_tenant", "wait_tenant",
                                      "start_server", "wait_ready",
                                      "request:lmc-p0-cold",
                                      "request:lmc-p0-warm",
                                      "stop_tenant", "stop_server"])
            self.assertEqual(tenant_stdin.getvalue(), "")
            p = record["pressure"]
            self.assertIn("GateError", p["error"])
            self.assertIs(p["ready"], False)
            self.assertIsNone(p["release"])
            self.assertEqual(p["returncode"], 0)
            self.assertEqual(p["cleanup_errors"], [])
            self.assertTrue((Path(tmp) / "cell" / "result.json").is_file())

    def test_loader_arm_pressure_started_before_loader_and_server(self):
        pressure = {"gib": 2, "passes": 3, "pause_ms": 250, "binary": "/bin/tenant"}
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "cell"
            events = []
            loader_stdin = io.StringIO()
            tenant_stdin = io.StringIO()

            fake_server = mock.Mock()
            fake_server.pid = 1
            fake_server.returncode = None
            fake_server.poll.return_value = None

            fake_loader = mock.Mock()
            fake_loader.pid = 3
            fake_loader.returncode = None
            fake_loader.poll.return_value = None
            fake_loader.stdin = loader_stdin

            fake_tenant = mock.Mock()
            fake_tenant.pid = 2
            fake_tenant.returncode = None
            fake_tenant.poll.return_value = None
            fake_tenant.stdin = tenant_stdin

            def fake_start_server(*args, **kwargs):
                events.append("start_server")
                return fake_server, (run_dir / "server.log").open("w"), \
                    ["vllm"], ["vllm"]

            def fake_wait_ready(*args, **kwargs):
                events.append("wait_ready")

            def fake_streamed_completion(port, token_ids, request_id):
                events.append(f"request:{request_id}")
                return self.fake_response(request_id)

            def fake_stop_server(proc, log_file):
                events.append("stop_server")
                proc.returncode = 0
                log_file.close()

            def fake_start_tenant(binary, gib, passes, pause_ms, log_path, result_path):
                events.append("start_tenant")
                return fake_tenant, log_path.open("w"), [str(binary)]

            def fake_wait_tenant(proc, log_path):
                events.append("wait_tenant")

            def fake_stop_tenant(proc, log_file):
                events.append("stop_tenant")
                proc.returncode = 0
                log_file.close()
                return 0, []

            def fake_start_loader(loader_path, log_path, allocator_path):
                events.append("start_loader")
                return fake_loader, log_path.open("w"), \
                    [str(loader_path), "-a", str(allocator_path), "-w", "0"], \
                    ["/usr/bin/sudo", "-n"]

            def fake_wait_loader(proc, log_path):
                events.append("wait_loader")

            def fake_stop_loader(proc, log_file):
                events.append("stop_loader")
                proc.returncode = 0
                log_file.close()
                return 0, []

            with mock.patch.object(ops, "start_server",
                                   side_effect=fake_start_server), \
                    mock.patch.object(ops, "wait_ready",
                                      side_effect=fake_wait_ready), \
                    mock.patch.object(ops, "streamed_completion",
                                      side_effect=fake_streamed_completion), \
                    mock.patch.object(ops, "stop_owned_server",
                                      side_effect=fake_stop_server), \
                    mock.patch.object(ops, "start_pressure_tenant",
                                      side_effect=fake_start_tenant), \
                    mock.patch.object(ops, "wait_pressure_tenant_ready",
                                      side_effect=fake_wait_tenant), \
                    mock.patch.object(ops, "stop_pressure_tenant",
                                      side_effect=fake_stop_tenant), \
                    mock.patch.object(ops, "start_eviction_loader",
                                      side_effect=fake_start_loader), \
                    mock.patch.object(ops, "wait_eviction_loader_ready",
                                      side_effect=fake_wait_loader), \
                    mock.patch.object(ops, "stop_eviction_loader",
                                      side_effect=fake_stop_loader):
                record = runner.run_cell(runner.LOADER_ARM, 0, 0, run_dir, 18080,
                                         Path("/model"), self.PREFIXES,
                                         EXPECTED_DRIVER, 0.0,
                                         pressure=pressure)
            self.assertEqual(events, ["start_tenant", "wait_tenant",
                                      "start_loader", "wait_loader",
                                      "start_server", "wait_ready",
                                      "request:lmc-p0-cold",
                                      "request:lmc-p0-warm",
                                      "stop_tenant", "stop_server",
                                      "stop_loader"])
            self.assertEqual(loader_stdin.getvalue(), "w\n")
            self.assertEqual(tenant_stdin.getvalue(), "\n")
            self.assertIs(record["loader"]["ready"], True)
            self.assertIs(record["loader"]["warm_signal"]["sent"], True)
            self.assertIs(record["pressure"]["ready"], True)
            self.assertEqual(record["pressure"]["release"],
                             {"written": "\n", "released": True})
            self.assertEqual(record["pressure"]["returncode"], 0)
            self.assertEqual(record["loader"]["returncode"], 0)
            self.assertEqual(record["server_returncode"], 0)
            self.assertEqual(record["pressure"]["cleanup_errors"], [])
            self.assertEqual(record["loader"]["cleanup_errors"], [])
            self.assertTrue((run_dir / "result.json").is_file())


class PressureDryRunTests(unittest.TestCase):
    def capture_main(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = runner.main(argv)
        return rc, json.loads(buf.getvalue())

    def test_dry_run_default_pressure_is_disabled(self):
        rc, plan = self.capture_main(["--dry-run"])
        self.assertEqual(rc, 0)
        self.assertIsNone(plan["kv_cache_memory_bytes"])
        self.assertIs(plan["pressure"]["enabled"], False)
        self.assertEqual(plan["pressure"]["gib"], 0)
        self.assertEqual(plan["pressure"]["binary"], str(ops.PRESSURE_TENANT))
        self.assertIn("--wait-for-monitor", plan["pressure"]["launch"])
        self.assertIn("pressure.log", plan["pressure"]["stdout_stderr"])

    def test_dry_run_pressure_plan_reflects_the_cli(self):
        rc, plan = self.capture_main(
            ["--dry-run", "--pressure-gib", "4", "--pressure-passes", "2",
             "--pressure-pause-ms", "100", "--kv-cache-memory-bytes",
             "21474836480", "--pressure-binary", "/tmp/uvm_fault_stream"])
        self.assertEqual(rc, 0)
        self.assertEqual(plan["kv_cache_memory_bytes"], 21474836480)
        pressure = plan["pressure"]
        self.assertIs(pressure["enabled"], True)
        self.assertEqual(pressure["gib"], 4)
        self.assertEqual(pressure["passes"], 2)
        self.assertEqual(pressure["pause_ms"], 100)
        self.assertEqual(pressure["binary"], "/tmp/uvm_fault_stream")
        self.assertEqual(pressure["recorded"],
                         ["command", "log_path", "result_path", "ready", "release",
                          "returncode", "result_json", "cleanup_errors", "error"])
        self.assertIn("start_pressure_tenant", plan["reuse"])
        self.assertIn("wait_pressure_tenant_ready", plan["reuse"])
        self.assertIn("stop_pressure_tenant", plan["reuse"])


class CampaignCpuOffloadTests(unittest.TestCase):
    def run_campaign_with_fake_cells(self, extra_argv):
        kwargs_seen = {}

        def fake_run_cell(*args, **kwargs):
            kwargs_seen.update(kwargs)
            return {"ready": True, "warm_phase": {"requests_per_s": 1.0},
                    "server_returncode": 0}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "campaign"
            args = runner.parse_args(
                ["--configs", "recompute", "--output", str(out)] + extra_argv)
            with mock.patch.object(ops, "resolve_model",
                                   return_value=Path("/model")), \
                    mock.patch.object(runner, "run_cell",
                                      side_effect=fake_run_cell) as run_cell:
                rc = runner.run_campaign(args)
            campaign = json.loads((out / "campaign.json").read_text(encoding="utf-8"))
            return rc, run_cell, kwargs_seen, campaign

    def test_run_campaign_records_default_cpu_offload_gb(self):
        rc, run_cell, kwargs_seen, campaign = self.run_campaign_with_fake_cells([])
        self.assertEqual(rc, 0)
        self.assertEqual(run_cell.call_count, 1)
        self.assertEqual(campaign["params"]["cpu_offload_gb"], 0.0)
        self.assertEqual(kwargs_seen.get("cpu_offload_gb"), 0.0)

    def test_run_campaign_forwards_override_to_cells_and_records_it(self):
        rc, run_cell, kwargs_seen, campaign = self.run_campaign_with_fake_cells(
            ["--cpu-offload-gb", "8"])
        self.assertEqual(rc, 0)
        self.assertEqual(run_cell.call_count, 1)
        self.assertEqual(campaign["params"]["cpu_offload_gb"], 8.0)
        self.assertEqual(kwargs_seen.get("cpu_offload_gb"), 8.0)


class DryRunTests(unittest.TestCase):
    def capture_main(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = runner.main(argv)
        return rc, json.loads(buf.getvalue())

    def test_main_dry_run_defaults_to_one_block(self):
        rc, plan = self.capture_main(["--dry-run"])
        self.assertEqual(rc, 0)
        self.assertIs(plan["dry_run"], True)
        self.assertEqual(plan["blocks"], 1)
        self.assertEqual(plan["configs"], list(runner.CONFIGS))
        self.assertEqual(len(plan["configs"]), 4)
        self.assertEqual(plan["block_orders"], [list(runner.CONFIGS)])
        self.assertEqual(plan["expected_driver_parameter"], EXPECTED_DRIVER)
        self.assertEqual(plan["port"], 18080)
        self.assertEqual(plan["store_barrier"]["timeout_s"], 120.0)
        self.assertEqual(plan["eviction_loader"]["arm"], "lmcache_disk_uvm_kv_gpubpf_debt")
        self.assertEqual(plan["eviction_loader"]["path"], str(ops.EVICTION_LOADER))
        self.assertEqual(plan["eviction_loader"]["allocator"], str(ops.UVM_KV_ALLOCATOR_SO))
        self.assertIn("sudo -n", plan["eviction_loader"]["launch"])
        self.assertIn("-a <allocator> -w 0", plan["eviction_loader"]["launch"])
        self.assertIn("loader.log", plan["eviction_loader"]["stdout_stderr"])

    def test_main_dry_run_eviction_loader_override(self):
        rc, plan = self.capture_main(
            ["--dry-run", "--eviction-loader", "/tmp/eviction_debt"])
        self.assertEqual(rc, 0)
        self.assertEqual(plan["eviction_loader"]["path"], "/tmp/eviction_debt")

    def test_main_dry_run_records_gpu_memory_utilization(self):
        rc, plan = self.capture_main(["--dry-run"])
        self.assertEqual(rc, 0)
        self.assertEqual(plan["gpu_memory_utilization"], 0.98)
        rc, plan = self.capture_main(
            ["--dry-run", "--gpu-memory-utilization", "0.9"])
        self.assertEqual(rc, 0)
        self.assertEqual(plan["gpu_memory_utilization"], 0.9)

    def test_main_dry_run_records_cpu_offload_gb(self):
        rc, plan = self.capture_main(["--dry-run"])
        self.assertEqual(rc, 0)
        self.assertEqual(plan["cpu_offload_gb"], 0.0)
        rc, plan = self.capture_main(["--dry-run", "--cpu-offload-gb", "8"])
        self.assertEqual(rc, 0)
        self.assertEqual(plan["cpu_offload_gb"], 8.0)

    def test_main_dry_run_three_blocks_matches_rotation(self):
        rc, plan = self.capture_main(["--dry-run", "--blocks", "3"])
        self.assertEqual(rc, 0)
        self.assertEqual(plan["blocks"], 3)
        self.assertEqual(plan["block_orders"], runner.rotation_orders(3))

    def test_main_dry_run_one_arm_selection_drives_configs_and_orders(self):
        rc, plan = self.capture_main(
            ["--dry-run", "--configs", "lmcache_disk_uvm_kv_gpubpf_debt"])
        self.assertEqual(rc, 0)
        self.assertEqual(plan["configs"], ["lmcache_disk_uvm_kv_gpubpf_debt"])
        self.assertEqual(plan["block_orders"], [["lmcache_disk_uvm_kv_gpubpf_debt"]])

    def test_main_dry_run_two_arm_selection_rotates_selected_arms(self):
        rc, plan = self.capture_main(
            ["--dry-run", "--configs", "lmcache_disk_uvm_kv_gpubpf_debt,recompute",
             "--blocks", "2"])
        self.assertEqual(rc, 0)
        self.assertEqual(plan["configs"],
                         ["lmcache_disk_uvm_kv_gpubpf_debt", "recompute"])
        self.assertEqual(plan["block_orders"],
                         [["lmcache_disk_uvm_kv_gpubpf_debt", "recompute"],
                          ["recompute", "lmcache_disk_uvm_kv_gpubpf_debt"]])

    def test_main_dry_run_rejects_invalid_configs_without_output(self):
        with self.assertRaises(ValueError):
            self.capture_main(["--dry-run", "--configs", "nosuch_arm"])

    def test_main_dry_run_rejects_duplicate_configs_without_output(self):
        with self.assertRaises(ValueError):
            self.capture_main(["--dry-run", "--configs",
                               "lmcache_disk,lmcache_disk"])

    def test_main_dry_run_rejects_empty_configs_without_output(self):
        with self.assertRaises(ValueError):
            self.capture_main(["--dry-run", "--configs", ""])

    def test_main_dry_run_rejects_zero_blocks_without_output(self):
        with self.assertRaises(ValueError):
            self.capture_main(["--dry-run", "--blocks", "0"])


if __name__ == "__main__":
    unittest.main()
