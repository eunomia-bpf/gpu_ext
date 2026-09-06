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

    def test_start_eviction_loader_runs_sudo_n_with_w0_and_new_session(self):
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
            self.assertEqual(argv, [str(ops.EVICTION_LOADER), "-w", "0"])
            self.assertEqual(launch, ["/usr/bin/sudo", "-n", "/usr/bin/stdbuf", "-oL", "-eL",
                                      str(ops.EVICTION_LOADER), "-w", "0"])
            self.assertIs(captured["kwargs"]["stdin"], subprocess.PIPE)
            self.assertIs(captured["kwargs"]["stderr"], subprocess.STDOUT)
            self.assertIsInstance(captured["kwargs"]["stdout"], io.TextIOBase)
            self.assertIs(captured["kwargs"]["start_new_session"], True)
            self.assertEqual(captured["kwargs"]["env"]["PATH"],
                             "/usr/local/cuda-12.9/bin:/usr/bin:/bin")
            self.assertEqual(captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"], "")
            log_file.close()
            self.assertTrue(log_path.is_file())

    def test_start_eviction_loader_missing_binary_is_a_recordable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "loader.log"
            with mock.patch.object(Path, "is_file", return_value=False):
                with self.assertRaises(FileNotFoundError):
                    ops.start_eviction_loader(ops.EVICTION_LOADER, log_path)
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
                                      ["loader", "-w", "0"],
                                      ["/usr/bin/sudo", "-n", "/usr/bin/stdbuf",
                                       "-oL", "-eL", "loader", "-w", "0"])), \
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
        self.assertIn("sudo -n", plan["eviction_loader"]["launch"])
        self.assertIn("loader.log", plan["eviction_loader"]["stdout_stderr"])

    def test_main_dry_run_eviction_loader_override(self):
        rc, plan = self.capture_main(
            ["--dry-run", "--eviction-loader", "/tmp/eviction_debt"])
        self.assertEqual(rc, 0)
        self.assertEqual(plan["eviction_loader"]["path"], "/tmp/eviction_debt")

    def test_main_dry_run_three_blocks_matches_rotation(self):
        rc, plan = self.capture_main(["--dry-run", "--blocks", "3"])
        self.assertEqual(rc, 0)
        self.assertEqual(plan["blocks"], 3)
        self.assertEqual(plan["block_orders"], runner.rotation_orders(3))

    def test_main_dry_run_rejects_zero_blocks_without_output(self):
        with self.assertRaises(ValueError):
            self.capture_main(["--dry-run", "--blocks", "0"])


if __name__ == "__main__":
    unittest.main()
