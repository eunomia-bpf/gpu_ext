#!/usr/bin/env python3
"""Pure CPU/mocked checks; never execute a GPU client or build a probe."""
import ast
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import run_targeted_diagnostic as diagnostic

runner = diagnostic.runner


class TargetedTests(unittest.TestCase):
    def exercise(self, directory, task="threadhist", defect=None):
        args = SimpleNamespace(output_dir=directory / "new")
        calls = []
        def cell(config, attempt, args, output, tools, **kwargs):
            calls.append((config, attempt, kwargs.get("diagnostic_log_level")))
            if defect == "fatal" and config != "baseline":
                raise runner.OwnedCleanupError("client survived", {"role": "CUDA client"})
            if defect == "interrupt" and config != "baseline":
                raise KeyboardInterrupt("owned interruption")
            result = {"returncode": 0, "normalized_stdout": "exact eight-token output", "valid": True}
            if config != "baseline":
                count = 80 if defect == "count" and config.startswith("gpubpf") else 100
                result["probe"] = {"sample_count": count, "nonzero_threads": 12}
            if config.startswith("gpubpf"):
                if defect == "stdout":
                    result["normalized_stdout"] = "runtime diagnostic\nexact eight-token output"
                if defect == "probe":
                    result["valid"] = False
                    result["probe"].update(host_launches=0, device_entries=100, clock_errors=100)
            return result
        snapshot = {"gpu": "RTX 5090, 610.43.02" if defect == "driver" else "RTX 5090, 575.57.08",
                    "compute_apps": ""}
        with (patch.object(runner, "defining_params", return_value={}),
              patch.object(diagnostic, "current_inventory", return_value=[]),
              patch.object(runner.core, "nvidia_smi_snapshot", return_value=snapshot),
              patch.object(runner, "run_correctness_cell", side_effect=cell),
              patch.object(runner, "run_cell", side_effect=AssertionError("no timing")),
              patch.object(runner, "build_nvbit", side_effect=AssertionError("no build"))):
            expected = {"fatal": runner.OwnedCleanupError, "interrupt": KeyboardInterrupt,
                        "driver": RuntimeError}.get(defect)
            if expected:
                with self.assertRaises(expected):
                    diagnostic.run_diagnostic(args, {}, task, directory / "old")
                status = None
            else:
                status = diagnostic.run_diagnostic(args, {}, task, directory / "old")
        record = json.loads((args.output_dir / "diagnostic.json").read_text())
        self.assertEqual(record["timing_cells_started"], 0)
        self.assertIn("ended_utc", record)
        return status, calls, record

    def test_histogram_three_fresh_cells_and_aggregate_difference(self):
        for defect, expected in ((None, 0), ("count", 2)):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                status, calls, record = self.exercise(Path(tmp), defect=defect)
                self.assertEqual(status, expected)
                self.assertEqual(calls, [("baseline", 1, None), ("nvbit_threadhist", 1, None),
                                         ("gpubpf_threadhist", 1, None)])
                comparison = record["histogram_comparison"]
                self.assertEqual(comparison["gpubpf_over_nvbit"], 0.8 if defect else 1.0)
                self.assertEqual(comparison["difference_percent"], -20.0 if defect else 0.0)
                self.assertTrue(all(row["valid"] for row in record["cells"]))

    def test_stdout_pollution_and_probe_failure_are_not_sanitized(self):
        for defect in ("stdout", "probe"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                status, calls, record = self.exercise(Path(tmp), defect=defect)
                self.assertEqual(status, 2)
                self.assertFalse(record["diagnostic_passed"])
                self.assertNotIn("histogram_comparison", record)
                self.assertFalse(record["cells"][-1]["diagnostic_valid"])
                if defect == "stdout":
                    self.assertIn("runtime diagnostic", record["cells"][-1]["normalized_stdout"])

    def test_launchlate_info_does_not_hide_clock_or_host_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, calls, record = self.exercise(Path(tmp), task="launchlate", defect="probe")
        self.assertEqual(status, 2)
        self.assertEqual(calls, [("baseline", 1, None), ("gpubpf_launchlate", 1, "info")])
        self.assertEqual(record["cells"][-1]["probe"]["host_launches"], 0)
        self.assertFalse(record["cells"][-1]["valid"])

    def test_fatal_cleanup_interrupt_and_wrong_driver_stop_the_sequence(self):
        for defect in ("fatal", "interrupt", "driver"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                _, calls, record = self.exercise(Path(tmp), defect=defect)
                self.assertEqual(len(calls), 0 if defect == "driver" else 2)
                self.assertFalse(record["diagnostic_passed"])
                self.assertIn("error", record)
                if defect == "fatal":
                    self.assertEqual(record["cells"][-1]["fatal_cleanup"]["role"], "CUDA client")

    def test_existing_output_is_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner.core, "nvidia_smi_snapshot") as gpu:
            with self.assertRaises(FileExistsError):
                diagnostic.run_diagnostic(SimpleNamespace(output_dir=Path(tmp)), {}, "threadhist", Path("/old"))
            gpu.assert_not_called()

    def test_reuse_paths_but_inventory_current_files_and_new_nvbit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "old", root / "new"
            source.mkdir()
            probe = root / "threadhist"
            probe.write_text("prepared probe")
            nvbit = root / "new-nvbit.so"
            nvbit.write_text("new guarded library")
            fields = ("model", "llama_bench", "llama_cli", "bpftime_root", "bpftime_build_dir", "uprobe_binary")
            params = {name: str(root / name) for name in fields}
            params["phase"] = "preflight"
            (source / "result.json").write_text(json.dumps({"params": params, "artifacts": {
                "gpubpf_threadhist": {"path": str(probe), "bytes": 9999},
                "nvbit_tool": {"path": str(root / "missing-old.so")}}}))
            with patch.object(runner, "validate") as validate:
                args, tools = diagnostic.load_inputs(source, output, "threadhist", root / "new-build", nvbit)
                validate.assert_called_once_with(args)
            self.assertEqual(args.nvbit_tool, nvbit)
            self.assertEqual(args.bpftime_build_dir, root / "new-build")
            inventory = {row["path"]: row for row in diagnostic.current_inventory(args, tools)}
            self.assertEqual(inventory[str(probe)]["bytes"], len("prepared probe"))
            self.assertEqual(inventory[str(nvbit)]["bytes"], len("new guarded library"))
            with self.assertRaisesRegex(ValueError, "outside"):
                diagnostic.load_inputs(source, source / "nested", "threadhist", None)

    def test_main_holds_leases_and_restores_handler_on_interrupt(self):
        lease = Mock()
        previous = signal.getsignal(signal.SIGTERM)
        previous_helper = runner.core.run_cmd
        def interrupt(*args):
            self.assertIs(runner.core.run_cmd, runner.run_cmd_owned)
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        with (patch.object(runner, "reject_ambient_injection"),
              patch.object(diagnostic, "load_inputs", return_value=(SimpleNamespace(), {})),
              patch.object(runner.shared, "Leases", return_value=lease),
              patch.object(diagnostic, "run_diagnostic", side_effect=interrupt)):
            with self.assertRaises(KeyboardInterrupt):
                diagnostic.main(["--output-dir", "/new-output"])
        lease.close.assert_called_once()
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        self.assertIs(runner.core.run_cmd, previous_helper)

    def test_private_info_logging_is_recorded_for_both_loader_and_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segment = root / f"rq4_{os.getpid()}_123"
            process = Mock(pid=98765, returncode=0)
            process.poll.return_value = None
            def start(command, **kwargs):
                segment.write_text("private fixture")
                self.assertEqual(kwargs["env"]["SPDLOG_LEVEL"], "info")
                self.assertTrue(kwargs["start_new_session"])
                return process
            with (patch.object(runner, "SHM_ROOT", root),
                  patch.object(runner.time, "monotonic_ns", return_value=123),
                  patch.object(runner.core, "probe_env", return_value={"SPDLOG_LEVEL": "warn"}),
                  patch.object(runner.core, "agent_env", return_value={"SPDLOG_LEVEL": "warn"}),
                  patch.object(runner.subprocess, "Popen", side_effect=start),
                  patch.object(runner.shared, "stop_owned"),
                  patch.object(runner.shared, "group_members", return_value=[])):
                with runner.private_probe("threadhist", SimpleNamespace(probe_startup_s=0), root,
                                          root / "cell", diagnostic_log_level="info") as env:
                    self.assertEqual(env["SPDLOG_LEVEL"], "info")
            saved = json.loads((root / "cell/probe-execution.json").read_text())
            self.assertEqual(saved["loader_environment"]["SPDLOG_LEVEL"], "info")
            self.assertEqual(saved["agent_environment"]["SPDLOG_LEVEL"], "info")
            self.assertTrue(saved["private_segment_removed"])

    def test_ambient_injection_stops_before_inputs_and_lease(self):
        with (patch.dict(os.environ, {"LD_PRELOAD": "/unowned.so"}, clear=True),
              patch.object(diagnostic, "load_inputs") as inputs,
              patch.object(runner.shared, "Leases") as lease):
            with self.assertRaisesRegex(RuntimeError, "uninjected"):
                diagnostic.main(["--output-dir", "/new-output"])
        inputs.assert_not_called()
        lease.assert_not_called()

    def test_no_build_or_performance_entry_points(self):
        tree = ast.parse(Path(diagnostic.__file__).read_text())
        names = [node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        for forbidden in ("build_nvbit", "build_tool", "prepare_tool_source", "run_campaign", "run_cell", "run_bench"):
            self.assertNotIn(forbidden, names)


if __name__ == "__main__":
    unittest.main()
