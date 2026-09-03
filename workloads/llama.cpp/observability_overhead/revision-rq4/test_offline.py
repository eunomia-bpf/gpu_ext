#!/usr/bin/env python3
"""CPU-only checks for the inactive revision-RQ4 adapter."""

import ast
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import run_revision_rq4 as runner


class OfflineTests(unittest.TestCase):
    def test_cpu_helper_preserves_results_errors_and_streamed_logs(self):
        for outcome in ("success", "unchecked failure", "failure", "timeout", "interrupt", "survivor"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                log = directory / "helper.log"
                process = Mock(pid=98766, returncode=7 if "failure" in outcome else 0,
                               stdout=io.StringIO("first line\nlast line\n"))
                process.wait.return_value = process.returncode
                if outcome == "timeout":
                    process.wait.side_effect = runner.subprocess.TimeoutExpired(["/helper"], 1)
                elif outcome == "interrupt":
                    process.wait.side_effect = KeyboardInterrupt("interrupted helper")
                with (patch.object(runner.subprocess, "Popen", return_value=process) as popen,
                      patch.object(runner.shared, "stop_owned",
                                   side_effect=RuntimeError("owned group survives") if outcome == "survivor" else None) as stop,
                      patch.object(runner.shared, "group_members", return_value=[98766])):
                    def execute():
                        return runner.run_cmd_owned(["/helper"], cwd=directory, env={"PATH": "/usr/bin"},
                                                    timeout=1, log_path=log, check=outcome != "unchecked failure")
                    expected = {"failure": RuntimeError, "timeout": runner.subprocess.TimeoutExpired,
                                "interrupt": KeyboardInterrupt, "survivor": runner.OwnedCleanupError}.get(outcome)
                    if expected:
                        with self.assertRaises(expected) as caught:
                            execute()
                        if outcome == "timeout":
                            self.assertEqual(caught.exception.output, "first line\nlast line\n")
                    else:
                        result = execute()
                        self.assertIsInstance(result, runner.subprocess.CompletedProcess)
                        self.assertEqual((result.args, result.returncode, result.stdout, result.stderr),
                                         (["/helper"], process.returncode, "first line\nlast line\n", ""))
                    stop.assert_called_once_with(process)
                    self.assertTrue(popen.call_args.kwargs["start_new_session"])
                    self.assertEqual(popen.call_args.kwargs["env"], {"PATH": "/usr/bin"})
                text = log.read_text()
                self.assertIn("## output\nfirst line\nlast line\n", text)
                self.assertIn(f"# exit: {process.returncode}", text)
                if outcome == "timeout":
                    self.assertIn("# timeout_s: 1", text)
                if outcome == "interrupt":
                    self.assertIn("KeyboardInterrupt: interrupted helper", text)
                if outcome == "survivor":
                    self.assertIn('"live_group_members": [98766]', text)

    def test_cpu_helper_closes_log_when_spawn_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "helper.log"
            with (patch.object(runner.subprocess, "Popen", side_effect=FileNotFoundError("missing helper")),
                  patch.object(runner.shared, "stop_owned") as stop):
                with self.assertRaises(FileNotFoundError):
                    runner.run_cmd_owned(["/missing"], log_path=log)
            stop.assert_called_once_with(None)
            self.assertIn("FileNotFoundError: missing helper", log.read_text())

    def test_cpu_helper_interrupt_leaves_no_real_child_group(self):
        actual_popen = runner.subprocess.Popen
        launched = []
        def start(*args, **kwargs):
            process = actual_popen(*args, **kwargs)
            launched.append(process)
            wait = process.wait
            def interrupt_once(timeout=None):
                process.wait = wait
                raise KeyboardInterrupt("CPU-only interrupt check")
            process.wait = interrupt_once
            return process
        with patch.object(runner.subprocess, "Popen", side_effect=start):
            with self.assertRaises(KeyboardInterrupt):
                runner.run_cmd_owned([runner.sys.executable, "-B", "-c", "import time; time.sleep(30)"])
        self.assertEqual(runner.shared.group_members(launched[0].pid), [])
        self.assertIsNotNone(launched[0].returncode)

    def test_injection_occurs_only_after_affinity_wrapper(self):
        original = {"LD_PRELOAD": "/instrumentation.so", "PATH": "/usr/bin", "BPFTIME_GLOBAL_SHM_NAME": "owned"}
        command, environment = runner.target_launch(["/client", "argument"], original)
        self.assertEqual(command, ["taskset", "-c", "8-15", "/usr/bin/env",
                                   "LD_PRELOAD=/instrumentation.so", "/client", "argument"])
        self.assertNotIn("LD_PRELOAD", environment)
        self.assertEqual(original["LD_PRELOAD"], "/instrumentation.so")
        self.assertEqual(environment["BPFTIME_GLOBAL_SHM_NAME"], "owned")

    def test_safety_records_failures_and_checks_owned_teardown(self):
        for defect in (None, "client", "telemetry", "post-safety"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                process, stream = Mock(), Mock()
                process.poll.return_value = None
                snapshot = {"gpu": {"driver": "575.57.08"}}
                with (patch.object(runner.shared.safety, "safety_snapshot", return_value=snapshot),
                      patch.object(runner.shared.safety, "validate_pre_server_safety"),
                      patch.object(runner.shared.safety, "start_gpu_telemetry", return_value=(process, stream, directory / "gpu.csv")),
                      patch.object(runner.shared.safety, "wait_for_post_server_safety", return_value=snapshot,
                                   side_effect=RuntimeError("post-safety") if defect == "post-safety" else None),
                      patch.object(runner.shared.safety, "validate_gpu_telemetry", return_value={"samples": 2},
                                   side_effect=RuntimeError("telemetry") if defect == "telemetry" else None),
                      patch.object(runner.shared, "stop_owned") as stop):
                    def execute():
                        with runner.cell_safety(directory):
                            if defect == "client":
                                raise RuntimeError("client")
                    if defect:
                        with self.assertRaisesRegex(RuntimeError, defect):
                            execute()
                    else:
                        execute()
                record = json.loads((directory / "gpu-safety.json").read_text())
                self.assertEqual(record["passed"], defect is None)
                self.assertEqual(record["worker_cpus"], "8-15")
                stop.assert_called_once_with(process)
                stream.close.assert_called_once()

    def test_admission_requires_exact_575_before_build(self):
        for driver in ("575.57.08", "575.99", "570.124.06", "610.43.02"):
            with self.subTest(driver=driver), tempfile.TemporaryDirectory() as tmp:
                args = SimpleNamespace(output_dir=Path(tmp), phase="preflight", resume=False)
                snapshot = {"gpu": f"RTX 5090, {driver}, 32607, 0, 0, 0", "compute_apps": ""}
                with (patch.object(runner.core, "nvidia_smi_snapshot", return_value=snapshot),
                      patch.object(runner.shutil, "copytree"),
                      patch.object(runner, "build_nvbit", side_effect=RuntimeError("build boundary")) as build):
                    message = "build boundary" if driver == "575.57.08" else "requires driver"
                    with self.assertRaisesRegex(RuntimeError, message):
                        runner.run_campaign(args)
                self.assertEqual(build.call_count, int(driver == "575.57.08"))
                admission = json.loads(next(Path(tmp).glob("admission-*.json")).read_text())
                self.assertEqual(admission["driver"], driver)
                self.assertEqual(admission["expected_driver"], "575.57.08")
                self.assertEqual(admission["cpu_affinity"], sorted(os.sched_getaffinity(0)))

    def test_main_holds_existing_leases_and_releases_on_failure(self):
        lease = Mock()
        events = []
        original = runner.core.run_cmd
        def campaign(args):
            self.assertIs(runner.core.run_cmd, runner.run_cmd_owned)
            events.append("campaign")
            return 2
        with (patch.dict(os.environ, {}, clear=True), patch.object(runner.sys, "argv", ["runner"]),
              patch.object(runner, "validate"),
              patch.object(runner.shared, "Leases", side_effect=lambda: events.append("lease") or lease),
              patch.object(runner, "run_campaign", side_effect=campaign)):
            self.assertEqual(runner.main(), 2)
        self.assertEqual(events, ["lease", "campaign"])
        self.assertIs(runner.core.run_cmd, original)
        lease.close.assert_called_once()

    def test_main_restores_cpu_helper_and_lease_on_interrupt(self):
        lease = Mock()
        original = runner.core.run_cmd
        with (patch.dict(os.environ, {}, clear=True), patch.object(runner.sys, "argv", ["runner"]),
              patch.object(runner, "validate"), patch.object(runner.shared, "Leases", return_value=lease),
              patch.object(runner, "run_campaign", side_effect=KeyboardInterrupt("campaign interrupted"))):
            with self.assertRaises(KeyboardInterrupt):
                runner.main()
        self.assertIs(runner.core.run_cmd, original)
        lease.close.assert_called_once()

    def test_ambient_injection_is_rejected_before_launch(self):
        with patch.dict(os.environ, {}, clear=True):
            runner.reject_ambient_injection()
        for key in ("LD_PRELOAD", "LD_AUDIT", "BPFTIME_GLOBAL_SHM_NAME", "OBS_TRACE_LAUNCHES",
                    "GGML_CUDA_ENABLE_UNIFIED_MEMORY", "CUDA_INJECTION64_PATH", "CUDA_VISIBLE_DEVICES"):
            with self.subTest(key=key), patch.dict(os.environ, {key: "foreign"}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "uninjected"):
                    runner.reject_ambient_injection()

    def test_legacy_execution_and_broad_cleanup_are_never_called(self):
        tree = ast.parse(Path(runner.__file__).read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                 and node.func.value.id == "core"]
        self.assertFalse(any(node.func.attr in ("cleanup_gpu", "cleanup_bpftime_shm", "run_tool_once",
                                               "run_llama_once", "start_probe", "stop_probe")
                             for node in calls))

    def test_interrupted_cuda_client_is_stopped_before_return(self):
        process = Mock(pid=98766, returncode=0)
        process.communicate.side_effect = KeyboardInterrupt()
        with (tempfile.TemporaryDirectory() as tmp,
              patch.object(runner.subprocess, "Popen", return_value=process),
              patch.object(runner.shared, "stop_owned") as stop):
            with self.assertRaises(KeyboardInterrupt):
                runner.run_cli_separate(["/client"], cwd=Path(tmp), env={}, timeout=1,
                                        log_path=Path(tmp) / "client.log")
        stop.assert_called_once_with(process)

    def test_cuda_survivor_preserves_loader_segment_and_fatal_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            segment = directory / f"rq4_{os.getpid()}_123"
            loader = Mock(pid=98765, returncode=None)
            loader.poll.return_value = None
            client = Mock(pid=98766, returncode=None)
            client.communicate.side_effect = runner.subprocess.TimeoutExpired(["/client"], 1)
            telemetry, stream = Mock(pid=98767), Mock()
            telemetry.poll.return_value = None
            snapshot = {"gpu": {"driver": "575.57.08"}}
            def start(command, **kwargs):
                if command[-1] == str(directory / "threadhist"):
                    segment.write_text("owned state")
                    return loader
                return client
            def stop(process):
                if process is client:
                    raise RuntimeError("owned client group survived bounded cleanup")
            with (patch.object(runner, "SHM_ROOT", directory),
                  patch.object(runner.time, "monotonic_ns", return_value=123),
                  patch.object(runner.core, "probe_env", return_value={}),
                  patch.object(runner.core, "agent_env", return_value={"LD_PRELOAD": "/agent"}),
                  patch.object(runner.subprocess, "Popen", side_effect=start),
                  patch.object(runner.shared, "stop_owned", side_effect=stop) as stopped,
                  patch.object(runner.shared, "group_members", return_value=[98766]),
                  patch.object(runner.shared.safety, "safety_snapshot", return_value=snapshot),
                  patch.object(runner.shared.safety, "validate_pre_server_safety"),
                  patch.object(runner.shared.safety, "start_gpu_telemetry", return_value=(telemetry, stream, directory / "gpu.csv")),
                  patch.object(runner.shared.safety, "wait_for_post_server_safety", side_effect=RuntimeError("GPU not idle"))):
                with self.assertRaises(runner.OwnedCleanupError) as caught:
                    with runner.cell_safety(directory / "safety"):
                        with runner.private_probe("threadhist", SimpleNamespace(probe_startup_s=0), directory,
                                                  directory / "probe") as env:
                            runner.run_cli_separate(["/client"], cwd=directory, env=env, timeout=1,
                                                    log_path=directory / "client.log")
            self.assertEqual(caught.exception.details["role"], "CUDA client")
            self.assertEqual([call.args[0] for call in stopped.call_args_list], [client, telemetry])
            self.assertTrue(segment.exists())
            record = json.loads((directory / "probe/probe-execution.json").read_text())
            self.assertEqual(record["loader_identity"]["pid"], 98765)
            self.assertEqual(record["client_cleanup_failure"]["identity"]["pid"], 98766)
            self.assertTrue(record["loader_preserved"])
            self.assertFalse(record["private_segment_removed"])
            safety = json.loads((directory / "safety/gpu-safety.json").read_text())
            self.assertFalse(safety["passed"])
            self.assertEqual(safety["fatal_cleanup"]["role"], "CUDA client")

    def test_normal_client_stops_before_private_loader_and_segment(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            segment = directory / f"rq4_{os.getpid()}_123"
            loader, client = Mock(pid=98765, returncode=0), Mock(pid=98766, returncode=0)
            loader.poll.return_value = None
            client.communicate.return_value = ("output", "diagnostics")
            events = []
            def start(command, **kwargs):
                if command[-1] == str(directory / "threadhist"):
                    segment.write_text("owned state")
                    return loader
                return client
            def stop(process):
                self.assertTrue(segment.exists())
                events.append("client" if process is client else "loader")
            with (patch.object(runner, "SHM_ROOT", directory),
                  patch.object(runner.time, "monotonic_ns", return_value=123),
                  patch.object(runner.core, "probe_env", return_value={}),
                  patch.object(runner.core, "agent_env", return_value={}),
                  patch.object(runner.subprocess, "Popen", side_effect=start),
                  patch.object(runner.shared, "stop_owned", side_effect=stop),
                  patch.object(runner.shared, "group_members", return_value=[])):
                with runner.private_probe("threadhist", SimpleNamespace(probe_startup_s=0), directory,
                                          directory / "probe") as env:
                    result = runner.run_cli_separate(["/client"], cwd=directory, env=env, timeout=1,
                                                     log_path=directory / "client.log")
            self.assertEqual(events, ["client", "loader"])
            self.assertFalse(segment.exists())
            self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "output", "diagnostics"))

    def test_fatal_cleanup_stops_correctness_and_timing_campaigns(self):
        for phase in ("correctness", "timing"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                state = {"schedule": {str(block): list(runner.CONFIGS) for block in range(1, 11)},
                         "correctness": {config: {"attempts": []} for config in runner.CONFIGS},
                         "configs": {config: {"runs": []} for config in runner.CONFIGS}}
                (directory / "result.json").write_text(json.dumps(state))
                args = SimpleNamespace(output_dir=directory, resume=True, phase="full", runs=10)
                failure = runner.OwnedCleanupError("target survived", {"role": "CUDA client", "identity": {"pid": 98766}})
                snapshot = {"gpu": "RTX 5090, 575.57.08, 32607, 0, 0, 0", "compute_apps": ""}
                with (patch.object(runner.core, "nvidia_smi_snapshot", return_value=snapshot),
                      patch.object(runner, "verify_resume", return_value={}),
                      patch.object(runner, "valid_correctness", return_value={"valid": True} if phase == "timing" else None),
                      patch.object(runner, "run_correctness_cell", side_effect=failure) as correctness,
                      patch.object(runner, "run_cell", side_effect=failure) as timing,
                      patch.object(runner, "write_state") as write):
                    with self.assertRaises(runner.OwnedCleanupError):
                        runner.run_campaign(args)
                self.assertEqual(correctness.call_count, int(phase == "correctness"))
                self.assertEqual(timing.call_count, int(phase == "timing"))
                recorded = write.call_args.args[1]
                self.assertEqual(recorded["fatal_cleanup"]["identity"]["pid"], 98766)
                entries = recorded["correctness"]["baseline"]["attempts"] if phase == "correctness" else recorded["configs"]["baseline"]["runs"]
                self.assertEqual(len(entries), 1)
                self.assertFalse(entries[0]["valid"])

    def test_bench_preserves_prompt_and_positive_throughput_gate(self):
        args = SimpleNamespace(uvm=False, timeout_s=1, pp=32)
        for tokens, throughput, valid in ((32, 100.0, True), (16, 100.0, False), (32, 0.0, False)):
            output = json.dumps([dict(n_prompt=tokens, n_gen=0, avg_ts=throughput)])
            with (patch.object(runner.core, "make_llama_cmd", return_value=["/client"]),
                  patch.object(runner, "run_cli_separate", return_value=SimpleNamespace(
                      returncode=0, stdout=output, stderr=""))):
                result = runner.run_bench("baseline", 1, args, Path("/output"))
            self.assertEqual(result["valid"], valid)

    def test_private_probe_preserves_unowned_segments(self):
        for defect in (None, "preexisting", "replaced", "early exit", "survivor"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                segment = root / f"rq4_{os.getpid()}_123"
                unrelated = root / "bpftime_maps_shm"
                unrelated.write_text("not ours")
                if defect == "preexisting":
                    segment.write_text("not ours")
                process = Mock(pid=98765, returncode=0)
                process.poll.return_value = 2 if defect == "early exit" else None
                args = SimpleNamespace(probe_startup_s=0)
                def start(*args, **kwargs):
                    segment.write_text("owned loader state")
                    return process
                with (patch.object(runner, "SHM_ROOT", root),
                      patch.object(runner.time, "monotonic_ns", return_value=123),
                      patch.object(runner.core, "probe_env", return_value={"LD_PRELOAD": "/server"}),
                      patch.object(runner.core, "agent_env", return_value={"LD_PRELOAD": "/agent"}),
                      patch.object(runner.subprocess, "Popen", side_effect=start) as popen,
                      patch.object(runner.shared, "stop_owned") as stop,
                      patch.object(runner.shared, "group_members", return_value=[98765] if defect == "survivor" else [])):
                    def run():
                        with runner.private_probe("threadhist", args, root, root / "cell") as env:
                            self.assertEqual(env["BPFTIME_GLOBAL_SHM_NAME"], segment.name)
                            if defect == "replaced":
                                segment.rename(root / "old-owned-segment")
                                segment.write_text("replacement must survive")
                    if defect is None:
                        run()
                    else:
                        with self.assertRaises(RuntimeError):
                            run()
                self.assertEqual(unrelated.read_text(), "not ours")
                self.assertEqual(segment.exists(), defect in ("preexisting", "replaced", "survivor"))
                self.assertEqual(popen.call_count, int(defect != "preexisting"))
                self.assertEqual(stop.call_count, int(defect != "preexisting"))

    def test_clock_patch_matches_real_source_without_modifying_it(self):
        source = runner.core.DEFAULT_BPFTIME_ROOT / "example/gpu/launchlate"
        originals = {name: (source / name).read_text() for name in ("launchlate.c", "launchlate.bpf.c")}
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            for name in originals:
                shutil.copyfile(source / name, target / name)
            runner.patch_launchlate_clock(target)
            self.assertIn("if (gpu_ts < *launch_ts)", (target / "launchlate.bpf.c").read_text())
            self.assertIn("__uint(max_entries, 5)", (target / "launchlate.bpf.c").read_text())
            self.assertIn("Clock errors:", (target / "launchlate.c").read_text())
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                runner.patch_launchlate_clock(target)
        for name, before in originals.items():
            self.assertEqual((source / name).read_text(), before)

    def test_all_probe_paths_need_real_samples_and_complete_clock_counters(self):
        probe = dict(sample_count=2, nonzero_timestamps=2, selected_launches=2, nonzero_threads=1,
                     clock_errors=0, histogram_sum=2, queue_underflows=0, queue_overflows=0,
                     host_launches=2, device_entries=2)
        for check in (runner.gpubpf_probe_valid, runner.nvbit_probe_valid):
            for tool in runner.TASKS:
                self.assertTrue(check(tool, probe))
                self.assertFalse(check(tool, {**probe, "sample_count": 0}))
            self.assertFalse(check("launchlate", {**probe, "clock_errors": 1}))
            self.assertFalse(check("launchlate", {key: value for key, value in probe.items() if key != "clock_errors"}))
        text = "Total samples: 2\nHost launches: 2\nDevice entries: 2\nQueue underflows: 0\nQueue overflows: 0\nClock errors: 0\n"
        self.assertTrue(runner.gpubpf_probe_valid("launchlate", runner.parse_gpubpf("launchlate", text)))
        for label in ("Clock errors: 0\n", "Queue underflows: 0\n", "Queue overflows: 0\n"):
            self.assertFalse(runner.gpubpf_probe_valid("launchlate", runner.parse_gpubpf("launchlate", text.replace(label, ""))))
        self.assertEqual(runner.parse_nvbit("launchlate", "NVBIT launchlate samples=2")["clock_errors"], -1)

    def test_file_metadata_does_not_read_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.bin"
            path.write_bytes(b"ordinary metadata only")
            observed = runner.file_metadata(path)
            self.assertEqual(observed["path"], str(path.absolute()))
            self.assertTrue(observed["exists"])
            self.assertEqual(observed["bytes"], path.stat().st_size)
            self.assertNotIn("content", observed)

    def test_normalized_output_is_compared_exactly(self):
        state = {
            "correctness": {
                "baseline": {
                    "attempts": [
                        {"valid": True, "normalized_stdout": "fixed output"}
                    ]
                },
                "nvbit_launchlate": {
                    "attempts": [
                        {"valid": True, "normalized_stdout": "different output"},
                        {"valid": True, "normalized_stdout": "fixed output"},
                    ]
                },
            }
        }
        selected = runner.valid_correctness(state, "nvbit_launchlate")
        self.assertIsNotNone(selected)
        self.assertEqual(selected["normalized_stdout"], "fixed output")

    def test_normalization_removes_only_presentation_noise(self):
        self.assertEqual(
            runner.normalized_output("\x1b[31mline one\x1b[0m  \nline two\n"),
            "line one\nline two",
        )

    def test_active_runner_has_no_content_fingerprint_logic(self):
        source = Path(runner.__file__).read_text().lower()
        for forbidden in ("hashlib", "sha256", "checksum", "digest"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
