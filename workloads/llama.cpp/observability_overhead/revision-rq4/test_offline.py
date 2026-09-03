#!/usr/bin/env python3
"""CPU-only checks for the inactive revision-RQ4 adapter."""

import ast
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
        with (patch.dict(os.environ, {}, clear=True), patch.object(runner.sys, "argv", ["runner"]),
              patch.object(runner, "validate"),
              patch.object(runner.shared, "Leases", side_effect=lambda: events.append("lease") or lease),
              patch.object(runner, "run_campaign", side_effect=lambda args: events.append("campaign") or 2)):
            self.assertEqual(runner.main(), 2)
        self.assertEqual(events, ["lease", "campaign"])
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
        process = Mock()
        process.communicate.side_effect = KeyboardInterrupt()
        with (tempfile.TemporaryDirectory() as tmp,
              patch.object(runner.subprocess, "Popen", return_value=process),
              patch.object(runner.shared, "stop_owned") as stop):
            with self.assertRaises(KeyboardInterrupt):
                runner.run_cli_separate(["/client"], cwd=Path(tmp), env={}, timeout=1,
                                        log_path=Path(tmp) / "client.log")
        stop.assert_called_once_with(process)

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
