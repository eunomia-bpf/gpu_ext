#!/usr/bin/env python3
"""CPU-only checks for the inactive revision-RQ4 adapter."""

import ast
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import run_revision_rq4 as runner


def lossless_exit_log(**overrides):
    values = {
        "requested": 22528,
        "allocated": 22528,
        "entries": 256,
        "record_bytes": 56,
        "committed": 720896,
        "collected": 720896,
        "runtime_collected": 720896,
        "nonzero": 720896,
        "oob": 0,
        "full": 0,
        "bad_size": 0,
        "other": 0,
        "dirty": 0,
        "pending": 0,
        "final_drain": 720896,
        "second_drain": 0,
        "launches": 220,
        "coordinates": 22528,
        "cartesian_complete": 1,
        "multiplicity_220": 1024,
        "multiplicity_44": 1024,
        "multiplicity_22": 20480,
        "other_multiplicity": 0,
        "segment_mismatches": 0,
        "unique_coordinates": 22528,
        "oracle_enabled": 1,
        "oracle_total_events": 720896,
        "oracle_passed": 1,
        "collector_gate": 1,
    }
    values.update(overrides)
    return "\n".join((
        f"Requested thread slots: {values['requested']}",
        f"Allocated thread slots: {values['allocated']}",
        f"Ring entries per thread: {values['entries']}",
        f"Record bytes: {values['record_bytes']}",
        f"Committed events: {values['committed']}",
        f"Total events collected: {values['collected']}",
        f"Runtime collected events: {values['runtime_collected']}",
        f"Nonzero timestamps: {values['nonzero']}",
        f"OOB drops: {values['oob']}",
        f"Full drops: {values['full']}",
        f"Bad-size drops: {values['bad_size']}",
        f"Other drops: {values['other']}",
        f"Dirty slots: {values['dirty']}",
        f"Pending events: {values['pending']}",
        f"Final drain events: {values['final_drain']}",
        f"Second drain events: {values['second_drain']}",
        f"Cartesian launches: {values['launches']}",
        f"Cartesian coordinates: {values['coordinates']}",
        f"Cartesian complete: {values['cartesian_complete']}",
        f"Coordinate multiplicity 220: {values['multiplicity_220']}",
        f"Coordinate multiplicity 44: {values['multiplicity_44']}",
        f"Coordinate multiplicity 22: {values['multiplicity_22']}",
        f"Coordinate multiplicity other: {values['other_multiplicity']}",
        f"Coordinate segment mismatches: {values['segment_mismatches']}",
        f"Unique coordinates: {values['unique_coordinates']}",
        f"Multiplicity oracle enabled: {values['oracle_enabled']}",
        f"Multiplicity oracle total events: {values['oracle_total_events']}",
        f"Multiplicity oracle passed: {values['oracle_passed']}",
        f"Collector gate passed: {values['collector_gate']}",
    ))


def lossless_launchlate_log(**overrides):
    values = {
        "samples": 2,
        "histogram": 2,
        "host_launches": 2,
        "host_enqueued": 2,
        "device_entries": 2,
        "matched": 2,
        "underflows": 0,
        "overflows": 0,
        "update_errors": 0,
        "classified": 2,
        "uncertain": 0,
        "clock_errors": 0,
        "accounting": 1,
        "pairing": 1,
        "detached": 1,
        "start_low": -120,
        "start_high": -80,
        "start_uncertainty": 20,
        "end_low": -100,
        "end_high": -70,
        "end_uncertainty": 15,
        "endpoint_overlap": 1,
        "intersection_low": -100,
        "intersection_high": -80,
    }
    values.update(overrides)
    return "\n".join((
        "Clock calibration method: bracketed %globaltimer kernel against CLOCK_MONOTONIC",
        f"Start clock offset lower: {values['start_low']} ns",
        f"Start clock offset upper: {values['start_high']} ns",
        f"Start clock uncertainty: {values['start_uncertainty']} ns",
        f"Probes detached before final readback: {values['detached']}",
        f"End clock offset lower: {values['end_low']} ns",
        f"End clock offset upper: {values['end_high']} ns",
        f"End clock uncertainty: {values['end_uncertainty']} ns",
        f"Clock calibration endpoint overlap: {values['endpoint_overlap']}",
        f"Clock offset intersection lower: {values['intersection_low']} ns",
        f"Clock offset intersection upper: {values['intersection_high']} ns",
        f"Histogram samples: {values['histogram']}",
        f"Total samples: {values['samples']}",
        f"Host launches: {values['host_launches']}",
        f"Host enqueued: {values['host_enqueued']}",
        f"Device entries: {values['device_entries']}",
        f"Matched samples: {values['matched']}",
        f"Queue underflows: {values['underflows']}",
        f"Queue overflows: {values['overflows']}",
        f"Queue update errors: {values['update_errors']}",
        f"Classified samples: {values['classified']}",
        f"Uncertain samples: {values['uncertain']}",
        f"Clock errors: {values['clock_errors']}",
        f"Accounting complete: {values['accounting']}",
        f"Pairing complete: {values['pairing']}",
    ))


class OfflineTests(unittest.TestCase):
    def test_correctness_keeps_generated_token_output_enabled(self):
        command = runner.llama_cli_cmd(SimpleNamespace(
            llama_cli=Path("/llama-cli"), model=Path("/model.gguf"), n_gpu_layers=99))
        # llama-cli emits generated tokens through LOG(); pausing that logger
        # suppresses the correctness oracle, not merely diagnostic messages.
        self.assertNotIn("--log-disable", command)
        for option, value in (("-n", "8"), ("--seed", "1797"), ("--temp", "0")):
            self.assertEqual(command[command.index(option) + 1], value)

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
        argv = ["runner", "--bpftime-root", "/source", "--bpftime-build-dir", "/build",
                "--gpu-thread-count", "22528"]
        with (patch.dict(os.environ, {}, clear=True), patch.object(runner.sys, "argv", argv),
              patch.object(runner, "validate"),
              patch.object(runner, "ReadOnlyLeases", side_effect=lambda: events.append("lease") or lease),
              patch.object(runner, "run_campaign", side_effect=campaign)):
            self.assertEqual(runner.main(), 2)
        self.assertEqual(events, ["lease", "campaign"])
        self.assertIs(runner.core.run_cmd, original)
        lease.close.assert_called_once()

    def test_main_restores_cpu_helper_and_lease_on_interrupt(self):
        lease = Mock()
        original = runner.core.run_cmd
        argv = ["runner", "--bpftime-root", "/source", "--bpftime-build-dir", "/build",
                "--gpu-thread-count", "22528"]
        with (patch.dict(os.environ, {}, clear=True), patch.object(runner.sys, "argv", argv),
              patch.object(runner, "validate"), patch.object(runner, "ReadOnlyLeases", return_value=lease),
              patch.object(runner, "run_campaign", side_effect=KeyboardInterrupt("campaign interrupted"))):
            with self.assertRaises(KeyboardInterrupt):
                runner.main()
        self.assertIs(runner.core.run_cmd, original)
        lease.close.assert_called_once()

    def test_read_only_leases_lock_precreated_nonwritable_inodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = tuple(Path(tmp) / name for name in ("gpu0.lock", "struct-ops.lock"))
            for path in paths:
                path.write_text("pre-created by coordinator\n")
                path.chmod(0o444)
            before = [(path.stat().st_dev, path.stat().st_ino, path.stat().st_mode,
                       path.stat().st_size) for path in paths]
            lease = runner.ReadOnlyLeases(paths)
            try:
                self.assertEqual([stream.mode for stream in lease.files], ["r", "r"])
                contender = paths[0].open("r")
                try:
                    with self.assertRaises(BlockingIOError):
                        runner.fcntl.flock(
                            contender.fileno(),
                            runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB,
                        )
                finally:
                    contender.close()
            finally:
                lease.close()
            after = [(path.stat().st_dev, path.stat().st_ino, path.stat().st_mode,
                      path.stat().st_size) for path in paths]
            self.assertEqual(after, before)

    def test_read_only_leases_never_create_missing_path_and_release_partial_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "gpu0.lock"
            missing = Path(tmp) / "struct-ops.lock"
            first.write_text("existing\n")
            with self.assertRaises(FileNotFoundError):
                runner.ReadOnlyLeases((first, missing))
            self.assertFalse(missing.exists())
            stream = first.open("r")
            try:
                runner.fcntl.flock(
                    stream.fileno(), runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB
                )
            finally:
                stream.close()

    def test_read_only_leases_fail_when_an_existing_inode_is_locked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gpu0.lock"
            path.write_text("existing\n")
            blocker = path.open("r")
            try:
                runner.fcntl.flock(
                    blocker.fileno(), runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB
                )
                with self.assertRaises(BlockingIOError):
                    runner.ReadOnlyLeases((path,))
            finally:
                blocker.close()

    def test_read_only_leases_reject_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target.lock"
            link = Path(tmp) / "gpu0.lock"
            target.write_text("existing\n")
            link.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "not a regular file"):
                runner.ReadOnlyLeases((link,))

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

    def test_exit_probe_sets_exact_oracle_only_for_correctness(self):
        for exact in (True, False):
            with self.subTest(exact=exact), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                segment = root / f"rq4_{os.getpid()}_123"
                process = Mock(pid=98765, returncode=0)
                process.poll.return_value = None
                def start(*args, **kwargs):
                    segment.write_text("owned loader state")
                    return process
                args = SimpleNamespace(probe_startup_s=0, gpu_thread_count=22528)
                with (patch.object(runner, "SHM_ROOT", root),
                      patch.object(runner.time, "monotonic_ns", return_value=123),
                      patch.object(runner.core, "probe_env", return_value={}),
                      patch.object(runner.core, "agent_env", return_value={}),
                      patch.object(runner.subprocess, "Popen", side_effect=start) as popen,
                      patch.object(runner.shared, "stop_owned"),
                      patch.object(runner.shared, "group_members", return_value=[])):
                    with runner.private_probe(
                        "kernelretsnoop", args, root, root / "cell",
                        exact_exit_oracle=exact,
                    ) as target_env:
                        self.assertEqual(
                            target_env["BPFTIME_KERNELRETSNOOP_EXACT_ORACLE"],
                            "1" if exact else "0",
                        )
                loader_env = popen.call_args.kwargs["env"]
                self.assertEqual(loader_env["BPFTIME_MAP_GPU_THREAD_COUNT"], "22528")
                self.assertEqual(loader_env["BPFTIME_SHM_MEMORY_MB"], "1000")
                self.assertEqual(
                    loader_env["BPFTIME_KERNELRETSNOOP_EXACT_ORACLE"],
                    "1" if exact else "0",
                )

    def test_native_launchlate_schema_is_validated_without_rewriting_source(self):
        sources = {
            "launchlate.bpf.c": "\n".join((
                "BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP",
                "LAUNCHLATE_TARGET_SYMBOL",
                "MATCHED_SAMPLES",
                "UNCERTAIN_SAMPLES",
            )),
            "launchlate.c": "\n".join((
                "bracketed %%globaltimer kernel against CLOCK_MONOTONIC",
                "Host enqueued:",
                "Matched samples:",
                "Queue update errors:",
                "Uncertain samples:",
                "Accounting complete:",
                "Pairing complete:",
                "Probes detached before final readback:",
                "Clock calibration endpoint overlap:",
            )),
        }
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            for name, text in sources.items():
                (target / name).write_text(text)
            runner.validate_launchlate_source_schema(target)
            self.assertEqual(
                {name: (target / name).read_text() for name in sources}, sources
            )
            (target / "launchlate.c").write_text(
                sources["launchlate.c"].replace("Accounting complete:", "")
            )
            with self.assertRaisesRegex(RuntimeError, "Accounting complete"):
                runner.validate_launchlate_source_schema(target)

        tree = ast.parse(Path(runner.__file__).read_text())
        definitions = {
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertNotIn("patch_launchlate_clock", definitions)
        campaign = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_campaign"
        )
        calls = {
            node.func.id for node in ast.walk(campaign)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("validate_launchlate_source_schema", calls)

    def test_all_probe_paths_need_real_samples_and_complete_clock_counters(self):
        probe = dict(sample_count=2, nonzero_timestamps=2, selected_launches=2, nonzero_threads=1,
                     clock_errors=0, histogram_sum=2, queue_underflows=0, queue_overflows=0,
                     host_launches=2, device_entries=2, configured_entries=4,
                     readback_entries=4, readback_bytes=32, readback_complete=1)
        gpubpf_probes = {tool: probe for tool in runner.TASKS}
        gpubpf_probes["kernelretsnoop"] = runner.parse_gpubpf(
            "kernelretsnoop", lossless_exit_log())
        gpubpf_probes["launchlate"] = runner.parse_gpubpf(
            "launchlate", lossless_launchlate_log())
        for tool, tool_probe in gpubpf_probes.items():
            expected_threads = (runner.EXPECTED_GPU_THREAD_SLOTS
                                if tool == "kernelretsnoop" else 4)
            self.assertTrue(runner.gpubpf_probe_valid(
                tool, tool_probe, expected_thread_count=expected_threads,
                exact_exit_oracle=tool == "kernelretsnoop"))
            self.assertFalse(runner.gpubpf_probe_valid(
                tool, {**tool_probe, "sample_count": 0},
                expected_thread_count=expected_threads,
                exact_exit_oracle=tool == "kernelretsnoop"))
            self.assertTrue(runner.nvbit_probe_valid(tool, probe))
            self.assertFalse(runner.nvbit_probe_valid(
                tool, {**probe, "sample_count": 0}))
        for check in (
            lambda data: runner.gpubpf_probe_valid(
                "launchlate", data, expected_thread_count=4),
            lambda data: runner.nvbit_probe_valid("launchlate", data),
        ):
            self.assertFalse(check({**probe, "clock_errors": 1}))
            self.assertFalse(check({key: value for key, value in probe.items() if key != "clock_errors"}))
        text = lossless_launchlate_log()
        self.assertTrue(runner.gpubpf_probe_valid("launchlate", runner.parse_gpubpf("launchlate", text)))
        for label in (
            "Clock errors: 0", "Queue underflows: 0", "Queue overflows: 0",
            "Host enqueued: 2", "Matched samples: 2", "Queue update errors: 0",
            "Uncertain samples: 0", "Accounting complete: 1",
            "Pairing complete: 1", "Clock calibration endpoint overlap: 1",
            "Probes detached before final readback: 1",
        ):
            self.assertFalse(runner.gpubpf_probe_valid("launchlate", runner.parse_gpubpf("launchlate", text.replace(label, ""))))
        self.assertEqual(runner.parse_nvbit("launchlate", "NVBIT launchlate samples=2")["clock_errors"], -1)

    def test_launchlate_accounting_and_calibration_gate_is_fail_closed(self):
        probe = runner.parse_gpubpf("launchlate", lossless_launchlate_log())
        self.assertTrue(runner.gpubpf_probe_valid("launchlate", probe))
        self.assertEqual(probe["host_enqueued"], 2)
        self.assertEqual(probe["matched_samples"], 2)
        self.assertEqual(probe["clock_intersection_lower_ns"], -100)
        corruptions = {
            "sample_count": 1,
            "histogram_samples": 1,
            "host_launches": 1,
            "host_enqueued": 1,
            "device_entries": 1,
            "matched_samples": 1,
            "classified_samples": 1,
            "queue_underflows": 1,
            "queue_overflows": 1,
            "queue_update_errors": 1,
            "uncertain_samples": 1,
            "clock_errors": 1,
            "accounting_complete": 0,
            "pairing_complete": 0,
            "probes_detached_before_readback": 0,
            "clock_endpoint_overlap": 0,
            "start_clock_uncertainty_ns": 19,
            "end_clock_uncertainty_ns": 14,
            "clock_intersection_lower_ns": -99,
            "clock_intersection_upper_ns": -81,
            "clock_calibration_method": "CLOCK_REALTIME approximation",
        }
        for key, value in corruptions.items():
            with self.subTest(key=key):
                self.assertFalse(runner.gpubpf_probe_valid(
                    "launchlate", {**probe, key: value}
                ))

    def test_lossless_exit_parser_and_correctness_oracle_are_fail_closed(self):
        probe = runner.parse_gpubpf("kernelretsnoop", lossless_exit_log())
        self.assertTrue(runner.gpubpf_probe_valid(
            "kernelretsnoop", probe,
            expected_thread_count=runner.EXPECTED_GPU_THREAD_SLOTS,
            expected_exit_events=runner.CORRECTNESS_EXIT_EVENTS,
            expected_exit_launches=runner.CORRECTNESS_EXIT_LAUNCHES,
            expected_exit_coordinates=runner.CORRECTNESS_EXIT_COORDINATES,
            exact_exit_oracle=True,
        ))
        for key in (
            "sample_count", "nonzero_timestamps", "requested_thread_slots",
            "allocated_thread_slots", "entries_per_thread", "record_bytes",
            "committed_events", "runtime_collected_events", "oob_drops",
            "full_drops", "bad_size_drops", "other_drops", "dirty_slots",
            "pending_events", "final_drain_events", "second_drain_events",
            "cartesian_launches", "cartesian_coordinates", "cartesian_complete",
            "multiplicity_220", "multiplicity_44", "multiplicity_22",
            "other_multiplicity", "segment_mismatches", "unique_coordinates",
            "oracle_enabled", "oracle_total_events", "oracle_passed",
            "collector_gate_passed",
        ):
            with self.subTest(key=key):
                if key == "final_drain_events":
                    bad_value = -1
                elif key in ("cartesian_complete", "oracle_enabled", "oracle_passed",
                             "collector_gate_passed"):
                    bad_value = 0
                else:
                    bad_value = 1
                broken = {**probe, key: bad_value}
                self.assertFalse(runner.gpubpf_probe_valid(
                    "kernelretsnoop", broken,
                    expected_thread_count=runner.EXPECTED_GPU_THREAD_SLOTS,
                    expected_exit_events=runner.CORRECTNESS_EXIT_EVENTS,
                    expected_exit_launches=runner.CORRECTNESS_EXIT_LAUNCHES,
                    expected_exit_coordinates=runner.CORRECTNESS_EXIT_COORDINATES,
                    exact_exit_oracle=True,
                ))

        # Preserve the bin counts while moving them to the wrong coordinate
        # segments: the collector exposes this separately from count equality.
        swapped_segment = {**probe, "segment_mismatches": 2}
        self.assertFalse(runner.gpubpf_probe_valid(
            "kernelretsnoop", swapped_segment,
            expected_thread_count=runner.EXPECTED_GPU_THREAD_SLOTS,
            expected_exit_events=runner.CORRECTNESS_EXIT_EVENTS,
            expected_exit_launches=runner.CORRECTNESS_EXIT_LAUNCHES,
            expected_exit_coordinates=runner.CORRECTNESS_EXIT_COORDINATES,
            exact_exit_oracle=True,
        ))

        nvbit = runner.parse_nvbit(
            "kernelretsnoop",
            "NVBIT selected_launches=220\n"
            "NVBIT kernelretsnoop events=720896 nonzero_timestamps=720896\n",
        )
        self.assertTrue(runner.nvbit_probe_valid(
            "kernelretsnoop", nvbit,
            expected_exit_events=runner.CORRECTNESS_EXIT_EVENTS,
            expected_exit_launches=runner.CORRECTNESS_EXIT_LAUNCHES,
        ))
        self.assertFalse(runner.nvbit_probe_valid(
            "kernelretsnoop", {**nvbit, "selected_launches": 175},
            expected_exit_events=runner.CORRECTNESS_EXIT_EVENTS,
            expected_exit_launches=runner.CORRECTNESS_EXIT_LAUNCHES,
        ))

    def test_timed_exit_gate_accepts_accounted_nonuniform_multiplicity(self):
        text = lossless_exit_log(
            committed=40, collected=40, runtime_collected=40, nonzero=40,
            final_drain=8, launches=5, coordinates=6,
            multiplicity_220=0, multiplicity_44=0, multiplicity_22=0,
            other_multiplicity=6, segment_mismatches=6,
            unique_coordinates=6, oracle_enabled=0,
            oracle_total_events=40, oracle_passed=0,
        )
        probe = runner.parse_gpubpf("kernelretsnoop", text)
        # Timed workloads need complete coordinate accounting, but their
        # nonuniform multiplicities need not satisfy the llama-cli oracle.
        self.assertNotEqual(5 * 6, 40)
        self.assertTrue(runner.gpubpf_probe_valid(
            "kernelretsnoop", probe,
            expected_thread_count=runner.EXPECTED_GPU_THREAD_SLOTS,
            exact_exit_oracle=False,
        ))
        self.assertFalse(runner.gpubpf_probe_valid(
            "kernelretsnoop", {**probe, "oracle_enabled": 1},
            expected_thread_count=runner.EXPECTED_GPU_THREAD_SLOTS,
            exact_exit_oracle=False,
        ))

    def test_timed_exit_pairs_require_equal_events_and_launches(self):
        for field in (None, "events", "launches"):
            with self.subTest(field=field):
                gpubpf = {"block": 1, "valid": True,
                           "probe": {"sample_count": 40, "cartesian_launches": 5}}
                nvbit = {"block": 1, "valid": True,
                         "probe": {"sample_count": 40, "selected_launches": 5}}
                if field == "events":
                    nvbit["probe"]["sample_count"] = 39
                if field == "launches":
                    nvbit["probe"]["selected_launches"] = 4
                state = {"configs": {
                    "gpubpf_kernelretsnoop": {"runs": [gpubpf]},
                    "nvbit_kernelretsnoop": {"runs": [nvbit]},
                }}
                runner.reconcile_kernelret_block(state, 1)
                self.assertEqual(gpubpf["valid"], field is None)
                self.assertEqual(nvbit["valid"], field is None)
                self.assertEqual(gpubpf["kernelret_pair"]["matched"], field is None)

    def test_correctness_requires_the_exact_47_byte_oracle(self):
        self.assertEqual(len(runner.EXPECTED_NORMALIZED_STDOUT.encode()), 47)
        args = SimpleNamespace(uvm=False, timeout_s=1, llama_cli=Path("/llama-cli"),
                               model=Path("/model"), n_gpu_layers=99)
        for output, valid in ((runner.EXPECTED_NORMALIZED_STDOUT, True),
                              (runner.EXPECTED_NORMALIZED_STDOUT + "!", False)):
            with (tempfile.TemporaryDirectory() as tmp,
                  patch.object(runner, "idle_gpu_or_error"),
                  patch.object(runner.core, "nvidia_smi_snapshot", return_value={}),
                  patch.object(runner, "cell_safety", return_value=runner.nullcontext({})),
                  patch.object(runner, "run_cli_separate", return_value=SimpleNamespace(
                      returncode=0, stdout=output, stderr=""))):
                result = runner.run_correctness_cell(
                    "baseline", 1, args, Path(tmp), {})
            self.assertEqual(result["valid"], valid)
            self.assertEqual(result["stdout_bytes"], len(output.encode()))

    def test_threadhist_full_width_readback_including_zero_tail(self):
        sentinel = (1 << 64) - 1
        expected = 4096
        for copied, valid in (([8] * 1024, False),
                              ([8] * 1024 + [0] * 3072, True)):
            # CPU double for the real lookup output buffer. Unwritten entries
            # remain sentinel; legitimately zero GPU entries are still copied.
            values = [sentinel] * expected
            values[:len(copied)] = copied
            observed = sum(value != sentinel for value in values)
            text = (f"Configured thread entries: {expected}\n"
                    f"Readback entries: {observed}\nReadback bytes: {observed * 8}\n"
                    f"Readback complete: {int(observed == expected)}\n"
                    "Nonzero threads: 1024\nTotal exit probes: 8192\n")
            probe = runner.parse_gpubpf("threadhist", text)
            self.assertEqual(runner.gpubpf_probe_valid(
                "threadhist", probe, expected_thread_count=expected), valid)
            if valid:
                self.assertFalse(runner.gpubpf_probe_valid("threadhist", probe))
                self.assertFalse(runner.gpubpf_probe_valid(
                    "threadhist", probe, expected_thread_count=1048576))
                for key in ("configured_entries", "readback_entries", "readback_bytes", "readback_complete"):
                    incomplete = {name: value for name, value in probe.items() if name != key}
                    self.assertFalse(runner.gpubpf_probe_valid(
                        "threadhist", incomplete, expected_thread_count=expected))
                self.assertFalse(runner.gpubpf_probe_valid(
                    "threadhist", {**probe, "readback_bytes": 8192}, expected_thread_count=expected))

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
