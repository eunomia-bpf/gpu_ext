#!/usr/bin/env python3
"""CPU-only checks for the inactive revision-RQ4 adapter."""

import ast
import io
import json
import os
import subprocess
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
        "requested_entries": 256,
        "entries": 256,
        "record_bytes": 32,
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
        "extent_x": 88,
        "extent_y": 256,
        "extent_z": 1,
        "multiplicity_220": 1024,
        "multiplicity_44": 1024,
        "multiplicity_22": 20480,
        "other_multiplicity": 0,
        "segment_mismatches": 0,
        "invalid_launch_coordinates": 0,
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
        f"Requested ring entries per thread: {values['requested_entries']}",
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
        f"Coordinate extent x: {values['extent_x']}",
        f"Coordinate extent y: {values['extent_y']}",
        f"Coordinate extent z: {values['extent_z']}",
        f"Coordinate multiplicity 220: {values['multiplicity_220']}",
        f"Coordinate multiplicity 44: {values['multiplicity_44']}",
        f"Coordinate multiplicity 22: {values['multiplicity_22']}",
        f"Coordinate multiplicity other: {values['other_multiplicity']}",
        f"Coordinate segment mismatches: {values['segment_mismatches']}",
        f"Invalid launch coordinates: {values['invalid_launch_coordinates']}",
        f"Unique coordinates: {values['unique_coordinates']}",
        f"Multiplicity oracle enabled: {values['oracle_enabled']}",
        f"Multiplicity oracle total events: {values['oracle_total_events']}",
        f"Multiplicity oracle passed: {values['oracle_passed']}",
        f"Collector gate passed: {values['collector_gate']}",
    ))


def lossless_nvbit_exit_log(**overrides):
    values = {
        "selected": 220,
        "events": 720896,
        "nonzero": 720896,
        "record_bytes": 32,
        "bad_size_bytes": 0,
        "launches": 220,
        "coordinates": 22528,
        "complete": 1,
        "extent_x": 88,
        "extent_y": 256,
        "extent_z": 1,
        "multiplicity_220": 1024,
        "multiplicity_44": 1024,
        "multiplicity_22": 20480,
        "multiplicity_other": 0,
        "segment_mismatches": 0,
        "invalid_coordinates": 0,
        "unique_coordinates": 22528,
        "collector_gate": 1,
    }
    values.update(overrides)
    return "\n".join((
        f"NVBIT kernelretsnoop record_bytes={values['record_bytes']}",
        f"NVBIT kernelretsnoop bad_size_bytes={values['bad_size_bytes']}",
        f"NVBIT kernelretsnoop cartesian_launches={values['launches']}",
        f"NVBIT kernelretsnoop cartesian_coordinates={values['coordinates']}",
        f"NVBIT kernelretsnoop cartesian_complete={values['complete']}",
        f"NVBIT kernelretsnoop extent_x={values['extent_x']} extent_y={values['extent_y']} extent_z={values['extent_z']}",
        f"NVBIT kernelretsnoop multiplicity_220={values['multiplicity_220']} multiplicity_44={values['multiplicity_44']} multiplicity_22={values['multiplicity_22']} multiplicity_other={values['multiplicity_other']}",
        f"NVBIT kernelretsnoop segment_mismatches={values['segment_mismatches']} invalid_coordinates={values['invalid_coordinates']} unique_coordinates={values['unique_coordinates']}",
        f"NVBIT kernelretsnoop collector_gate_passed={values['collector_gate']}",
        f"NVBIT selected_launches={values['selected']}",
        f"NVBIT kernelretsnoop events={values['events']} nonzero_timestamps={values['nonzero']}",
        f"NVBIT_OBS process_selected_launches={values['selected']}",
    ))


def lossless_launchlate_log(**overrides):
    values = {
        "samples": 220,
        "histogram": 220,
        "host_launches": 220,
        "host_enqueued": 220,
        "device_entries": 220,
        "matched": 220,
        "underflows": 0,
        "overflows": 0,
        "update_errors": 0,
        "classified": 220,
        "uncertain": 0,
        "clock_errors": 0,
        "online_accounting": 1,
        "accounting": 1,
        "pairing": 1,
        "detached": 1,
        "start_low": 868,
        "start_high": 1032,
        "start_uncertainty": 82,
        "start_anchor": 1000000050,
        "end_low": 888,
        "end_high": 1052,
        "end_uncertainty": 82,
        "end_anchor": 2000000050,
        "change_low": -144,
        "change_high": 184,
        "elapsed": 1000000000,
        "drift_rate": 184,
        "drift_limit": 10000,
        "drift_bounded": 1,
        "rm_samples": 32,
        "rm_accepted": 32,
        "rm_rejected": 0,
        "start_outer_before": 999999900,
        "start_cpu_before": 1000000000,
        "start_gpu": 1000001000,
        "start_cpu_after": 1000000100,
        "start_outer_after": 1000000200,
        "end_outer_before": 1999999900,
        "end_cpu_before": 2000000000,
        "end_gpu": 2000001020,
        "end_cpu_after": 2000000100,
        "end_outer_after": 2000000200,
        "rm_outer_width": 300,
        "rm_selected_gap": 100,
        "rm_bracket_width": 164,
        "rm_status": 0,
        "rm_cleanup": 1,
    }
    values.update(overrides)
    return "\n".join((
        "Clock calibration method: " + runner.GPUBPF_LAUNCH_CLOCK_METHOD,
        f"Start clock offset lower: {values['start_low']} ns",
        f"Start clock offset upper: {values['start_high']} ns",
        f"Start clock uncertainty: {values['start_uncertainty']} ns",
        f"Start clock host anchor: {values['start_anchor']} ns",
        f"Start RM samples requested: {values['rm_samples']}",
        f"Start RM samples accepted: {values['rm_accepted']}",
        f"Start RM samples rejected: {values['rm_rejected']}",
        f"Start RM outer before RAW: {values['start_outer_before']} ns",
        f"Start RM CPU before RAW: {values['start_cpu_before']} ns",
        f"Start RM GPU PTIMER: {values['start_gpu']} ns",
        f"Start RM CPU after RAW: {values['start_cpu_after']} ns",
        f"Start RM outer after RAW: {values['start_outer_after']} ns",
        f"Start RM outer width: {values['rm_outer_width']} ns",
        f"Start RM selected gap: {values['rm_selected_gap']} ns",
        f"Start RM bracket width: {values['rm_bracket_width']} ns",
        f"Start RM status: 0x{values['rm_status']:08x}",
        f"Start RM cleanup complete: {values['rm_cleanup']}",
        f"Probes detached before final readback: {values['detached']}",
        f"End clock offset lower: {values['end_low']} ns",
        f"End clock offset upper: {values['end_high']} ns",
        f"End clock uncertainty: {values['end_uncertainty']} ns",
        f"End clock host anchor: {values['end_anchor']} ns",
        f"End RM samples requested: {values['rm_samples']}",
        f"End RM samples accepted: {values['rm_accepted']}",
        f"End RM samples rejected: {values['rm_rejected']}",
        f"End RM outer before RAW: {values['end_outer_before']} ns",
        f"End RM CPU before RAW: {values['end_cpu_before']} ns",
        f"End RM GPU PTIMER: {values['end_gpu']} ns",
        f"End RM CPU after RAW: {values['end_cpu_after']} ns",
        f"End RM outer after RAW: {values['end_outer_after']} ns",
        f"End RM outer width: {values['rm_outer_width']} ns",
        f"End RM selected gap: {values['rm_selected_gap']} ns",
        f"End RM bracket width: {values['rm_bracket_width']} ns",
        f"End RM status: 0x{values['rm_status']:08x}",
        f"End RM cleanup complete: {values['rm_cleanup']}",
        f"Clock offset change lower: {values['change_low']} ns",
        f"Clock offset change upper: {values['change_high']} ns",
        f"Clock calibration elapsed: {values['elapsed']} ns",
        f"Clock drift rate bound: {values['drift_rate']} ppb",
        f"Clock drift limit: {values['drift_limit']} ppb",
        f"Clock drift bounded: {values['drift_bounded']}",
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
        f"Online accounting complete: {values['online_accounting']}",
        f"Accounting complete: {values['accounting']}",
        f"Pairing complete: {values['pairing']}",
    ))


def lossless_nvbit_launchlate_log(**overrides):
    values = {
        "selected": 220,
        "process_selected": 220,
        "samples": 220,
        "uncertain": 0,
        "clock_errors": 0,
        "pair_capacity": 65536,
        "stored_pairs": 220,
        "device_entries": 220,
        "pair_overflows": 0,
        "capture_errors": 0,
        "selected_counter_overflow": 0,
        "accounting": 1,
        "start_low": 868,
        "start_high": 1032,
        "start_uncertainty": 82,
        "start_anchor": 1000000050,
        "start_valid": 1,
        "end_low": 888,
        "end_high": 1052,
        "end_uncertainty": 82,
        "end_anchor": 2000000050,
        "end_valid": 1,
        "change_low": -144,
        "change_high": 184,
        "elapsed": 1000000000,
        "drift_rate": 184,
        "drift_limit": 10000,
        "drift_bounded": 1,
        "rm_samples": 32,
        "rm_accepted": 32,
        "rm_rejected": 0,
        "start_outer_before": 999999900,
        "start_cpu_before": 1000000000,
        "start_gpu": 1000001000,
        "start_cpu_after": 1000000100,
        "start_outer_after": 1000000200,
        "end_outer_before": 1999999900,
        "end_cpu_before": 2000000000,
        "end_gpu": 2000001020,
        "end_cpu_after": 2000000100,
        "end_outer_after": 2000000200,
        "rm_outer_width": 300,
        "rm_selected_gap": 100,
        "rm_bracket_width": 164,
        "rm_status": 0,
        "rm_cleanup": 1,
    }
    values.update(overrides)
    if "stored_pairs" not in overrides:
        values["stored_pairs"] = values["selected"]
    if "device_entries" not in overrides:
        values["device_entries"] = values["selected"]
    if "process_selected" not in overrides:
        values["process_selected"] = values["selected"]
    bins = [0, values["samples"], 0, 0, 0, 0, 0, 0, 0, 0]
    return "\n".join((
        "NVBIT launchlate clock_calibration_method=" + runner.NVBIT_LAUNCH_CLOCK_METHOD,
        f"NVBIT launchlate start_clock_offset_lower_ns={values['start_low']}",
        f"NVBIT launchlate start_clock_offset_upper_ns={values['start_high']}",
        f"NVBIT launchlate start_clock_uncertainty_ns={values['start_uncertainty']}",
        f"NVBIT launchlate start_clock_host_anchor_ns={values['start_anchor']}",
        f"NVBIT launchlate start_clock_calibration_valid={values['start_valid']}",
        f"NVBIT launchlate start_rm_samples_requested={values['rm_samples']}",
        f"NVBIT launchlate start_rm_samples_accepted={values['rm_accepted']}",
        f"NVBIT launchlate start_rm_samples_rejected={values['rm_rejected']}",
        f"NVBIT launchlate start_rm_outer_before_raw_ns={values['start_outer_before']}",
        f"NVBIT launchlate start_rm_cpu_before_raw_ns={values['start_cpu_before']}",
        f"NVBIT launchlate start_rm_gpu_ptimer_ns={values['start_gpu']}",
        f"NVBIT launchlate start_rm_cpu_after_raw_ns={values['start_cpu_after']}",
        f"NVBIT launchlate start_rm_outer_after_raw_ns={values['start_outer_after']}",
        f"NVBIT launchlate start_rm_outer_width_ns={values['rm_outer_width']}",
        f"NVBIT launchlate start_rm_selected_gap_ns={values['rm_selected_gap']}",
        f"NVBIT launchlate start_rm_bracket_width_ns={values['rm_bracket_width']}",
        f"NVBIT launchlate start_rm_status={values['rm_status']}",
        f"NVBIT launchlate start_rm_cleanup_complete={values['rm_cleanup']}",
        f"NVBIT launchlate end_clock_offset_lower_ns={values['end_low']}",
        f"NVBIT launchlate end_clock_offset_upper_ns={values['end_high']}",
        f"NVBIT launchlate end_clock_uncertainty_ns={values['end_uncertainty']}",
        f"NVBIT launchlate end_clock_host_anchor_ns={values['end_anchor']}",
        f"NVBIT launchlate end_clock_calibration_valid={values['end_valid']}",
        f"NVBIT launchlate end_rm_samples_requested={values['rm_samples']}",
        f"NVBIT launchlate end_rm_samples_accepted={values['rm_accepted']}",
        f"NVBIT launchlate end_rm_samples_rejected={values['rm_rejected']}",
        f"NVBIT launchlate end_rm_outer_before_raw_ns={values['end_outer_before']}",
        f"NVBIT launchlate end_rm_cpu_before_raw_ns={values['end_cpu_before']}",
        f"NVBIT launchlate end_rm_gpu_ptimer_ns={values['end_gpu']}",
        f"NVBIT launchlate end_rm_cpu_after_raw_ns={values['end_cpu_after']}",
        f"NVBIT launchlate end_rm_outer_after_raw_ns={values['end_outer_after']}",
        f"NVBIT launchlate end_rm_outer_width_ns={values['rm_outer_width']}",
        f"NVBIT launchlate end_rm_selected_gap_ns={values['rm_selected_gap']}",
        f"NVBIT launchlate end_rm_bracket_width_ns={values['rm_bracket_width']}",
        f"NVBIT launchlate end_rm_status={values['rm_status']}",
        f"NVBIT launchlate end_rm_cleanup_complete={values['rm_cleanup']}",
        f"NVBIT launchlate clock_offset_change_lower_ns={values['change_low']}",
        f"NVBIT launchlate clock_offset_change_upper_ns={values['change_high']}",
        f"NVBIT launchlate clock_calibration_elapsed_ns={values['elapsed']}",
        f"NVBIT launchlate clock_drift_rate_bound_ppb={values['drift_rate']}",
        f"NVBIT launchlate clock_drift_limit_ppb={values['drift_limit']}",
        f"NVBIT launchlate clock_drift_bounded={values['drift_bounded']}",
        *(f"NVBIT launchlate bin_{index}={count}"
          for index, count in enumerate(bins)),
        f"NVBIT launchlate pair_capacity={values['pair_capacity']}",
        f"NVBIT launchlate stored_pairs={values['stored_pairs']}",
        f"NVBIT launchlate device_entries={values['device_entries']}",
        f"NVBIT launchlate pair_overflows={values['pair_overflows']}",
        f"NVBIT launchlate capture_errors={values['capture_errors']}",
        "NVBIT launchlate selected_counter_overflow="
        f"{values['selected_counter_overflow']}",
        f"NVBIT launchlate uncertain_samples={values['uncertain']}",
        f"NVBIT launchlate samples={values['samples']} "
        f"clock_errors={values['clock_errors']}",
        f"NVBIT launchlate accounting_complete={values['accounting']}",
        f"NVBIT selected_launches={values['selected']}",
        f"NVBIT_OBS process_selected_launches={values['process_selected']}",
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
                      patch.object(runner, "new_state", return_value={}),
                      patch.object(runner, "write_state"),
                      patch.object(runner, "run_launch_clock_controls", return_value=None),
                      patch.object(runner.shutil, "copytree"),
                      patch.object(runner, "validate_nvbit_kernelretsnoop_source_schema"),
                      patch.object(runner, "validate_nvbit_launchlate_source_schema"),
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

    def test_main_uses_validated_inherited_leases_without_reacquiring(self):
        argv = ["runner", "--bpftime-root", "/source", "--bpftime-build-dir", "/build",
                "--gpu-thread-count", "22528", "--inherited-lease-fds", "71", "72"]
        with (patch.dict(os.environ, {}, clear=True), patch.object(runner.sys, "argv", argv),
              patch.object(runner, "validate"),
              patch.object(runner, "validate_inherited_lease_fds") as inherited,
              patch.object(runner, "ReadOnlyLeases") as local,
              patch.object(runner, "run_campaign", return_value=0)):
            self.assertEqual(runner.main(), 0)
        inherited.assert_called_once_with((71, 72))
        local.assert_not_called()

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

    def test_inherited_leases_require_exact_read_only_inodes(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = tuple(Path(temp) / name for name in ("gpu.lock", "struct.lock"))
            for path in paths:
                path.touch(mode=0o644)
            descriptors = tuple(os.open(path, os.O_RDONLY) for path in paths)
            try:
                with patch.object(runner, "LEASE_PATHS", paths):
                    inventory = runner.validate_inherited_lease_fds(descriptors)
                    self.assertEqual([item["path"] for item in inventory],
                                     [str(path) for path in paths])
                    with self.assertRaises(RuntimeError):
                        runner.validate_inherited_lease_fds((descriptors[0],))
                writable = os.open(paths[1], os.O_RDWR)
                try:
                    with patch.object(runner, "LEASE_PATHS", paths), \
                         self.assertRaises(RuntimeError):
                        runner.validate_inherited_lease_fds(
                            (descriptors[0], writable))
                finally:
                    os.close(writable)
            finally:
                for descriptor in descriptors:
                    os.close(descriptor)

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

    def test_cuda_client_never_inherits_parent_stdin(self):
        process = Mock(pid=98766, returncode=0)
        process.communicate.return_value = (
            runner.EXPECTED_NORMALIZED_STDOUT, "diagnostics"
        )
        with (tempfile.TemporaryDirectory() as tmp,
              patch.object(runner.subprocess, "Popen", return_value=process) as popen,
              patch.object(runner.shared, "stop_owned")):
            runner.run_cli_separate(
                ["/client"], cwd=Path(tmp), env={}, timeout=1,
                log_path=Path(tmp) / "client.log",
            )
        self.assertIs(popen.call_args.kwargs["stdin"], subprocess.DEVNULL)

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
                      patch.object(runner, "replay_launch_clock_controls", return_value=None),
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
                args = SimpleNamespace(probe_startup_s=0, gpu_thread_count=22528, pp=32)
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
                expected_slots = "22528" if exact else "32768"
                expected_entries = "256" if exact else "44"
                self.assertEqual(loader_env["BPFTIME_MAP_GPU_THREAD_COUNT"], expected_slots)
                self.assertEqual(loader_env["BPFTIME_KERNELRETSNOOP_RING_ENTRIES"], expected_entries)
                self.assertEqual(loader_env["BPFTIME_SHM_MEMORY_MB"], "1000")
                self.assertEqual(
                    loader_env["BPFTIME_KERNELRETSNOOP_EXACT_ORACLE"],
                    "1" if exact else "0",
                )

    def test_native_launchlate_schema_is_validated_without_rewriting_source(self):
        sources = {
            "Makefile": "rm_ptimer_575.c\nrm_ptimer_575.o",
            "launchlate.bpf.c": "\n".join((
                "BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP",
                "LAUNCHLATE_TARGET_SYMBOL",
                "MATCHED_SAMPLES",
                "UNCERTAIN_SAMPLES",
                "gpu_entry_ns",
                "host_raw_ns",
                "bpftime_ktime_get_raw_ns",
            )),
            "launchlate.c": "\n".join((
                runner.GPUBPF_LAUNCH_CLOCK_METHOD,
                "rm_ptimer_575_sample",
                "RM cleanup complete:",
                "Host enqueued:",
                "Matched samples:",
                "Queue update errors:",
                "Uncertain samples:",
                "Accounting complete:",
                "Online accounting complete:",
                "Pairing complete:",
                "Probes detached before final readback:",
                "Clock drift rate bound:",
                "Clock drift bounded:",
                "classify_affine_sample(",
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

    def test_native_kernelretsnoop_schema_requires_compact_coordinate_abi(self):
        sources = {
            "kernelretsnoop.bpf.c": "\n".join((
                "u64 coordinate_x, coordinate_y, coordinate_z;",
                "data.coordinate_x = block_x * block_dim_x + thread_x;",
                "data.coordinate_y = block_y * block_dim_y + thread_y;",
                "data.coordinate_z = block_z * block_dim_z + thread_z;",
                "sizeof(struct data)",
            )),
            "kernelretsnoop.c": "\n".join((
                "uint64_t coordinate_x, coordinate_y, coordinate_z;",
                "event_coordinate(&state->events[i]",
                "Invalid launch coordinates:",
                "sizeof(struct data)",
                "BPFTIME_KERNELRETSNOOP_RING_ENTRIES",
                "bpf_map__set_max_entries(skel->maps.rb, requested_entries)",
                "Requested ring entries per thread:",
            )),
        }
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            for name, text in sources.items():
                (target / name).write_text(text)
            runner.validate_kernelretsnoop_source_schema(target)
            self.assertEqual(
                {name: (target / name).read_text() for name in sources}, sources
            )
            (target / "kernelretsnoop.bpf.c").write_text(
                sources["kernelretsnoop.bpf.c"].replace(
                    "data.coordinate_x = block_x * block_dim_x + thread_x;", ""
                )
            )
            with self.assertRaisesRegex(RuntimeError, "data.coordinate_x"):
                runner.validate_kernelretsnoop_source_schema(target)

        tree = ast.parse(Path(runner.__file__).read_text())
        campaign = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_campaign"
        )
        calls = {
            node.func.id for node in ast.walk(campaign)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("validate_kernelretsnoop_source_schema", calls)

    def test_nvbit_kernelretsnoop_schema_requires_same_compact_abi(self):
        names = ("common.h", "inject_funcs.cu", "observability.cu")
        sources = {
            name: (runner.NVBIT_SOURCE_DIR / name).read_text()
            for name in names
        }
        runner.validate_nvbit_kernelretsnoop_source_schema(runner.NVBIT_SOURCE_DIR)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            for name, text in sources.items():
                (target / name).write_text(text)
            (target / "common.h").write_text(
                sources["common.h"].replace("uint64_t coordinate_x;", "")
            )
            with self.assertRaisesRegex(RuntimeError, "coordinate_x"):
                runner.validate_nvbit_kernelretsnoop_source_schema(target)

    def test_nvbit_launchlate_schema_requires_bounded_clock_accounting(self):
        names = (
            "Makefile", "clock_domain.h", "common.h", "inject_funcs.cu",
            "observability.cu", "rm_ptimer_575.c",
        )
        sources = {
            name: (runner.NVBIT_SOURCE_DIR / name).read_text()
            for name in names
        }
        runner.validate_nvbit_launchlate_source_schema(runner.NVBIT_SOURCE_DIR)
        self.assertEqual({
            name: (runner.NVBIT_SOURCE_DIR / name).read_text()
            for name in names
        }, sources)
        corruptions = (
            ("Makefile", "observability.o: observability.cu common.h clock_domain.h"),
            ("clock_domain.h", "int64_t offset_low_ns;"),
            ("clock_domain.h", "CLOCK_MIN_CALIBRATION_SPAN_NS"),
            ("common.h", "struct launch_pair_t"),
            ("inject_funcs.cu", 'asm volatile("mov.u64 %0, %%globaltimer;"'),
            ("inject_funcs.cu", "pair->gpu_entry_ns = gpu_ns"),
            ("observability.cu", "clock_drift_bounded="),
            ("observability.cu", "wait_for_minimum_clock_span("),
            ("rm_ptimer_575.c", "RM_ENDPOINTS_V1_COMMAND"),
        )
        for broken_name, marker in corruptions:
            with self.subTest(source=broken_name), tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp)
                for name, text in sources.items():
                    path = target / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        text.replace(marker, "") if name == broken_name else text
                    )
                with self.assertRaises(RuntimeError) as caught:
                    runner.validate_nvbit_launchlate_source_schema(target)
                self.assertIn(marker, str(caught.exception))

        tree = ast.parse(Path(runner.__file__).read_text())
        campaign = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_campaign"
        )
        calls = {
            node.func.id for node in ast.walk(campaign)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("validate_nvbit_launchlate_source_schema", calls)
        manifest = runner.source_manifest(
            SimpleNamespace(bpftime_root=Path("/missing-bpftime"))
        )
        self.assertIn(
            str(runner.NVBIT_SOURCE_DIR / "clock_domain.h"), manifest
        )

    def test_all_copied_tool_runtime_includes_are_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bpftime_root = root / "bpftime"
            (bpftime_root / "runtime/include").mkdir(parents=True)
            expected = str((bpftime_root / "runtime/include").resolve())
            for tool in runner.TASKS:
                with self.subTest(tool=tool):
                    directory = root / tool
                    directory.mkdir()
                    makefile = directory / "Makefile"
                    makefile.write_text(
                        "INCLUDES := -I../../../runtime/include\n"
                        "RUNTIME_HEADER := ../../../runtime/include/bpftime_gpu_ringbuf.h\n"
                    )
                    spec = runner.core.TOOLS[tool]
                    with patch.object(
                        runner.core, "prepare_tool_source", return_value=directory
                    ) as copied:
                        self.assertEqual(
                            runner.prepare_tool_source(
                                spec,
                                bpftime_root=bpftime_root,
                                build_root=root / "build",
                                target_symbol="target",
                            ),
                            directory,
                        )
                    copied.assert_called_once_with(
                        spec,
                        bpftime_root=bpftime_root,
                        build_root=root / "build",
                        target_symbol="target",
                    )
                    text = makefile.read_text()
                    self.assertNotIn(runner.RELATIVE_RUNTIME_INCLUDE, text)
                    self.assertEqual(text.count(expected), 2)

    def test_copied_tool_stale_runtime_include_marker_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "kernelretsnoop"
            directory.mkdir()
            makefile = directory / "Makefile"
            original = "INCLUDES := -I../../runtime/include\n"
            makefile.write_text(original)
            with (
                patch.object(
                    runner.core, "prepare_tool_source", return_value=directory
                ),
                self.assertRaisesRegex(RuntimeError, "stale runtime include marker"),
            ):
                runner.prepare_tool_source(
                    runner.core.TOOLS["kernelretsnoop"],
                    bpftime_root=root / "bpftime",
                    build_root=root / "build",
                    target_symbol="target",
                )
            self.assertEqual(makefile.read_text(), original)

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
            exit_kwargs = dict(
                expected_exit_events=(runner.CORRECTNESS_EXIT_EVENTS
                                      if tool == "kernelretsnoop" else None),
                expected_exit_launches=(runner.CORRECTNESS_EXIT_LAUNCHES
                                        if tool == "kernelretsnoop" else None),
                expected_exit_coordinates=(runner.CORRECTNESS_EXIT_COORDINATES
                                           if tool == "kernelretsnoop" else None),
                exact_exit_oracle=tool == "kernelretsnoop",
            )
            self.assertTrue(runner.gpubpf_probe_valid(
                tool, tool_probe, expected_thread_count=expected_threads,
                expected_ring_entries=(256 if tool == "kernelretsnoop" else None),
                **exit_kwargs))
            self.assertFalse(runner.gpubpf_probe_valid(
                tool, {**tool_probe, "sample_count": 0},
                expected_thread_count=expected_threads,
                expected_ring_entries=(256 if tool == "kernelretsnoop" else None),
                **exit_kwargs))
            nvbit_probe = (
                runner.parse_nvbit("launchlate", lossless_nvbit_launchlate_log())
                if tool == "launchlate" else
                runner.parse_nvbit("kernelretsnoop", lossless_nvbit_exit_log())
                if tool == "kernelretsnoop" else probe
            )
            self.assertTrue(runner.nvbit_probe_valid(tool, nvbit_probe, **exit_kwargs))
            self.assertFalse(runner.nvbit_probe_valid(
                tool, {**nvbit_probe, "sample_count": 0}, **exit_kwargs))
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
            "Host enqueued: 220", "Matched samples: 220", "Queue update errors: 0",
            "Uncertain samples: 0", "Accounting complete: 1",
            "Online accounting complete: 1", "Pairing complete: 1",
            "Clock drift bounded: 1",
            "Probes detached before final readback: 1",
        ):
            self.assertFalse(runner.gpubpf_probe_valid("launchlate", runner.parse_gpubpf("launchlate", text.replace(label, ""))))
        self.assertEqual(runner.parse_nvbit("launchlate", "NVBIT launchlate samples=2")["clock_errors"], -1)

    def test_nvbit_launchlate_accounting_and_calibration_gate_is_fail_closed(self):
        probe = runner.parse_nvbit("launchlate", lossless_nvbit_launchlate_log())
        self.assertTrue(runner.nvbit_probe_valid("launchlate", probe))
        self.assertEqual(probe["uncertain_samples"], 0)
        self.assertEqual(probe["start_clock_offset_lower_ns"], 868)
        self.assertEqual(probe["end_clock_offset_upper_ns"], 1052)
        self.assertEqual(probe["clock_offset_change_lower_ns"], -144)
        self.assertEqual(probe["clock_drift_rate_bound_ppb"], 184)
        corruptions = {
            "sample_count": 1,
            "selected_launches": 1,
            "histogram_sum": 1,
            "histogram": [-1, 3, 0, 0, 0, 0, 0, 0, 0, 0],
            "uncertain_samples": 1,
            "clock_errors": 1,
            "pair_capacity": 1,
            "stored_pairs": 1,
            "device_entries": 1,
            "pair_overflows": 1,
            "capture_errors": 1,
            "selected_counter_overflow": 1,
            "accounting_complete": 0,
            "process_selected_launches": 1,
            "result_blocks": 2,
            "calibration_blocks": 2,
            "start_clock_calibration_valid": 2,
            "end_clock_calibration_valid": 2,
            "start_clock_offset_lower_ns": -79,
            "end_clock_offset_lower_ns": -69,
            "start_clock_uncertainty_ns": 19,
            "end_clock_uncertainty_ns": 14,
            "start_clock_host_anchor_ns": 0,
            "end_clock_host_anchor_ns": 999999999,
            "clock_offset_change_lower_ns": -19,
            "clock_offset_change_upper_ns": 49,
            "clock_calibration_elapsed_ns": 999999999,
            "clock_drift_rate_bound_ppb": 49,
            "clock_drift_limit_ppb": 9999,
            "clock_drift_bounded": 0,
            "clock_calibration_method": "CLOCK_REALTIME_approximation",
        }
        for key, value in corruptions.items():
            with self.subTest(key=key):
                self.assertFalse(runner.nvbit_probe_valid(
                    "launchlate", {**probe, key: value}
                ))

        text = lossless_nvbit_launchlate_log()
        for label in (
            f"clock_calibration_method={runner.NVBIT_LAUNCH_CLOCK_METHOD}",
            "uncertain_samples=0",
            "pair_capacity=65536",
            "stored_pairs=220",
            "device_entries=220",
            "pair_overflows=0",
            "capture_errors=0",
            "selected_counter_overflow=0",
            "accounting_complete=1",
            "process_selected_launches=220",
            "start_clock_offset_lower_ns=868",
            "start_clock_offset_upper_ns=1032",
            "start_clock_uncertainty_ns=82",
            "start_clock_host_anchor_ns=1000000050",
            "start_clock_calibration_valid=1",
            "end_clock_offset_lower_ns=888",
            "end_clock_offset_upper_ns=1052",
            "end_clock_uncertainty_ns=82",
            "end_clock_host_anchor_ns=2000000050",
            "end_clock_calibration_valid=1",
            "clock_offset_change_lower_ns=-144",
            "clock_offset_change_upper_ns=184",
            "clock_calibration_elapsed_ns=1000000000",
            "clock_drift_rate_bound_ppb=184",
            "clock_drift_limit_ppb=10000",
            "clock_drift_bounded=1",
            "start_rm_samples_requested=32",
            "start_rm_bracket_width_ns=164",
            "end_rm_samples_requested=32",
            "end_rm_bracket_width_ns=164",
            "samples=220 clock_errors=0",
        ):
            with self.subTest(missing=label):
                parsed = runner.parse_nvbit("launchlate", text.replace(label, ""))
                self.assertFalse(runner.nvbit_probe_valid("launchlate", parsed))

        legal_negative_one = lossless_nvbit_launchlate_log(
            start_low=-1,
            start_high=163,
            start_uncertainty=82,
            start_gpu=1000000131,
            end_low=-1,
            end_high=163,
            end_uncertainty=82,
            end_gpu=2000000131,
            change_low=-164,
            change_high=164,
            drift_rate=164,
        )
        parsed = runner.parse_nvbit("launchlate", legal_negative_one)
        self.assertTrue(runner.nvbit_probe_valid("launchlate", parsed))
        for label in (
            "start_clock_offset_lower_ns=-1",
            "end_clock_offset_lower_ns=-1",
            "clock_offset_change_lower_ns=-164",
        ):
            with self.subTest(missing_legal_negative_one=label):
                parsed = runner.parse_nvbit(
                    "launchlate", legal_negative_one.replace(label, "")
                )
                self.assertFalse(runner.nvbit_probe_valid("launchlate", parsed))

        at_limit = runner.parse_nvbit(
            "launchlate",
            lossless_nvbit_launchlate_log(
                selected=220, samples=198, uncertain=22
            ),
        )
        above_limit = runner.parse_nvbit(
            "launchlate",
            lossless_nvbit_launchlate_log(
                selected=220, samples=197, uncertain=23
            ),
        )
        self.assertTrue(runner.nvbit_probe_valid("launchlate", at_limit))
        self.assertFalse(runner.nvbit_probe_valid("launchlate", above_limit))
        excessive_drift = runner.parse_nvbit(
            "launchlate",
            lossless_nvbit_launchlate_log(
                end_low=20000, end_high=20030, end_uncertainty=15,
                change_low=20080, change_high=20150, drift_rate=20150,
                drift_bounded=0,
            ),
        )
        self.assertFalse(runner.nvbit_probe_valid(
            "launchlate", excessive_drift
        ))

        short_but_low_drift = runner.parse_nvbit(
            "launchlate",
            lossless_nvbit_launchlate_log(
                end_anchor=1_500_000_000,
                elapsed=500_000_000,
                drift_rate=100,
            ),
        )
        self.assertEqual(short_but_low_drift["clock_drift_bounded"], 1)
        self.assertFalse(runner.nvbit_probe_valid(
            "launchlate", short_but_low_drift
        ))

        for index in range(10):
            with self.subTest(missing_bin=index):
                line = next(
                    line for line in text.splitlines()
                    if line.startswith(f"NVBIT launchlate bin_{index}=")
                )
                parsed = runner.parse_nvbit("launchlate", text.replace(line, ""))
                self.assertFalse(runner.nvbit_probe_valid("launchlate", parsed))

        duplicated = runner.parse_nvbit("launchlate", text + "\n" + text)
        self.assertEqual(duplicated["result_blocks"], 2)
        self.assertEqual(duplicated["calibration_blocks"], 2)
        self.assertFalse(runner.nvbit_probe_valid("launchlate", duplicated))

        compensated_missing_bin = text.replace(
            "NVBIT launchlate bin_0=0\n", ""
        ).replace("NVBIT launchlate bin_1=220", "NVBIT launchlate bin_1=221")
        parsed = runner.parse_nvbit("launchlate", compensated_missing_bin)
        self.assertEqual(parsed["histogram_sum"], 220)
        self.assertFalse(runner.nvbit_probe_valid("launchlate", parsed))
        for valid, malformed in (
            ("uncertain_samples=0", "uncertain_samples=0junk"),
            ("selected_launches=220", "selected_launches=220junk"),
        ):
            with self.subTest(malformed=malformed):
                parsed = runner.parse_nvbit(
                    "launchlate", text.replace(valid, malformed)
                )
                self.assertFalse(runner.nvbit_probe_valid("launchlate", parsed))

    def test_launchlate_accounting_and_calibration_gate_is_fail_closed(self):
        probe = runner.parse_gpubpf("launchlate", lossless_launchlate_log())
        self.assertTrue(runner.gpubpf_probe_valid("launchlate", probe))
        self.assertEqual(probe["host_enqueued"], 220)
        self.assertEqual(probe["matched_samples"], 220)
        self.assertEqual(probe["clock_offset_change_lower_ns"], -144)
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
            "online_accounting_complete": 0,
            "accounting_complete": 0,
            "pairing_complete": 0,
            "probes_detached_before_readback": 0,
            "start_clock_uncertainty_ns": 19,
            "end_clock_uncertainty_ns": 14,
            "start_clock_host_anchor_ns": 0,
            "end_clock_host_anchor_ns": 999999999,
            "clock_offset_change_lower_ns": -19,
            "clock_offset_change_upper_ns": 49,
            "clock_calibration_elapsed_ns": 999999999,
            "clock_drift_rate_bound_ppb": 49,
            "clock_drift_limit_ppb": 9999,
            "clock_drift_bounded": 0,
            "clock_calibration_method": "CLOCK_REALTIME approximation",
        }
        for key, value in corruptions.items():
            with self.subTest(key=key):
                self.assertFalse(runner.gpubpf_probe_valid(
                    "launchlate", {**probe, key: value}
                ))

        at_limit = runner.parse_gpubpf(
            "launchlate",
            lossless_launchlate_log(
                samples=20, histogram=18, host_launches=20,
                host_enqueued=20, device_entries=20, matched=20,
                classified=18, uncertain=2,
            ),
        )
        above_limit = runner.parse_gpubpf(
            "launchlate",
            lossless_launchlate_log(
                samples=20, histogram=17, host_launches=20,
                host_enqueued=20, device_entries=20, matched=20,
                classified=17, uncertain=3,
            ),
        )
        self.assertTrue(runner.gpubpf_probe_valid("launchlate", at_limit))
        self.assertFalse(runner.gpubpf_probe_valid("launchlate", above_limit))

    def test_lossless_exit_parser_and_correctness_oracle_are_fail_closed(self):
        probe = runner.parse_gpubpf("kernelretsnoop", lossless_exit_log())
        self.assertTrue(runner.gpubpf_probe_valid(
            "kernelretsnoop", probe,
            expected_thread_count=runner.EXPECTED_GPU_THREAD_SLOTS,
            expected_ring_entries=runner.CORRECTNESS_RING_ENTRIES_PER_THREAD,
            expected_exit_events=runner.CORRECTNESS_EXIT_EVENTS,
            expected_exit_launches=runner.CORRECTNESS_EXIT_LAUNCHES,
            expected_exit_coordinates=runner.CORRECTNESS_EXIT_COORDINATES,
            exact_exit_oracle=True,
        ))
        for key in (
            "sample_count", "nonzero_timestamps", "requested_thread_slots",
            "allocated_thread_slots", "requested_entries_per_thread",
            "entries_per_thread", "record_bytes",
            "committed_events", "runtime_collected_events", "oob_drops",
            "full_drops", "bad_size_drops", "other_drops", "dirty_slots",
            "pending_events", "final_drain_events", "second_drain_events",
            "cartesian_launches", "cartesian_coordinates", "cartesian_complete",
            "extent_x", "extent_y", "extent_z",
            "multiplicity_220", "multiplicity_44", "multiplicity_22",
            "other_multiplicity", "segment_mismatches",
            "invalid_launch_coordinates", "unique_coordinates",
            "oracle_enabled", "oracle_total_events", "oracle_passed",
            "collector_gate_passed",
        ):
            with self.subTest(key=key):
                if key in ("cartesian_complete", "oracle_enabled", "oracle_passed",
                             "collector_gate_passed"):
                    bad_value = 0
                else:
                    bad_value = probe[key] + 1
                broken = {**probe, key: bad_value}
                self.assertFalse(runner.gpubpf_probe_valid(
                    "kernelretsnoop", broken,
                    expected_thread_count=runner.EXPECTED_GPU_THREAD_SLOTS,
                    expected_ring_entries=runner.CORRECTNESS_RING_ENTRIES_PER_THREAD,
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
            expected_ring_entries=runner.CORRECTNESS_RING_ENTRIES_PER_THREAD,
            expected_exit_events=runner.CORRECTNESS_EXIT_EVENTS,
            expected_exit_launches=runner.CORRECTNESS_EXIT_LAUNCHES,
            expected_exit_coordinates=runner.CORRECTNESS_EXIT_COORDINATES,
            exact_exit_oracle=True,
        ))

        nvbit = runner.parse_nvbit("kernelretsnoop", lossless_nvbit_exit_log())
        self.assertTrue(runner.nvbit_probe_valid(
            "kernelretsnoop", nvbit,
            expected_exit_events=runner.CORRECTNESS_EXIT_EVENTS,
            expected_exit_launches=runner.CORRECTNESS_EXIT_LAUNCHES,
            expected_exit_coordinates=runner.CORRECTNESS_EXIT_COORDINATES,
            exact_exit_oracle=True,
        ))
        self.assertFalse(runner.nvbit_probe_valid(
            "kernelretsnoop", {**nvbit, "selected_launches": 175},
            expected_exit_events=runner.CORRECTNESS_EXIT_EVENTS,
            expected_exit_launches=runner.CORRECTNESS_EXIT_LAUNCHES,
            expected_exit_coordinates=runner.CORRECTNESS_EXIT_COORDINATES,
            exact_exit_oracle=True,
        ))

    def test_kernelretsnoop_invalid_launch_coordinates_fail_closed(self):
        probe = runner.parse_gpubpf("kernelretsnoop", lossless_exit_log())
        self.assertEqual(probe["invalid_launch_coordinates"], 0)
        self.assertFalse(runner.gpubpf_probe_valid(
            "kernelretsnoop", {**probe, "invalid_launch_coordinates": 1},
            expected_thread_count=runner.EXPECTED_GPU_THREAD_SLOTS,
            expected_ring_entries=runner.CORRECTNESS_RING_ENTRIES_PER_THREAD,
            expected_exit_events=runner.CORRECTNESS_EXIT_EVENTS,
            expected_exit_launches=runner.CORRECTNESS_EXIT_LAUNCHES,
            expected_exit_coordinates=runner.CORRECTNESS_EXIT_COORDINATES,
            exact_exit_oracle=True,
        ))

    def test_nvbit_kernelretsnoop_compact_coordinate_gate_is_fail_closed(self):
        probe = runner.parse_nvbit("kernelretsnoop", lossless_nvbit_exit_log())
        kwargs = {
            "expected_exit_events": runner.CORRECTNESS_EXIT_EVENTS,
            "expected_exit_launches": runner.CORRECTNESS_EXIT_LAUNCHES,
            "expected_exit_coordinates": runner.CORRECTNESS_EXIT_COORDINATES,
            "exact_exit_oracle": True,
        }
        self.assertTrue(runner.nvbit_probe_valid("kernelretsnoop", probe, **kwargs))
        for field in (
            "sample_count", "nonzero_timestamps", "selected_launches",
            "record_bytes", "bad_size_bytes", "cartesian_launches",
            "cartesian_coordinates", "cartesian_complete", "extent_x",
            "extent_y", "extent_z", "multiplicity_220", "multiplicity_44",
            "multiplicity_22", "other_multiplicity", "segment_mismatches",
            "invalid_launch_coordinates", "unique_coordinates",
            "collector_gate_passed", "validation_blocks",
            "process_selected_launches",
        ):
            with self.subTest(field=field):
                broken = {**probe, field: probe[field] + 1}
                self.assertFalse(runner.nvbit_probe_valid(
                    "kernelretsnoop", broken, **kwargs
                ))

    def test_timed_exit_gate_requires_frozen_pp32_geometry(self):
        layout = runner.kernelretsnoop_layout(32, correctness=False)
        text = lossless_exit_log(
            requested=layout["thread_slots"], allocated=layout["thread_slots"],
            requested_entries=layout["entries_per_thread"], entries=layout["entries_per_thread"],
            committed=layout["events"], collected=layout["events"],
            runtime_collected=layout["events"], nonzero=layout["events"],
            final_drain=8, launches=layout["launches"], coordinates=layout["coordinates"],
            extent_x=layout["extent_x"], extent_y=layout["extent_y"],
            extent_z=layout["extent_z"],
            multiplicity_220=0, multiplicity_44=layout["coordinates"], multiplicity_22=0,
            other_multiplicity=0, segment_mismatches=0,
            unique_coordinates=layout["coordinates"], oracle_enabled=0,
            oracle_total_events=layout["events"], oracle_passed=0,
        )
        probe = runner.parse_gpubpf("kernelretsnoop", text)
        self.assertTrue(runner.gpubpf_probe_valid(
            "kernelretsnoop", probe,
            expected_thread_count=layout["thread_slots"],
            expected_ring_entries=layout["entries_per_thread"],
            expected_exit_events=layout["events"],
            expected_exit_launches=layout["launches"],
            expected_exit_coordinates=layout["coordinates"],
            exact_exit_oracle=False,
        ))
        for field in ("requested_thread_slots", "requested_entries_per_thread",
                      "entries_per_thread", "sample_count", "cartesian_launches",
                      "cartesian_coordinates", "extent_x", "extent_y", "extent_z"):
            self.assertFalse(runner.gpubpf_probe_valid(
                "kernelretsnoop", {**probe, field: probe[field] - 1},
                expected_thread_count=layout["thread_slots"],
                expected_ring_entries=layout["entries_per_thread"],
                expected_exit_events=layout["events"],
                expected_exit_launches=layout["launches"],
                expected_exit_coordinates=layout["coordinates"],
                exact_exit_oracle=False,
            ))

    def test_phase_layouts_fit_budget_without_reusing_pp32_width(self):
        correctness = runner.kernelretsnoop_layout(32, correctness=True)
        preflight = runner.kernelretsnoop_layout(32, correctness=False)
        full = runner.kernelretsnoop_layout(512, correctness=False)
        self.assertEqual((correctness["thread_slots"], correctness["entries_per_thread"]),
                         (22528, 256))
        self.assertEqual((preflight["thread_slots"], preflight["events"]),
                         (32768, 1441792))
        self.assertEqual((full["thread_slots"], full["events"]),
                         (524288, 23068672))
        self.assertEqual(full["entries_per_thread"], 44)
        self.assertEqual(full["shared_bytes"], 935329824)
        self.assertLess(full["shared_bytes"], 1000 * 1024 * 1024)

    def test_timed_exit_pairs_require_equal_events_and_launches(self):
        for field in (None, "events", "launches"):
            with self.subTest(field=field):
                layout = runner.kernelretsnoop_layout(32, correctness=False)
                gp_probe = runner.parse_gpubpf("kernelretsnoop", lossless_exit_log(
                    requested=layout["thread_slots"], allocated=layout["thread_slots"],
                    requested_entries=layout["entries_per_thread"],
                    entries=layout["entries_per_thread"], committed=layout["events"],
                    collected=layout["events"], runtime_collected=layout["events"],
                    nonzero=layout["events"], launches=layout["launches"],
                    coordinates=layout["coordinates"], extent_x=layout["extent_x"],
                    extent_y=layout["extent_y"], extent_z=layout["extent_z"],
                    multiplicity_220=0, multiplicity_44=layout["coordinates"],
                    multiplicity_22=0, segment_mismatches=0,
                    unique_coordinates=layout["coordinates"], oracle_enabled=0,
                    oracle_total_events=layout["events"], oracle_passed=0))
                nv_probe = runner.parse_nvbit("kernelretsnoop", lossless_nvbit_exit_log(
                    selected=layout["launches"], events=layout["events"],
                    nonzero=layout["events"], launches=layout["launches"],
                    coordinates=layout["coordinates"], extent_x=layout["extent_x"],
                    multiplicity_220=0, multiplicity_44=layout["coordinates"],
                    multiplicity_22=0, unique_coordinates=layout["coordinates"]))
                gpubpf = {"block": 1, "valid": True, "probe": gp_probe}
                nvbit = {"block": 1, "valid": True, "probe": nv_probe}
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

    def test_tool_selection_is_predeclared_canonical_and_defaults_to_all_three(self):
        required = ["--bpftime-root", "/source", "--bpftime-build-dir", "/build",
                    "--gpu-thread-count", "22528", "--dry-run"]
        default = runner.parse_args(required)
        self.assertEqual(tuple(default.tools), runner.TASKS)
        self.assertEqual(runner.selected_configs(default), tuple(runner.CONFIGS))

        subset = runner.parse_args(required + ["--tools", "threadhist", "kernelretsnoop"])
        self.assertEqual(tuple(subset.tools), ("kernelretsnoop", "threadhist"))
        self.assertEqual(runner.selected_configs(subset), (
            "baseline", "gpubpf_kernelretsnoop", "nvbit_kernelretsnoop",
            "gpubpf_threadhist", "nvbit_threadhist",
        ))
        with self.assertRaisesRegex(ValueError, "duplicate tool"):
            runner.parse_args(required + ["--tools", "threadhist", "threadhist"])

    def test_verifier_treatment_is_explicit_auditable_and_defaults_to_legacy(self):
        required = ["--bpftime-root", "/source", "--bpftime-build-dir", "/build",
                    "--gpu-thread-count", "22528", "--dry-run"]
        default = runner.parse_args(required)
        self.assertEqual(default.verifier_level, "DEFAULT")
        self.assertEqual(runner.verifier_environment(default), {})
        strict = runner.parse_args(required + ["--verifier-level", "STRICT"])
        self.assertEqual(
            runner.verifier_environment(strict),
            {"BPFTIME_VERIFIER_LEVEL": "STRICT", "SPDLOG_LEVEL": "info"},
        )
        no_verify = runner.parse_args(required + ["--verifier-level", "NO_VERIFY"])
        self.assertEqual(runner.dry_run_plan(no_verify)["verifier_level"], "NO_VERIFY")

    def test_explicit_verifier_treatment_requires_one_enabled_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            args = SimpleNamespace(bpftime_build_dir=build, verifier_level="STRICT")
            build.joinpath("CMakeCache.txt").write_text("\n".join((
                "ENABLE_EBPF_VERIFIER:BOOL=ON",
                "BPFTIME_ENABLE_CUDA_ATTACH:BOOL=ON",
                "BPFTIME_LLVM_JIT:BOOL=ON",
            )))
            config = runner.require_explicit_verifier_build(args)
            self.assertEqual(set(config.values()), {"ON"})
            build.joinpath("CMakeCache.txt").write_text(
                "ENABLE_EBPF_VERIFIER:BOOL=OFF\n"
                "BPFTIME_ENABLE_CUDA_ATTACH:BOOL=ON\n"
                "BPFTIME_LLVM_JIT:BOOL=ON\n"
            )
            with self.assertRaisesRegex(RuntimeError, "verifier-enabled"):
                runner.require_explicit_verifier_build(args)

    def test_actual_table1_verifier_records_are_required_for_explicit_modes(self):
        args = SimpleNamespace(
            verifier_level="STRICT", target_symbol="_Z6targetv", pp=32
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            pid = 4321
            run_dir.joinpath("llama_cli.execution.json").write_text(json.dumps({
                "identity": {"pid": pid},
            }))
            prefix = f"[2026-09-04 00:00:00][info][{pid}] "
            accepted = (
                prefix
                + "GPU eBPF verification accepted: mode=STRICT program=cuda__retprobe "
                "attach=kretprobe/_Z6targetv instructions=13\n"
                + prefix
                + "GPU eBPF verified map: program=cuda__retprobe fd=16 type=1502 "
                "key_size=4 value_size=8 max_entries=1\n"
            )
            run_dir.joinpath("agent.log").write_text("Verifier mode: STRICT\n")
            run_dir.joinpath("llama_cli.log").write_text(accepted)
            evidence = runner.verifier_evidence(
                args, run_dir, "threadhist", correctness=True
            )
            self.assertTrue(evidence["passed"])
            self.assertEqual(evidence["instruction_counts"], [13])
            self.assertEqual(evidence["accepted_records"], 1)
            self.assertEqual(evidence["verified_map_records"], 1)
            self.assertEqual(evidence["target_pid"], pid)
            self.assertEqual(evidence["logs_scanned"], ["llama_cli.log"])
            self.assertEqual(evidence["matched_log_sources"], ["llama_cli.log"])

            args.verifier_level = "NO_VERIFY"
            run_dir.joinpath("llama_bench.execution.json").write_text(json.dumps({
                "identity": {"pid": pid},
            }))
            run_dir.joinpath("llama_bench.log").write_text(
                prefix + "Skipping GPU eBPF verification for cuda__retprobe\n"
            )
            evidence = runner.verifier_evidence(
                args, run_dir, "threadhist", correctness=False
            )
            self.assertTrue(evidence["passed"])
            self.assertEqual(evidence["matched_log_sources"], ["llama_bench.log"])

    def test_verifier_evidence_fails_closed_on_missing_wrong_or_mixed_records(self):
        args = SimpleNamespace(
            verifier_level="STRICT", target_symbol="_Z6targetv", pp=32
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            pid = 4321
            prefix = f"[2026-09-04 00:00:00][info][{pid}] "
            accepted_line = (
                prefix
                + "GPU eBPF verification accepted: mode=STRICT "
                "program=cuda__retprobe attach=kretprobe/_Z6targetv "
                "instructions=60\n"
            )
            map_line = (
                prefix
                + "GPU eBPF verified map: program=cuda__retprobe fd=16 "
                "type=1527 key_size=4 value_size=32 max_entries=256\n"
            )
            missing = runner.verifier_evidence(
                args, run_dir, "kernelretsnoop", correctness=True
            )
            self.assertFalse(missing["passed"])
            self.assertEqual(missing["logs_scanned"], [])
            self.assertEqual(missing["logs_missing"], ["llama_cli.log"])
            self.assertIsNotNone(missing["execution_error"])

            run_dir.joinpath("llama_cli.execution.json").write_text(json.dumps({
                "identity": {"pid": pid},
            }))
            cases = {
                "wrong attach": accepted_line.replace(
                    "_Z6targetv", "_Z11not_targetv"
                ) + map_line,
                "wrong mode": accepted_line.replace(
                    "mode=STRICT", "mode=WARNING"
                ) + map_line,
                "wrong map": accepted_line + map_line.replace(
                    "max_entries=256", "max_entries=44"
                ),
                "duplicate admission": accepted_line + accepted_line + map_line,
                "mixed skip": accepted_line + map_line + prefix
                    + "Skipping GPU eBPF verification for cuda__retprobe\n",
                "rejected": accepted_line + map_line + prefix
                    + "GPU eBPF verification failed for cuda__retprobe: rejected\n",
            }
            for name, text in cases.items():
                with self.subTest(name=name):
                    run_dir.joinpath("llama_cli.log").write_text(text)
                    evidence = runner.verifier_evidence(
                        args, run_dir, "kernelretsnoop", correctness=True
                    )
                    self.assertFalse(evidence["passed"])

            foreign = accepted_line.replace(f"][{pid}] ", "][9999] ")
            foreign += map_line.replace(f"][{pid}] ", "][9999] ")
            run_dir.joinpath("llama_cli.log").write_text(foreign)
            evidence = runner.verifier_evidence(
                args, run_dir, "kernelretsnoop", correctness=True
            )
            self.assertFalse(evidence["passed"])
            self.assertEqual(evidence["foreign_pid_records"], 2)

            # Admission in the loader log cannot be stitched to a target-log map.
            run_dir.joinpath("agent.log").write_text(accepted_line)
            run_dir.joinpath("llama_cli.log").write_text(map_line)
            evidence = runner.verifier_evidence(
                args, run_dir, "kernelretsnoop", correctness=True
            )
            self.assertFalse(evidence["passed"])
            self.assertEqual(evidence["accepted_records"], 0)

    def test_two_tool_dry_run_has_fixed_preflight_and_full_matrices_and_exact_gates(self):
        base = ["--bpftime-root", "/does-not-need-to-exist",
                "--bpftime-build-dir", "/also-missing", "--gpu-thread-count", "22528",
                "--dry-run", "--tools", "kernelretsnoop", "threadhist"]
        for phase, runs, pp in (("preflight", 1, 32), ("full", 10, 512)):
            with self.subTest(phase=phase):
                phase_args = base + ["--phase", phase]
                if phase == "full":
                    phase_args += ["--preflight-dir", "/campaigns/passed-preflight",
                                   "--output-dir", "/campaigns/full"]
                args = runner.parse_args(phase_args)
                runner.validate_plan(args)
                plan = runner.dry_run_plan(args)
                self.assertEqual(plan["tools"], ["kernelretsnoop", "threadhist"])
                self.assertEqual(len(plan["configs"]), 5)
                self.assertEqual((plan["runs"], plan["pp"]), (runs, pp))
                self.assertEqual(plan["timing_cell_count"], 5 * runs)
                self.assertEqual(plan["timing_schedule"], runner.fixed_schedule(args))
                for order in plan["timing_schedule"].values():
                    self.assertEqual(len(order), 5)
                    self.assertEqual(set(order), set(plan["configs"]))
                gates = plan["engagement_gates"]
                self.assertNotIn("launchlate_all_cells", gates)
                self.assertEqual(
                    gates["kernelretsnoop_correctness"]["gpubpf"]["events"],
                    runner.CORRECTNESS_EXIT_EVENTS,
                )
                layout = runner.kernelretsnoop_layout(pp, correctness=False)
                timing_gate = gates["kernelretsnoop_timing"]
                self.assertEqual(timing_gate["gpubpf_requested_and_allocated_thread_slots"],
                                 layout["thread_slots"])
                self.assertEqual(timing_gate["gpubpf_exact_ring_entries_per_thread"], 44)
                self.assertEqual(timing_gate["gpubpf_exact_events"], layout["events"])
                self.assertEqual(timing_gate["nvbit_exact_events"], layout["events"])
                self.assertEqual(timing_gate["gpubpf_exact_selected_launches"], 44)
                self.assertEqual(
                    gates["threadhist_all_cells"]["gpubpf_readback_bytes"],
                    args.threadhist_gpu_thread_count * 8,
                )
                self.assertIn("not relabeled", plan["scope_policy"])
                self.assertEqual(plan["preflight_gate"]["required"], phase == "full")

    def test_launchlate_dry_run_is_exactly_three_arms_and_ten_paired_blocks(self):
        base = [
            "--bpftime-root", "/source", "--bpftime-build-dir", "/build",
            "--gpu-thread-count", "22528", "--dry-run", "--tools", "launchlate",
        ]
        preflight = runner.parse_args(base + ["--phase", "preflight"])
        runner.validate_plan(preflight)
        preflight_plan = runner.dry_run_plan(preflight)
        self.assertEqual(preflight_plan["configs"], [
            "baseline", "gpubpf_launchlate", "nvbit_launchlate",
        ])
        self.assertEqual(preflight_plan["timing_cell_count"], 3)
        self.assertEqual(preflight_plan["engagement_gates"]["launchlate_correctness"], {
            "gpubpf_exact_launches": 220, "nvbit_exact_launches": 220,
            "minimum_classified": 198, "maximum_uncertain": 22,
        })

        full = runner.parse_args(base + [
            "--phase", "full", "--preflight-dir", "/campaigns/launch-preflight",
            "--output-dir", "/campaigns/launch-full",
        ])
        runner.validate_plan(full)
        full_plan = runner.dry_run_plan(full)
        self.assertEqual(full_plan["timing_cell_count"], 30)
        self.assertEqual(len(full_plan["timing_schedule"]), 10)
        for order in full_plan["timing_schedule"].values():
            self.assertEqual(len(order), 3)
            self.assertEqual(set(order), set(full_plan["configs"]))
        self.assertTrue(full_plan["preflight_gate"]["required"])

    def test_subset_full_requires_independently_complete_matching_preflight(self):
        required = ["--phase", "full", "--bpftime-root", "/source",
                    "--bpftime-build-dir", "/build", "--gpu-thread-count", "22528",
                    "--tools", "kernelretsnoop", "threadhist", "--output-dir", "/full"]
        missing = runner.parse_args(required)
        with self.assertRaisesRegex(ValueError, "subset full requires --preflight-dir"):
            runner.validate_plan(missing)

        args = runner.parse_args(required + ["--preflight-dir", "/preflight"])
        passed = {
            "phase": "preflight",
            "tools": ["kernelretsnoop", "threadhist"],
            "configs": list(runner.selected_configs(args)),
            "complete": True,
        }
        with patch("analyze_revision_rq4.analyze", return_value=passed) as analyze:
            runner.validate_subset_preflight(args)
        analyze.assert_called_once_with(Path("/preflight"))
        for defect in ("incomplete", "wrong tools", "wrong phase"):
            with self.subTest(defect=defect):
                result = dict(passed)
                if defect == "incomplete":
                    result["complete"] = False
                elif defect == "wrong tools":
                    result["tools"] = ["threadhist"]
                else:
                    result["phase"] = "full"
                with (patch("analyze_revision_rq4.analyze", return_value=result),
                      self.assertRaisesRegex(RuntimeError, "independently complete preflight")):
                    runner.validate_subset_preflight(args)

        default = runner.parse_args([
            "--phase", "full", "--bpftime-root", "/source",
            "--bpftime-build-dir", "/build", "--gpu-thread-count", "22528",
        ])
        runner.validate_plan(default)
        with patch("analyze_revision_rq4.analyze") as analyze:
            runner.validate_subset_preflight(default)
        analyze.assert_not_called()

        for preflight, full in (("/campaign", "/campaign/full"),
                                ("/campaign/preflight", "/campaign"),
                                ("/campaign", "/campaign")):
            with self.subTest(preflight=preflight, full=full):
                nested = runner.parse_args([
                    "--phase", "full", "--bpftime-root", "/source",
                    "--bpftime-build-dir", "/build", "--gpu-thread-count", "22528",
                    "--tools", "kernelretsnoop", "threadhist",
                    "--preflight-dir", preflight, "--output-dir", full,
                ])
                with self.assertRaisesRegex(ValueError, "mutually non-nested"):
                    runner.validate_plan(nested)

    def test_summary_contains_only_predeclared_tools_and_configs(self):
        tools = ["kernelretsnoop", "threadhist"]
        configs = [config for config in runner.CONFIGS
                   if config == "baseline" or config.split("_", 1)[1] in tools]
        state = {
            "params": {"tools": tools, "runs": 1},
            "configs": {config: {"runs": []} for config in configs},
        }
        summary = runner.summarize(state)
        self.assertEqual([row["config"] for row in summary["configs"]], configs)
        self.assertEqual([row["task"] for row in summary["comparisons"]], tools)
        self.assertNotIn("launchlate", json.dumps(summary))

    def test_two_tool_campaign_builds_and_runs_only_selected_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "campaign"
            output.mkdir()
            args = SimpleNamespace(
                output_dir=output, resume=False, phase="preflight", runs=1, pp=32,
                tools=["kernelretsnoop", "threadhist"], target_symbol="target",
                bpftime_root=Path("/source"),
            )
            configs = runner.selected_configs(args)
            state = {
                "phase": "preflight",
                "params": {"tools": list(args.tools), "runs": 1, "target_symbol": "target"},
                "provenance": {"driver": runner.EXPECTED_DRIVER},
                "schedule": runner.fixed_schedule(args),
                "correctness": {config: {"attempts": []} for config in configs},
                "configs": {config: {"runs": []} for config in configs},
                "artifacts": {},
            }
            snapshot = {
                "gpu": f"RTX 5090, {runner.EXPECTED_DRIVER}, 32607, 0, 0, 0",
                "compute_apps": "",
            }

            def correctness(config, attempt, *unused):
                result = {
                    "valid": True, "returncode": 0,
                    "normalized_stdout": runner.EXPECTED_NORMALIZED_STDOUT,
                }
                return result

            def timing(config, run_id, *unused):
                result = {"valid": True, "returncode": 0,
                          "metrics": {"pp_tokens": 32, "pp_tok_s": 100.0}}
                if config == "gpubpf_kernelretsnoop":
                    result["probe"] = {"sample_count": 40, "cartesian_launches": 5}
                elif config == "nvbit_kernelretsnoop":
                    result["probe"] = {"sample_count": 40,
                                       "cartesian_launches": 5,
                                       "selected_launches": 5}
                return result

            with (patch.object(runner.core, "nvidia_smi_snapshot", return_value=snapshot),
                  patch.object(runner.shutil, "copytree"),
                  patch.object(runner, "validate_nvbit_kernelretsnoop_source_schema") as exit_schema,
                  patch.object(runner, "validate_nvbit_launchlate_source_schema") as launch_schema,
                  patch.object(runner, "validate_kernelretsnoop_source_schema"),
                  patch.object(runner, "build_nvbit", return_value=output / "nvbit.so"),
                  patch.object(runner, "prepare_tool_source",
                               side_effect=lambda spec, **kwargs: output / spec.name) as prepare,
                  patch.object(runner.core, "build_tool"),
                  patch.object(runner, "new_state", return_value=state),
                  patch.object(runner, "run_correctness_cell", side_effect=correctness) as correct,
                  patch.object(runner, "run_cell", side_effect=timing) as timed):
                self.assertEqual(runner.run_campaign(args), 0)
            launch_schema.assert_not_called()
            exit_schema.assert_called_once()
            self.assertEqual([call.args[0].name for call in prepare.call_args_list],
                             ["kernelretsnoop", "threadhist"])
            correctness_order = [call.args[0] for call in correct.call_args_list]
            self.assertEqual(correctness_order[0], "baseline")
            self.assertEqual(set(correctness_order), set(configs))
            self.assertEqual({call.args[0] for call in timed.call_args_list}, set(configs))
            self.assertNotIn("launchlate", json.dumps(state))

    def test_active_runner_has_no_content_fingerprint_logic(self):
        source = Path(runner.__file__).read_text().lower()
        for forbidden in ("hashlib", "sha256", "checksum", "digest"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
