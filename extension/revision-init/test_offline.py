#!/usr/bin/env python3
"""CPU fixtures for scheduler-init joining; no record here is device evidence."""
from __future__ import annotations

import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("revision_init_run_live", HERE / "run_live.py")
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

TARGET_PID = 4242
TARGET_TID = 4243
DEFAULT_TIMESLICE = 2048


def fixture(row: runner.Row):
    common = {
        "event": "scheduler_init_diagnostic", "pid": TARGET_PID, "tid": TARGET_TID,
        "abi_version": 1, "abi_size": 168, "h_client": 11, "h_resource": 12,
        "gpu_instance": 0, "subdevice_instance": 0, "group_id": 13,
        "runlist_id": 14, "engine_type": 15, "constructor_epoch": 16,
        "default_timeslice": DEFAULT_TIMESLICE, "minimum_timeslice": 0,
        "default_interleave": 1,
        "timeslice_attempted": int(bool(row.timeslice_returns)),
        "timeslice_conflict": row.timeslice_conflict,
        "timeslice_request_value": DEFAULT_TIMESLICE if row.timeslice_returns else 0,
        "interleave_attempted": int(bool(row.interleave_returns)),
        "interleave_conflict": row.interleave_conflict,
        "interleave_request_value": row.interleave_request if row.interleave_returns else 0,
        "timeslice_validation_result": row.timeslice_result,
        "interleave_validation_result": row.interleave_result,
        "effective_timeslice": DEFAULT_TIMESLICE,
        "effective_interleave": 0 if runner.FIELD_INTERLEAVE in row.native_fields else 1,
        "timeslice_native_status": runner.STATUS_NOT_OBSERVED,
        "timeslice_post_value": 0,
        "interleave_native_status": runner.STATUS_NOT_OBSERVED,
        "interleave_post_value": 0,
        "constructor_status": 0, "final_interleave": 0, "final_timeslice": 0,
        "final_snapshot_valid": 0,
    }
    diagnostics = []
    validated = {**common, "phase": runner.PHASE_VALIDATED,
                 "field": runner.FIELD_NONE, "timestamp_ns": 100}
    diagnostics.append(validated)
    gsp = []
    timestamp = 110
    state = copy.deepcopy(common)
    for field in row.native_fields:
        command = runner.GSP_TIMESLICE if field == runner.FIELD_TIMESLICE else runner.GSP_INTERLEAVE
        size = 8 if field == runner.FIELD_TIMESLICE else 4
        value = DEFAULT_TIMESLICE if field == runner.FIELD_TIMESLICE else 0
        gsp.append({
            "event": "scheduler_init_gsp_completion", "pid": TARGET_PID,
            "tid": TARGET_TID, "timestamp_ns": timestamp,
            "h_client": 11, "h_object": 12, "command": command,
            "input_size": size, "wire_size": size, "input_value": value,
            "input_valid": 1, "transport_status": 0, "gsp_status": 0,
            "gsp_status_valid": 1,
        })
        timestamp += 10
        if field == runner.FIELD_TIMESLICE:
            state.update(timeslice_native_status=0,
                         timeslice_post_value=DEFAULT_TIMESLICE)
        else:
            state.update(interleave_native_status=0, interleave_post_value=0)
        diagnostics.append({**state, "event": "scheduler_init_diagnostic",
                            "phase": runner.PHASE_NATIVE_RETURN, "field": field,
                            "timestamp_ns": timestamp})
        timestamp += 10
    diagnostics.append({
        **state, "event": "scheduler_init_diagnostic",
        "phase": runner.PHASE_CONSTRUCTOR_RETURN, "field": runner.FIELD_NONE,
        "timestamp_ns": timestamp, "constructor_status": 0,
        "final_interleave": 0 if runner.FIELD_INTERLEAVE in row.native_fields else 1,
        "final_timeslice": DEFAULT_TIMESLICE, "final_snapshot_valid": 1,
    })
    prepolicy_gsp = {
        "event": "scheduler_init_gsp_completion", "pid": TARGET_PID,
        "tid": TARGET_TID, "timestamp_ns": 80, "h_client": 11,
        "h_object": 12, "command": runner.GSP_INTERLEAVE,
        "input_size": 4, "wire_size": 4, "input_value": 1,
        "input_valid": 1, "transport_status": 0, "gsp_status": 0,
        "gsp_status_valid": 1,
    }
    all_gsp = [prepolicy_gsp, *gsp]
    timeline = sorted([*diagnostics, *all_gsp], key=lambda event: event["timestamp_ns"])
    observer = [
        {"event": "scheduler_init_observer_ready", "target_tgid": TARGET_PID},
        *timeline,
        {"event": "scheduler_init_observer_summary",
         "diagnostic": {"observed": len(diagnostics), "emitted": len(diagnostics),
                        "read_errors": 0, "ring_drops": 0, "received": len(diagnostics)},
         "gsp": {"observed": len(all_gsp), "emitted": len(all_gsp),
                 "read_errors": 0, "ring_drops": 0, "received": len(all_gsp)}},
    ]
    loader = []
    if row.fixture is not None:
        policy = {
            "event": "scheduler_init_policy_request", "pid": TARGET_PID,
            "tid": TARGET_TID, "timestamp_ns": 90, "tsg_id": 13,
            "runlist_id": 14, "engine_type": 15,
            "default_timeslice": DEFAULT_TIMESLICE, "default_interleave": 1,
            "fixture": row.fixture_id, "complete": 1,
            "timeslice_count": len(row.timeslice_returns),
            "timeslice_returns": list(row.timeslice_returns) +
                                 [0] * (3 - len(row.timeslice_returns)),
            "interleave_count": len(row.interleave_returns),
            "interleave_returns": list(row.interleave_returns) +
                                  [0] * (3 - len(row.interleave_returns)),
        }
        loader = [
            {"event": "scheduler_init_loader_ready", "target_tgid": TARGET_PID,
             "struct_ops_map_id": 21, "struct_ops_link_id": 22},
            policy,
            {"event": "scheduler_init_loader_summary", "target_tgid": TARGET_PID,
             "struct_ops_map_id": 21, "struct_ops_link_id": 22,
             "init_seen": 1, "init_recorded": 1, "init_record_error": 0,
             "request_records": 1},
        ]
    return [runner.EXPECTED_CORRECTNESS], observer, loader


class MatrixFixtureTests(unittest.TestCase):
    def test_exact_block_major_sixteen_cell_plan(self):
        plan = runner.matrix_plan()
        self.assertEqual(len(plan), 16)
        self.assertEqual([entry["block"] for entry in plan], [0] * 8 + [1] * 8)
        self.assertEqual([entry["row"] for entry in plan[:8]],
                         [row.name for row in runner.ROWS])
        self.assertEqual([entry["row"] for entry in plan[8:]],
                         [row.name for row in runner.ROWS])
        self.assertEqual(len({(entry["block"], entry["row"]) for entry in plan}), 16)

    def test_all_frozen_rows_join(self):
        for row in runner.ROWS:
            with self.subTest(row=row.name):
                target, observer, loader = fixture(row)
                result = runner.validate_cell(row, target, observer, loader, TARGET_PID)
                self.assertTrue(result["passed"])
                self.assertEqual(len(result["constructors"]), 1)
                self.assertEqual(
                    result["gsp_events_ignored_outside_constructor_intervals"], 1)

    def test_rejected_field_cannot_have_native_or_gsp_event(self):
        row = next(row for row in runner.ROWS if row.name == "bpf_invalid_interleave")
        target, observer, loader = fixture(row)
        validated = next(event for event in observer if event.get("phase") == runner.PHASE_VALIDATED)
        final_index = next(index for index, event in enumerate(observer)
                           if event.get("phase") == runner.PHASE_CONSTRUCTOR_RETURN)
        observer.insert(final_index, {**validated, "phase": runner.PHASE_NATIVE_RETURN,
                                     "field": runner.FIELD_INTERLEAVE,
                                     "timestamp_ns": 105,
                                     "interleave_native_status": 0})
        observer[-1]["diagnostic"]["observed"] += 1
        observer[-1]["diagnostic"]["emitted"] += 1
        observer[-1]["diagnostic"]["received"] += 1
        with self.assertRaisesRegex(runner.GateError, "phase framing"):
            runner.validate_cell(row, target, observer, loader, TARGET_PID)

        target, observer, loader = fixture(row)
        observer.insert(-1, {
            "event": "scheduler_init_gsp_completion", "pid": TARGET_PID,
            "tid": TARGET_TID, "timestamp_ns": 110, "h_client": 11,
            "h_object": 12, "command": runner.GSP_INTERLEAVE,
            "input_size": 4, "wire_size": 4, "input_value": 3,
            "input_valid": 1, "transport_status": 0, "gsp_status": 0,
            "gsp_status_valid": 1,
        })
        observer[-1]["gsp"]["observed"] += 1
        observer[-1]["gsp"]["emitted"] += 1
        observer[-1]["gsp"]["received"] += 1
        with self.assertRaisesRegex(runner.GateError, "GSP completion count"):
            runner.validate_cell(row, target, observer, loader, TARGET_PID)

    def test_missing_or_failed_constructor_is_rejected(self):
        row = next(row for row in runner.ROWS if row.name == "bpf_legal")
        target, observer, loader = fixture(row)
        final = next(event for event in observer
                     if event.get("phase") == runner.PHASE_CONSTRUCTOR_RETURN)
        final["constructor_status"] = 7
        with self.assertRaisesRegex(runner.GateError, "constructor return"):
            runner.validate_cell(row, target, observer, loader, TARGET_PID)
        _, observer, loader = fixture(row)
        observer = [event for event in observer
                    if event.get("phase") != runner.PHASE_CONSTRUCTOR_RETURN]
        observer[-1]["diagnostic"]["observed"] -= 1
        observer[-1]["diagnostic"]["emitted"] -= 1
        observer[-1]["diagnostic"]["received"] -= 1
        with self.assertRaises(runner.GateError):
            runner.validate_cell(row, target, observer, loader, TARGET_PID)

    def test_out_of_order_diagnostic_and_gsp_events_are_rejected(self):
        row = next(row for row in runner.ROWS if row.name == "bpf_legal")
        target, observer, loader = fixture(row)
        native = next(event for event in observer
                      if event.get("phase") == runner.PHASE_NATIVE_RETURN)
        native["timestamp_ns"] = 99
        with self.assertRaisesRegex(runner.GateError, "diagnostic events are out of order"):
            runner.validate_cell(row, target, observer, loader, TARGET_PID)

        target, observer, loader = fixture(row)
        policy_gsp = [event for event in observer
                      if event.get("event") == "scheduler_init_gsp_completion" and
                      event["timestamp_ns"] >= 100]
        first = observer.index(policy_gsp[0])
        second = observer.index(policy_gsp[1])
        observer[first], observer[second] = observer[second], observer[first]
        with self.assertRaisesRegex(runner.GateError, "GSP events are out of order"):
            runner.validate_cell(row, target, observer, loader, TARGET_PID)

    def test_observer_loss_and_policy_accounting_are_rejected(self):
        row = next(row for row in runner.ROWS if row.name == "bpf_no_request")
        target, observer, loader = fixture(row)
        observer[-1]["diagnostic"]["ring_drops"] = 1
        with self.assertRaisesRegex(runner.GateError, "read/drop"):
            runner.validate_cell(row, target, observer, loader, TARGET_PID)
        target, observer, loader = fixture(row)
        loader[-1]["init_seen"] = 2
        with self.assertRaisesRegex(runner.GateError, "accounting"):
            runner.validate_cell(row, target, observer, loader, TARGET_PID)

    def test_native_row_refuses_policy_records(self):
        row = runner.ROWS[0]
        target, observer, _ = fixture(row)
        with self.assertRaisesRegex(runner.GateError, "native row"):
            runner.validate_cell(row, target, observer,
                                 [{"event": "unexpected"}], TARGET_PID)

    def test_exact_loaded_btf_layout_and_hook_prototypes(self):
        def structure(number, name, size, fields):
            members = "".join(
                f"\t'{field}' type_id=1 bits_offset={offset}\n" for field, offset in fields
            )
            return f"[{number}] STRUCT '{name}' size={size} vlen={len(fields)}\n{members}"

        raw = (
            structure(10, "nv_gpu_sched_init_diagnostic_ctx", 168,
                      runner.BTF_DIAGNOSTIC_FIELDS) +
            structure(20, "nv_gpu_gsp_control_complete_ctx", 48,
                      runner.BTF_GSP_FIELDS) +
            "[11] CONST '(anon)' type_id=10\n[12] PTR '(anon)' type_id=11\n"
            "[13] FUNC_PROTO '(anon)' ret_type_id=0 vlen=1\n\t'ctx' type_id=12\n"
            "[14] FUNC 'nv_gpu_sched_init_diagnostic' type_id=13 linkage=global\n"
            "[21] CONST '(anon)' type_id=20\n[22] PTR '(anon)' type_id=21\n"
            "[23] FUNC_PROTO '(anon)' ret_type_id=0 vlen=1\n\t'ctx' type_id=22\n"
            "[24] FUNC 'nv_gpu_sched_gsp_control_complete' type_id=23 linkage=global\n"
        )
        runner.validate_loaded_btf(raw)
        with self.assertRaisesRegex(runner.GateError, "layout mismatch"):
            runner.validate_loaded_btf(raw.replace("bits_offset=1312", "bits_offset=1313"))
        with self.assertRaisesRegex(runner.GateError, "hook prototype"):
            runner.validate_loaded_btf(raw.replace("ret_type_id=0 vlen=1", "ret_type_id=1 vlen=1", 1))


class ProcessAndInputTests(unittest.TestCase):
    def tearDown(self):
        runner.INTERRUPTED_SIGNALS.clear()

    def test_interrupt_is_queued_until_a_safe_gate(self):
        runner.note_interrupt(15, None)
        with self.assertRaisesRegex(InterruptedError, "signal 15"):
            runner.raise_if_interrupted()

    def test_json_reader_rejects_partial_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events"
            path.write_text('{"event":"ok"}\n{"event":')
            self.assertEqual(runner.json_events(path, allow_partial=True),
                             [{"event": "ok"}])
            with self.assertRaisesRegex(runner.GateError, "incomplete JSON"):
                runner.json_events(path)

    def test_compute_monitor_is_bounded_and_rejects_foreign_pids(self):
        samples = [
            {"event": "sample", "query_started_mono_ns": 10,
             "query_finished_mono_ns": 20, "pids": []},
            {"event": "sample", "query_started_mono_ns": 30,
             "query_finished_mono_ns": 40, "pids": [TARGET_PID]},
            {"event": "sample", "query_started_mono_ns": 50,
             "query_finished_mono_ns": 60, "pids": []},
            {"event": "final", "errors": 0},
        ]
        window = {
            "pretarget_query_started_mono_ns": 10,
            "pretarget_query_finished_mono_ns": 20,
            "target_started_mono_ns": 25,
            "target_exit_mono_ns": 42,
            "owned_cleanup_mono_ns": 45,
            "postcleanup_query_started_mono_ns": 50,
            "postcleanup_query_finished_mono_ns": 60,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "compute.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in samples) + "\n")
            result = runner.validate_compute_monitor(path, TARGET_PID, window)
            self.assertEqual(result["samples"], 3)
            samples[1]["pids"].append(9999)
            path.write_text("\n".join(json.dumps(row) for row in samples) + "\n")
            with self.assertRaisesRegex(runner.GateError, "foreign compute"):
                runner.validate_compute_monitor(path, TARGET_PID, window)
            samples[1]["pids"] = [TARGET_PID]
            samples[1]["query_started_mono_ns"] = runner.COMPUTE_MAX_GAP_NS + 21
            samples[1]["query_finished_mono_ns"] = runner.COMPUTE_MAX_GAP_NS + 31
            samples[2]["query_started_mono_ns"] = runner.COMPUTE_MAX_GAP_NS + 41
            samples[2]["query_finished_mono_ns"] = runner.COMPUTE_MAX_GAP_NS + 51
            path.write_text("\n".join(json.dumps(row) for row in samples) + "\n")
            with self.assertRaisesRegex(runner.GateError, "sampling gap"):
                runner.validate_compute_monitor(path, TARGET_PID)

    def test_read_only_leases_never_create_missing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            absent = Path(temporary) / "absent"
            with patch.object(runner, "LEASE_PATHS", (absent,)):
                leases = runner.ReadOnlyLeases()
                with self.assertRaises(FileNotFoundError):
                    leases.acquire()
                self.assertFalse(absent.exists())

    def test_inherited_read_only_descriptors_match_exact_lease_inodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = (Path(temporary) / "gpu.lock", Path(temporary) / "ops.lock")
            for path in paths:
                path.touch()
            descriptors = tuple(os.open(path, os.O_RDONLY | os.O_CLOEXEC)
                                for path in paths)
            try:
                for descriptor in descriptors:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(runner, "LEASE_PATHS", paths):
                    observed = runner.validate_inherited_lease_fds(descriptors)
                    self.assertEqual([item["path"] for item in observed],
                                     [str(path) for path in paths])
                    replacement = paths[1].with_suffix(".new")
                    replacement.touch()
                    replacement.replace(paths[1])
                    with self.assertRaisesRegex(runner.GateError, "no longer names"):
                        runner.validate_inherited_lease_fds(descriptors)
            finally:
                for descriptor in descriptors:
                    os.close(descriptor)

    def test_inherited_descriptor_must_be_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gpu.lock"
            path.touch()
            descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC)
            try:
                with patch.object(runner, "LEASE_PATHS", (path,)), \
                        self.assertRaisesRegex(runner.GateError, "not read-only"):
                    runner.validate_inherited_lease_fds((descriptor,))
            finally:
                os.close(descriptor)

    def test_matrix_body_uses_inherited_leases_without_reacquiring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = (root / "gpu.lock", root / "ops.lock")
            for path in paths:
                path.touch()
            descriptors = tuple(os.open(path, os.O_RDONLY | os.O_CLOEXEC)
                                for path in paths)
            try:
                for descriptor in descriptors:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(runner, "LEASE_PATHS", paths), \
                        patch.object(runner, "matrix_plan", return_value=[]), \
                        patch.object(runner, "admission", return_value={"ok": True}):
                    result = runner.run_matrix(
                        root / "matrix", inherited_lease_fds=descriptors)
                self.assertTrue(result["complete"])
                self.assertEqual(result["passed_cells"], 0)
                self.assertEqual(result["lease_mode"], "validated_inherited")
                self.assertEqual(len(result["inherited_leases"]), 2)
            finally:
                for descriptor in descriptors:
                    os.close(descriptor)

    def test_owned_orphan_is_removed_after_leader_exit(self):
        code = "import os,time\npid=os.fork()\nif pid: os._exit(0)\ntime.sleep(15)\n"
        process = subprocess.Popen([sys.executable, "-c", code],
                                   start_new_session=True,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        try:
            process.wait(timeout=2)
            self.assertTrue(runner.group_members(process.pid))
            runner.stop_owned(process)
            self.assertEqual(runner.group_members(process.pid), [])
        finally:
            runner.stop_owned(process)


if __name__ == "__main__":
    unittest.main()
