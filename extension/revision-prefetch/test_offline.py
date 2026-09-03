"""Synthetic admission-record tests only; these are never live Q2 evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("prefetch_runner", Path(__file__).with_name("run_safety.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


ERRORS = ("map_errors", "nesting_errors", "missing_frame", "order_errors",
          "read_errors", "request_errors", "action_errors", "state_errors",
          "phase_errors", "traversal_errors", "output_errors")


def counters(mode):
    row = dict.fromkeys(ERRORS, 0)
    decisions = 3
    native = mode != "bypass"
    row.update(valid=True, empty_frames=True, action=runner.MODES[mode],
               wrapper_enter=decisions, wrapper_exit=decisions,
               policy_calls=0 if mode == "native" else decisions,
               setter_ok=0 if mode == "native" else decisions,
               diagnostic_calls=2 * decisions, selected_events=decisions,
               finished_events=decisions, decisions_complete=decisions,
               returned_default=decisions if mode == "native" else 0,
               returned_bypass=decisions if mode == "bypass" else 0,
               returned_invalid99=decisions if mode == "invalid99" else 0,
               region_noop_default=decisions if mode == "native" else 0,
               region_apply=0 if mode == "native" else decisions,
               native_effects=decisions if native else 0,
               bypass_effects=0 if native else decisions,
               native_completions=decisions if native else 0,
               native_iterations=9 if native else 0,
               empty_outputs=decisions, nonempty_outputs=0)
    program_runs = {"wrapper_enter": decisions, "wrapper_exit": decisions,
                    "diagnostic_enter": 2 * decisions,
                    "gpu_page_prefetch": row["policy_calls"]}
    row["programs"] = [{"id": index + 1, "name": name, "recursion_misses": 0,
                        "run_count": runs}
                       for index, (name, runs) in enumerate(program_runs.items())]
    return row


VALID_BTF = """[1] INT 'long long' size=8 bits_offset=0 nr_bits=64 encoding=SIGNED
[2] INT 'unsigned long long' size=8 bits_offset=0 nr_bits=64 encoding=(none)
[3] INT 'unsigned int' size=4 bits_offset=0 nr_bits=32 encoding=(none)
[10] STRUCT 'uvm_bpf_prefetch_diagnostic_ctx' size=88 vlen=14
\t'raw_action' type_id=1 bits_offset=0
\t'requested_first' type_id=2 bits_offset=64
\t'requested_outer' type_id=2 bits_offset=128
\t'max_first' type_id=2 bits_offset=192
\t'max_outer' type_id=2 bits_offset=256
\t'output_first' type_id=2 bits_offset=320
\t'output_outer' type_id=2 bits_offset=384
\t'phase' type_id=3 bits_offset=448
\t'request_attempted' type_id=3 bits_offset=480
\t'request_conflict' type_id=3 bits_offset=512
\t'initial_region_result' type_id=3 bits_offset=544
\t'initial_effect' type_id=3 bits_offset=576
\t'native_iterations' type_id=3 bits_offset=608
\t'native_completed' type_id=3 bits_offset=640
[11] CONST '(anon)' type_id=10
[12] PTR '(anon)' type_id=11
[13] FUNC_PROTO '(anon)' ret_type_id=0 vlen=1
\t'ctx' type_id=12
[14] FUNC 'uvm_bpf_prefetch_diagnostic' type_id=13 linkage=static
[15] ENUM 'uvm_bpf_prefetch_diagnostic_phase' encoding=UNSIGNED size=4 vlen=2
\t'UVM_BPF_PREFETCH_DIAG_SELECTED' val=1
\t'UVM_BPF_PREFETCH_DIAG_FINISHED' val=2
"""


class CounterGates(unittest.TestCase):
    def test_three_controls_require_exact_transition_phases(self):
        for mode in runner.MODES:
            with self.subTest(mode=mode):
                runner.validate_metrics(mode, counters(mode))

    def test_observer_errors_or_missed_program_are_rejected(self):
        for field in ERRORS:
            row = counters("invalid99")
            row[field] = 1
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                runner.validate_metrics("invalid99", row)
        row = counters("invalid99")
        row["programs"][0]["recursion_misses"] = 1
        with self.assertRaises(RuntimeError):
            runner.validate_metrics("invalid99", row)
        row = counters("invalid99")
        row["programs"][2]["run_count"] = 3
        with self.assertRaises(RuntimeError):
            runner.validate_metrics("invalid99", row)

    def test_missing_phase_or_wrong_fallback_is_rejected(self):
        for field in ("selected_events", "finished_events", "decisions_complete"):
            row = counters("invalid99")
            row[field] -= 1
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                runner.validate_metrics("invalid99", row)
        for field in ("native_completions", "native_iterations"):
            row = counters("invalid99")
            row[field] = 0
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                runner.validate_metrics("invalid99", row)
        row = counters("bypass")
        row.update(native_completions=1, native_iterations=1)
        with self.assertRaises(RuntimeError):
            runner.validate_metrics("bypass", row)

    def test_output_and_program_sets_are_reconciled(self):
        row = counters("bypass")
        row.update(empty_outputs=2, nonempty_outputs=1)
        with self.assertRaises(RuntimeError):
            runner.validate_metrics("bypass", row)
        row = counters("native")
        row["programs"][2]["name"] = "range_enter"
        with self.assertRaises(RuntimeError):
            runner.validate_metrics("native", row)


class InterfaceGates(unittest.TestCase):
    def test_exact_address_free_void_const_pointer_interface(self):
        runner.validate_diagnostic_interface(VALID_BTF)
        for changed in (
            VALID_BTF.replace("size=88", "size=104"),
            VALID_BTF.replace("bits_offset=448", "bits_offset=512"),
            VALID_BTF.replace("ret_type_id=0", "ret_type_id=1"),
            VALID_BTF.replace("type_id=11\n[13]", "type_id=10\n[13]"),
            VALID_BTF.replace("FINISHED' val=2", "FINISHED' val=3"),
        ):
            with self.subTest(), self.assertRaises(RuntimeError):
                runner.validate_diagnostic_interface(changed)

    def test_fixture_persists_no_kernel_address(self):
        source = Path(__file__).with_name("fixture.bpf.c").read_text()
        header = Path(__file__).with_name("fixture.h").read_text()
        self.assertNotIn("pointer_id", source)
        self.assertNotIn("range_enter", source)
        self.assertNotIn("unsigned long long tree", header)
        self.assertNotIn("unsigned long long mask", header)
        self.assertIn('fentry/uvm_bpf_prefetch_diagnostic', source)


class ComputeMonitorGates(unittest.TestCase):
    @staticmethod
    def write_window(path, *, gap=10, foreign=False):
        target = 123
        starts = [100, 110, 120, 130, 140 + gap - 10, 150 + gap - 10]
        finishes = [value + 2 for value in starts]
        rows = [
            {"event": "sample", "wall_time_ns": 1000,
             "query_started_mono_ns": starts[0], "query_finished_mono_ns": finishes[0], "pids": []},
            {"event": "sample", "wall_time_ns": 1001,
             "query_started_mono_ns": starts[1], "query_finished_mono_ns": finishes[1], "pids": [target]},
            {"event": "sample", "wall_time_ns": 1002,
             "query_started_mono_ns": starts[2], "query_finished_mono_ns": finishes[2], "pids": [target]},
            {"event": "sample", "wall_time_ns": 1003,
             "query_started_mono_ns": starts[3], "query_finished_mono_ns": finishes[3],
             "pids": [target, 456] if foreign else [target]},
            {"event": "sample", "wall_time_ns": 1004,
             "query_started_mono_ns": starts[4], "query_finished_mono_ns": finishes[4], "pids": [target]},
            {"event": "sample", "wall_time_ns": 1005,
             "query_started_mono_ns": starts[5], "query_finished_mono_ns": finishes[5], "pids": []},
            {"event": "final", "wall_time_ns": 1006,
             "monotonic_ns": finishes[5] + 5, "errors": 0},
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return {
            "max_sample_gap_ns": runner.COMPUTE_MAX_GAP_NS,
            "pretarget_empty_query_started_mono_ns": starts[0],
            "pretarget_empty_query_finished_mono_ns": finishes[0],
            "target_started_mono_ns": 105,
            "pause_observed_mono_ns": 108,
            "paused_target_query_started_mono_ns": starts[1],
            "paused_target_query_finished_mono_ns": finishes[1],
            "ready_observed_mono_ns": 115,
            "ready_target_query_started_mono_ns": starts[2],
            "ready_target_query_finished_mono_ns": finishes[2],
            "released_mono_ns": 125,
            "post_release_target_query_started_mono_ns": starts[3],
            "post_release_target_query_finished_mono_ns": finishes[3],
            "target_exit_mono_ns": starts[4] + 5,
            "loader_stopped_mono_ns": starts[4] + 7,
            "post_exit_empty_query_started_mono_ns": starts[5],
            "post_exit_empty_query_finished_mono_ns": finishes[5],
        }

    def test_only_owned_target_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "apps.jsonl"
            window = self.write_window(path)
            result = runner.validate_compute_monitor(path, 123, window)
            self.assertEqual(result["target_samples"], 4)
            window = self.write_window(path, foreign=True)
            with self.assertRaises(RuntimeError):
                runner.validate_compute_monitor(path, 123, window)

    def test_lifecycle_hole_or_missing_tail_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "apps.jsonl"
            window = self.write_window(path, gap=runner.COMPUTE_MAX_GAP_NS + 20)
            with self.assertRaises(RuntimeError):
                runner.validate_compute_monitor(path, 123, window)
            window = self.write_window(path)
            window["post_exit_empty_query_started_mono_ns"] += 1
            with self.assertRaises(RuntimeError):
                runner.validate_compute_monitor(path, 123, window)

    def test_wall_clock_step_does_not_define_cross_process_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "apps.jsonl"
            window = self.write_window(path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for index, row in enumerate(rows):
                row["wall_time_ns"] = 10_000 - index
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            runner.validate_compute_monitor(path, 123, window)

    def test_release_timestamp_follows_successful_pipe_close(self):
        source = Path(__file__).with_name("run_safety.py").read_text()
        pause_log = source.index('while f"MONITOR_PID: {target.pid}\\n"')
        pause_marker = source.index(
            'record["compute_window"]["pause_observed_mono_ns"] = time.monotonic_ns()')
        paused_sample = source.index(
            'after_ns=record["compute_window"]["pause_observed_mono_ns"]')
        self.assertLess(pause_log, pause_marker)
        self.assertLess(pause_marker, paused_sample)
        close = source.index('target.stdin.close()')
        released = source.index('record["compute_window"]["released_mono_ns"] = time.monotonic_ns()')
        post_release = source.index('after_ns=record["compute_window"]["released_mono_ns"]')
        self.assertLess(close, released)
        self.assertLess(released, post_release)
        loader_stopped = source.index(
            'record["compute_window"]["loader_stopped_mono_ns"] = time.monotonic_ns()')
        tail = source.index('after_ns=record["compute_window"]["loader_stopped_mono_ns"]')
        self.assertLess(loader_stopped, tail)

    def test_live_reader_ignores_an_incomplete_last_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "apps.jsonl"
            complete = {"event": "sample", "monotonic_ns": 1, "pids": []}
            path.write_text(json.dumps(complete) + '\n{"event": "sample"')
            samples, rows = runner.read_compute_samples(path)
            self.assertEqual(samples, [complete])
            self.assertEqual(rows, [complete])

    def test_query_started_before_gate_cannot_be_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "apps.jsonl"
            stale = {"event": "sample", "query_started_mono_ns": 100,
                     "query_finished_mono_ns": 200, "pids": [123]}
            path.write_text(json.dumps(stale) + "\n")
            process = mock.Mock(returncode=None)
            process.poll.return_value = None
            with self.assertRaises(RuntimeError):
                runner.wait_compute_sample(
                    path, process, lambda pids: pids == [123], after_ns=150, timeout=0.001)

    def test_two_individually_bounded_queries_cannot_create_two_second_cadence(self):
        bound = runner.COMPUTE_MAX_GAP_NS
        rows = [
            {"query_started_mono_ns": 1, "query_finished_mono_ns": bound},
            {"query_started_mono_ns": 2 * bound - 1,
             "query_finished_mono_ns": 2 * bound},
        ]
        with self.assertRaises(RuntimeError):
            runner.validate_sample_cadence(rows, bound)

    def test_signal_handler_queues_without_raising(self):
        runner.INTERRUPTED_SIGNALS.clear()
        try:
            runner.note_interrupt(signal.SIGINT, None)
            self.assertEqual(runner.INTERRUPTED_SIGNALS, [signal.SIGINT])
            with self.assertRaises(InterruptedError):
                runner.raise_if_interrupted()
        finally:
            runner.INTERRUPTED_SIGNALS.clear()

    def test_link_check_cannot_be_skipped_by_monitor_cleanup(self):
        source = Path(__file__).with_name("run_safety.py").read_text()
        link_check = source.index('if record.get("ready") and loader_stopped:')
        monitor_cleanup = source.index('stop_monitors(monitors, record)', link_check)
        self.assertLess(link_check, monitor_cleanup)

    def test_body_exception_clears_complete_before_recording_error(self):
        source = Path(__file__).with_name("run_safety.py").read_text()
        body_except = source.index("    except BaseException as error:", source.index("def run_cell"))
        cleared = source.index('record["complete"] = False', body_except)
        recorded = source.index('record["error"] =', body_except)
        self.assertLess(cleared, recorded)


class CleanupGates(unittest.TestCase):
    def test_empty_group_does_not_shortcut_leader_reaping(self):
        process = mock.Mock(pid=42001)
        process.poll.side_effect = [None, None, 0]
        with mock.patch.object(runner.owned, "group_members", return_value=[]), \
             mock.patch.object(runner.os, "killpg", side_effect=ProcessLookupError), \
             mock.patch.object(runner.time, "sleep"):
            runner.stop_owned(process)
        process.wait.assert_not_called()
        self.assertEqual(process.poll.call_count, 3)

        process = mock.Mock(pid=42002)
        process.poll.return_value = None
        with mock.patch.object(runner.owned, "group_members", return_value=[]), \
             mock.patch.object(runner.os, "killpg", side_effect=ProcessLookupError), \
             mock.patch.object(runner.time, "monotonic", side_effect=[0, 9, 9, 15, 15, 21]), \
             self.assertRaisesRegex(RuntimeError, "survived cleanup"):
            runner.stop_owned(process)
        process.wait.assert_not_called()

    def test_monitor_failure_does_not_skip_remaining_owned_monitor(self):
        first, second = mock.Mock(pid=42003), mock.Mock(pid=42004)
        first.poll.return_value = 0
        second.poll.return_value = None
        record = {"complete": True}
        with mock.patch.object(runner, "stop_owned", side_effect=[RuntimeError("injected survivor"), None]) as stop, \
             self.assertRaisesRegex(RuntimeError, "owned monitor cleanup failed"):
            runner.stop_monitors([first, second], record)
        self.assertEqual(stop.call_args_list, [mock.call(second), mock.call(first)])
        self.assertFalse(record["complete"])


if __name__ == "__main__":
    unittest.main()
