"""Synthetic admission-record tests only; these are never live Q2 evidence."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("prefetch_runner", Path(__file__).with_name("run_safety.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def counters(mode):
    row = dict.fromkeys(("map_errors", "nesting_errors", "missing_frame", "identity_errors",
                         "order_errors", "read_errors", "request_errors", "action_errors",
                         "traversal_errors", "iterator_calls", "mask_bounds_errors"), 0)
    row.update(valid=True, empty_frames=True, action=runner.MODES[mode],
               mask_enter=2, mask_exit=2, wrapper_enter=3, wrapper_exit=3,
               decisions_complete=3, policy_calls=0 if mode == "native" else 3,
               setter_ok=0 if mode == "native" else 3,
               native_decisions=0 if mode == "bypass" else 3,
               bypass_decisions=3 if mode == "bypass" else 0,
               returned_default=3 if mode == "native" else 0,
               returned_bypass=3 if mode == "bypass" else 0,
               returned_invalid99=3 if mode == "invalid99" else 0,
               range_calls=0 if mode == "bypass" else 9,
               empty_masks=2, nonempty_masks=0,
               mask_samples=[{"cpu": 17, "first": 0, "outer": 512, "bitmap": [0] * 8}])
    counter_key = {"range_enter": "range_calls", "iterator_enter": "iterator_calls",
                   "gpu_page_prefetch": "policy_calls"}
    row["programs"] = [{"id": i + 1, "name": name, "recursion_misses": 0,
                        "run_count": row[counter_key.get(name, name)]}
                       for i, name in enumerate(("mask_enter", "mask_exit", "wrapper_enter", "wrapper_exit",
                                                 "range_enter", "iterator_enter", "gpu_page_prefetch"))]
    return row


class CounterGates(unittest.TestCase):
    def test_three_controls_require_real_traversal_not_nonempty_mask(self):
        for mode in runner.MODES:
            with self.subTest(mode=mode):
                runner.validate_metrics(mode, counters(mode))

    def test_missing_correlation_or_missed_program_is_rejected(self):
        for field in ("missing_frame", "map_errors", "traversal_errors", "iterator_calls"):
            row = counters("invalid99")
            row[field] = 1
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                runner.validate_metrics("invalid99", row)
        row = counters("invalid99")
        row["programs"][0]["recursion_misses"] = 1
        with self.assertRaises(RuntimeError):
            runner.validate_metrics("invalid99", row)
        row["programs"][0]["recursion_misses"] = 0
        row["programs"][0]["run_count"] += 1
        with self.assertRaises(RuntimeError):
            runner.validate_metrics("invalid99", row)

    def test_input_only_evidence_and_invalid_final_masks_are_rejected(self):
        row = counters("invalid99")
        row["range_calls"] = 0
        with self.assertRaises(RuntimeError):
            runner.validate_metrics("invalid99", row)
        for mode in ("bypass", "invalid99"):
            row = copy.deepcopy(counters(mode))
            row["mask_samples"][0].update(outer=16, bitmap=[1 << 32] + [0] * 7)
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                runner.validate_metrics(mode, row)


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
        self.assertEqual(record["monitor_cleanup"], [
            {"pid": 42004, "error": "RuntimeError: injected survivor", "returncode": None},
            {"pid": 42003, "error": None, "returncode": 0},
        ])


if __name__ == "__main__":
    unittest.main()
