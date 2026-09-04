#!/usr/bin/env python3
"""Pure mocked-child tests: never launch the cell CLI or touch the GPU."""

from contextlib import contextmanager
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import run_batch_575 as batch


class BatchTests(unittest.TestCase):
    @contextmanager
    def fixture(self, failures=None, interrupt_at=None, popen_error_at=None):
        failures = failures or {}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "batch"
            traced = root / "preflight"
            correctness = root / "correctness"
            for path, config in [(traced, "lmcache_disk")] + [
                    (correctness / name, name) for name in batch.CONFIGS]:
                path.mkdir(parents=True)
                (path / "result.json").write_text(json.dumps({"config": config, "prefix_count": 8}))
                (path / "environment.json").write_text(json.dumps({"gpu": {"driver": "575.57.08"}}))
            calls, processes = [], []
            handlers = {signal.SIGINT: object(), signal.SIGTERM: object()}
            previous = handlers.copy()

            def set_handler(signum, handler):
                old, handlers[signum] = handlers[signum], handler
                return old

            def start(command, **kwargs):
                index = len(calls)
                calls.append((command, kwargs))
                if index == popen_error_at:
                    raise OSError("simulated launch failure")
                process = Mock(pid=10000 + index)
                triggered = False
                def wait(timeout=None):
                    nonlocal triggered
                    if index == interrupt_at and not triggered:
                        triggered = True
                        handlers[signal.SIGTERM](signal.SIGTERM, None)
                        handlers[signal.SIGINT](signal.SIGINT, None)
                        raise subprocess.TimeoutExpired(command, timeout)
                    return failures.get(index, 0)
                process.wait.side_effect = wait
                processes.append(process)
                return process

            with (patch.object(batch, "OUTPUT", output), patch.object(batch, "TRACED", traced),
                  patch.object(batch, "CORRECTNESS", correctness),
                  patch.object(batch.signal, "signal", side_effect=set_handler),
                  patch.object(batch.subprocess, "Popen", side_effect=start),
                  patch("builtins.print")):
                yield output, traced, correctness, calls, processes
            self.assertEqual(handlers, previous, "original signal handlers must be restored")

    def test_exact_fixed_pipeline_and_owned_children(self):
        with self.fixture() as (output, traced, correctness, calls, processes):
            self.assertEqual(batch.main([]), 0)
            self.assertEqual(len(calls), 63)  # two prerequisites + 30 run/validate pairs + analysis
            self.assertEqual(calls[0][0][-3:], ["validate-cell", str(traced), "--require-trace"])
            self.assertEqual(calls[1][0][-4:], ["compare-outputs", *[str(correctness / name) for name in batch.CONFIGS]])
            schedule = json.loads(batch.SCHEDULE.read_text())
            expected = [(row["attempt"], position, config) for row in schedule["attempts"][:10]
                        for position, config in enumerate(row["order"])]
            for index, (attempt, position, config) in enumerate(expected):
                command = calls[2 + index * 2][0]
                cell = output / f"attempt-{attempt:02d}" / f"position-{position}-{config}"
                self.assertEqual(command[:3], ["/usr/bin/taskset", "-c", "8-16"])
                self.assertEqual(command[-9:], ["run-cell", "--expected-driver", "575.57.08",
                                              "--prefix-limit", "8", "--config", config, "--output", str(cell)])
                self.assertNotIn("--trace", command)
                self.assertEqual(calls[3 + index * 2][0][-2:], ["validate-cell", str(cell)])
            self.assertEqual(calls[-1][0][-2:], ["analyze", str(output)])
            for command, kwargs in calls:
                self.assertTrue(kwargs["start_new_session"])
                self.assertEqual(kwargs["env"]["HF_HOME"], batch.HF_CACHE)
                self.assertEqual(kwargs["cwd"], batch.HERE)
                self.assertEqual(command[3:7], [str(batch.PYTHON), "-u", "-B", str(batch.RUNNER)])
                if "run-cell" not in command:
                    self.assertEqual(command[2], "17")
            journal = (output / "batch.log").read_text()
            self.assertEqual(journal.count(" START "), 63)
            self.assertEqual(journal.count(" END "), 63)
            self.assertIn("returncode=0", journal)
            self.assertIn("cell=30/30 block=10/10", journal)
            self.assertFalse((output / "attempt-10").exists())
            self.assertEqual(len(list((output / "logs").iterdir())), 63)
            for process in processes:
                process.kill.assert_not_called()
                process.terminate.assert_not_called()
                process.send_signal.assert_not_called()

    def test_existing_output_is_never_overwritten(self):
        with self.fixture() as (output, _, _, calls, _):
            output.mkdir()
            sentinel = output / "existing.txt"
            sentinel.write_text("preserve")
            self.assertEqual(batch.main([]), 2)
            self.assertEqual(calls, [])
            self.assertEqual(sentinel.read_text(), "preserve")
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_each_child_failure_stops_without_retry_or_analysis(self):
        for failed_index in (0, 1, 2, 3, 8, 62):
            with self.subTest(index=failed_index), self.fixture({failed_index: 7}) as (output, _, _, calls, _):
                self.assertEqual(batch.main([]), 2)
                self.assertEqual(len(calls), failed_index + 1)
                failures = list(output.rglob("failure.md"))
                self.assertEqual(len(failures), 1)
                self.assertIn("exited 7", failures[0].read_text())
                if 2 <= failed_index < 62:
                    self.assertEqual(failures[0].parent.name, f"attempt-{(failed_index - 2) // 6:02d}")
                if failed_index < 62:
                    self.assertNotIn("analyze", calls[-1][0])

    def test_launch_error_is_preserved_without_retry(self):
        with self.fixture(popen_error_at=2) as (output, _, _, calls, _):
            self.assertEqual(batch.main([]), 2)
            self.assertEqual(len(calls), 3)
            self.assertIn("simulated launch failure", (output / "attempt-00/failure.md").read_text())
            self.assertIn("returncode=None", (output / "batch.log").read_text())

    def test_prerequisite_scope_rejects_small_wrong_driver_or_traced_correctness(self):
        for defect in ("prefix", "driver", "config", "trace"):
            with self.subTest(defect=defect), self.fixture() as (output, _, correctness, calls, _):
                cell = correctness / "recompute"
                if defect == "prefix":
                    (cell / "result.json").write_text('{"config":"recompute","prefix_count":1}')
                elif defect == "driver":
                    (cell / "environment.json").write_text('{"gpu":{"driver":"610.43.02"}}')
                elif defect == "config":
                    (cell / "result.json").write_text('{"config":"lmcache_cpu","prefix_count":8}')
                else:
                    (cell / "strace").mkdir()
                self.assertEqual(batch.main([]), 2)
                self.assertEqual(len(calls), 2)
                self.assertFalse((output / "attempt-00").exists())

    def test_existing_schedule_validator_runs_after_prerequisites(self):
        with self.fixture() as (output, _, _, calls, _):
            def reject(_schedule):
                self.assertEqual(len(calls), 2)
                raise ValueError("schedule changed")
            with patch.object(batch, "validate_schedule", side_effect=reject):
                self.assertEqual(batch.main([]), 2)
            self.assertEqual(len(calls), 2)
            self.assertIn("schedule changed", (output / "failure.md").read_text())

    def test_deferred_stop_reaps_current_cell_and_validates_without_next_cell(self):
        for interrupted_index, expected_count in ((0, 1), (2, 4), (3, 4)):
            with (self.subTest(index=interrupted_index),
                  self.fixture(interrupt_at=interrupted_index) as (output, _, _, calls, processes),
                  patch.object(batch.os, "killpg") as killpg):
                self.assertEqual(batch.main([]), 128 + signal.SIGTERM)
                self.assertEqual(len(calls), expected_count)
                self.assertGreaterEqual(processes[interrupted_index].wait.call_count, 3)
                journal = (output / "batch.log").read_text()
                self.assertEqual(journal.count("PENDING STOP"), 1)
                self.assertIn("returncode=0", journal)
                self.assertIn("deferred SIGTERM", next(output.rglob("failure.md")).read_text())
                if interrupted_index == 2:
                    self.assertIn("validate-cell", calls[-1][0])
                killpg.assert_not_called()
                for process in processes:
                    process.kill.assert_not_called()
                    process.terminate.assert_not_called()
                    process.send_signal.assert_not_called()

    def test_unexpected_wait_error_still_reaps_owned_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "logs").mkdir()
            process = Mock(pid=12345)
            process.wait.side_effect = [RuntimeError("simulated wait failure"), 0]
            with ((output / "batch.log").open("x") as journal,
                  patch.object(batch.subprocess, "Popen", return_value=process),
                  patch("builtins.print")):
                with self.assertRaisesRegex(RuntimeError, "simulated wait failure"):
                    batch.run_step("test", ["validate-cell", "/unused"], "17", output, journal, batch.DeferredStop())
            self.assertEqual(process.wait.call_count, 2)
            process.kill.assert_not_called()
            self.assertIn("END test returncode=0", (output / "batch.log").read_text())


if __name__ == "__main__":
    unittest.main()
