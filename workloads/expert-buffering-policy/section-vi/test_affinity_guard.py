"""Synthetic guard tests: affinity and signals are always mocked."""
import argparse
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import Mock, patch

import affinity_guard as guard


class GuardTests(unittest.TestCase):
    def test_workspace_and_exact_start_ticks(self):
        expected = Path(__file__).resolve().parents[4]
        self.assertEqual(guard.WORKSPACE, expected)
        self.assertEqual(expected.name, "gpu")
        with patch.object(guard, "start_ticks", return_value=77), \
                patch.object(Path, "read_text", return_value="opencode\n"), \
                patch.object(Path, "resolve", return_value=expected) as resolve:
            self.assertEqual(guard.identity(123456, 77), 77)
            with self.assertRaises(RuntimeError):
                guard.identity(123456, 76)
            resolve.return_value = expected / "gpu_ext"
            with self.assertRaises(RuntimeError):
                guard.identity(123456, 77)

    def test_restore_new_thread_and_preserve_external_mask_and_reused_pid(self):
        masks, calls = {1: [17], 2: [6]}, 0

        def snapshot(_pid):
            nonlocal calls
            calls += 1
            if calls == 2:
                masks[3] = [17]  # Created during restoration by a still-pinned thread.
            return [dict(tid=tid, start_ticks=7, cpus=list(mask)) for tid, mask in masks.items()]

        with patch.object(guard, "identity"), patch.object(guard, "threads", side_effect=snapshot), \
                patch.object(guard, "start_ticks", return_value=7), \
                patch.object(guard.os, "sched_getaffinity", side_effect=lambda tid: masks[tid]), \
                patch.object(guard.os, "sched_setaffinity", side_effect=lambda tid, mask: masks.update({tid: list(mask)})) as setmask:
            result = guard.restore(123456, 77, [0, 17], [17])
            self.assertEqual([call.args[0] for call in setmask.call_args_list], [1, 3])
            self.assertEqual(masks, {1: [0, 17], 2: [6], 3: [0, 17]})
            self.assertTrue(any(t["status"] == "preserved_external_mask" for t in result["actions"]))
            setmask.reset_mock()
            with patch.object(guard, "identity", side_effect=RuntimeError("reused PID")):
                with self.assertRaises(RuntimeError):
                    guard.restore(123456, 77, [0, 17], [17])
            setmask.assert_not_called()

    def test_signal_cleanup_then_restore_with_durable_original_record(self):
        masks, handlers, order = {1: [0, 17]}, {}, []
        child = Mock(pid=234567, returncode=None)
        child.poll.side_effect = lambda: child.returncode

        def wait(timeout):
            self.assertEqual(timeout, 30)
            order.append("cooperative_wait")
            child.returncode = -signal.SIGTERM
            return child.returncode

        def register(sig, handler):
            previous = handlers.get(sig, signal.SIG_DFL)
            handlers[sig] = handler
            return previous

        def interrupt(_seconds):
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            self.assertEqual(handlers[signal.SIGTERM], signal.SIG_IGN)
            self.assertEqual(handlers[signal.SIGINT], signal.SIG_IGN)

        child.wait.side_effect = wait
        with tempfile.TemporaryDirectory(prefix="eb-affinity-test-") as directory:
            path = Path(directory) / "new.json"

            def setmask(tid, mask):
                saved = json.loads(path.read_text())
                self.assertEqual(saved["initial_threads"][0]["cpus"], [0, 17])
                if list(mask) == [0, 17]:
                    self.assertEqual(order, ["cooperative_wait", "owned_cleanup"])
                    order.append("restore")
                masks[tid] = list(mask)

            args = argparse.Namespace(pid=123456, cpu=17, start_ticks=77, record=path, command=["fake-command"])
            with patch.object(guard, "identity", return_value=77), \
                    patch.object(guard, "threads", side_effect=lambda _: [dict(tid=1, start_ticks=7, cpus=list(masks[1]))]), \
                    patch.object(guard, "start_ticks", return_value=7), \
                    patch.object(guard.os, "sched_getaffinity", side_effect=lambda tid: masks[tid]), \
                    patch.object(guard.os, "sched_setaffinity", side_effect=setmask), \
                    patch.object(guard.signal, "signal", side_effect=register), \
                    patch.object(guard.subprocess, "Popen", return_value=child) as launch, \
                    patch.object(guard.os, "killpg") as kill, \
                    patch.object(guard.owned, "stop_owned", side_effect=lambda _: order.append("owned_cleanup")), \
                    patch.object(guard.time, "sleep", side_effect=interrupt):
                self.assertEqual(guard.run(args), 1)
            launch.assert_called_once_with(["fake-command"], start_new_session=True)
            kill.assert_called_once_with(child.pid, signal.SIGTERM)
            record = json.loads(path.read_text())
            self.assertFalse(record["complete"])
            self.assertEqual(record["checks"], 1)
            self.assertEqual(record["signal"], signal.SIGTERM)
            self.assertTrue(record["owned_child_group_empty"])
            self.assertEqual(masks[1], [0, 17])
            self.assertEqual(order, ["cooperative_wait", "owned_cleanup", "restore"])
            self.assertEqual(handlers[signal.SIGTERM], signal.SIG_DFL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
