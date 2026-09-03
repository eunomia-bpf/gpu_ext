"""Bounded CPU regressions; no CUDA imports or real affinity changes."""
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import build_adapter as owned


class OwnedCleanupTests(unittest.TestCase):
    def test_empty_group_waits_for_leader_reap(self):
        process = Mock(pid=123456, poll=Mock(side_effect=[None, None, 0]))
        with patch.object(owned, "group_members", return_value=[]), \
                patch.object(owned.os, "killpg", side_effect=ProcessLookupError) as kill, \
                patch.object(owned.time, "sleep") as sleep:
            owned.stop_owned(process)
        kill.assert_called_once_with(process.pid, owned.signal.SIGTERM)
        sleep.assert_called_once_with(0.05)
        process.wait.assert_not_called()

    def test_already_reaped_owned_child_returns_without_signal(self):
        process = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
        try:
            self.assertEqual(process.wait(timeout=2), 0)
            with patch.object(owned.os, "killpg") as kill:
                owned.stop_owned(process)
            kill.assert_not_called()
            self.assertEqual(owned.group_members(process.pid), [])
        finally:
            owned.stop_owned(process)


if __name__ == "__main__":
    unittest.main(verbosity=2)
