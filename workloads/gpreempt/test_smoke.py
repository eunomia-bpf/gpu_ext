#!/usr/bin/env python3
"""CPU-only runner tests: every GPU/process observation is replaced by a mock."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("gpreempt_smoke_under_test", Path(__file__).with_name("run_smoke.py"))
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeTests(unittest.TestCase):
    def test_lease_failure_is_recorded_without_gpu_observation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result"
            with patch.object(smoke.safety.LeaseSet, "acquire", side_effect=RuntimeError("busy")), \
                 patch.object(smoke.safety, "safety_snapshot") as snapshot:
                with self.assertRaisesRegex(RuntimeError, "busy"):
                    smoke.run(output)
                snapshot.assert_not_called()
            self.assertEqual(json.loads((output / "result.json").read_text())["status"], "failed")

    def run_mocked(self, temporary, returncode=0, timeout=False):
        output = Path(temporary) / "result"
        lease = Mock()
        process = Mock()
        process.wait.side_effect = subprocess.TimeoutExpired("mock-child", 30) if timeout else None
        process.wait.return_value = returncode
        def start(argv, **kwargs):
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(set(kwargs["env"]), {"PATH", "LANG", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"})
            self.assertTrue(kwargs["env"]["LD_LIBRARY_PATH"].startswith(
                str(smoke.HERE / "deps/gdrcopy-2.5.2/src") + ":"))
            self.assertEqual(argv[:3], ["taskset", "-c", "8-15"])
            kwargs["stdout"].write("PASS set priority\nPASS GDRcopy flag roundtrip\n"
                "PASS all (finite smoke only; not scheduling performance)\n")
            return process
        with patch.object(smoke.safety.LeaseSet, "acquire", return_value=lease), \
             patch.object(smoke.safety, "safety_snapshot", return_value={"gpu": {"driver": "575.57.08"}}), \
             patch.object(smoke.safety, "validate_pre_server_safety"), \
             patch.object(smoke.safety, "wait_for_post_server_safety", return_value={"mock": True}), \
             patch.object(smoke.safety, "stop_owned_process_group") as stop, \
             patch.object(smoke.subprocess, "Popen", side_effect=start), \
             patch.object(smoke.Path, "exists", return_value=True):
            if returncode or timeout:
                with self.assertRaises((RuntimeError, subprocess.TimeoutExpired)):
                    smoke.run(output)
            else:
                smoke.run(output)
            stop.assert_called_once_with(process)
            lease.close.assert_called_once()
        return json.loads((output / "result.json").read_text())

    def test_success_requires_checks_and_records_no_performance_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_mocked(temporary)
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["performance_claim"])
        self.assertFalse(result["effective_timeslice_verified"])

    def test_nonzero_child_fails_despite_pass_strings(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_mocked(temporary, returncode=1)
        self.assertEqual(result["status"], "failed")

    def test_timeout_cleans_only_owned_process_and_keeps_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_mocked(temporary, timeout=True)
        self.assertEqual(result["status"], "failed")
        self.assertIn("timed out", result["error"])


if __name__ == "__main__":
    unittest.main()
