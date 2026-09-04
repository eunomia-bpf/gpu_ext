"""Offline evidence-gate tests; these strings are fixtures, not device results."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_smoke as smoke


POSITIVE = (
    "GPU eBPF verification accepted: mode=STRICT program=cuda__count_ret "
    "attach=kretprobe/_Z9vectorAddPKfS0_Pfi instructions=13\n"
    "GPU eBPF verified map: program=cuda__count_ret fd=17 "
    "type=1502 key_size=4 value_size=8 max_entries=1\n"
)
NEGATIVE = (
    "GPU eBPF verification failed for cuda__count_ret: "
    "branch predicate is lane-varying (mode=STRICT, policy_entry_created=0)\n"
    "GPU verifier rejected handler 18\n"
    "Failed to initialize attach context, exiting..\n"
)


class StrictEvidenceTests(unittest.TestCase):
    def test_requires_actual_positive_admission_and_map(self):
        smoke.require_strict_verdict(POSITIVE, False)
        for text in ("", POSITIVE.replace("type=1502", "type=1503"),
                     POSITIVE.replace("max_entries=1", "max_entries=16"),
                     POSITIVE.replace("mode=STRICT", "mode=WARNING"), POSITIVE + NEGATIVE):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                smoke.require_strict_verdict(text, False)

    def test_matches_exact_syscall_name_not_elf_name_or_prefix(self):
        self.assertEqual(smoke.KERNEL_PROGRAM_NAME, "cuda__count_ret")
        for name in ("cuda__count_return", "cuda__count_re", "cuda__other_ret"):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                smoke.require_strict_verdict(POSITIVE.replace("cuda__count_ret", name), False)
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                smoke.require_strict_verdict(NEGATIVE.replace("cuda__count_ret", name), True)

    def test_negative_requires_specific_rejection_and_propagation(self):
        smoke.require_strict_verdict(NEGATIVE, True)
        for line in NEGATIVE.splitlines():
            with self.subTest(missing=line), self.assertRaises(RuntimeError):
                smoke.require_strict_verdict(NEGATIVE.replace(line, ""), True)
        for extra in (POSITIVE, "Recorded pass test", "Skipping GPU eBPF verification", "; continuing"):
            with self.subTest(extra=extra), self.assertRaises(RuntimeError):
                smoke.require_strict_verdict(NEGATIVE + extra, True)

    def test_negative_needs_nonempty_complete_zero_snapshot(self):
        zero = dict(device_thread_returns=0, nonzero_threads=0,
                    threads_with_eight_returns=0, maximum_returns=0)
        smoke.require_zero_counters([zero])
        for snapshots in ([], [{}], [{**zero, "device_thread_returns": 1}],
                          [{**zero, "nonzero_threads": False}], [{**zero, "maximum_returns": 0.0}],
                          [zero, {**zero, "maximum_returns": 1}]):
            with self.subTest(snapshots=snapshots), self.assertRaises(RuntimeError):
                smoke.require_zero_counters(snapshots)

    def test_unverified_build_and_unsafe_negative_stop_before_any_lease(self):
        with patch.object(smoke.Path, "is_file", return_value=False), \
                patch.object(smoke.safety.LeaseSet, "acquire") as acquire:
            with self.assertRaises(RuntimeError):
                smoke.run(Path("unused-output"), Path("unused-build"), strict=True)
            with self.assertRaises(RuntimeError):
                smoke.run(Path("unused-output"), Path("unused-build"), negative=True)
            acquire.assert_not_called()

    def test_pair_stops_if_positive_fails(self):
        with patch.object(smoke.sys, "argv", ["run_smoke.py", "--strict", "--output", "unused"]), \
                patch.object(smoke.Path, "exists", return_value=False), \
                patch.object(smoke.signal, "signal"), \
                patch.object(smoke, "run", side_effect=RuntimeError("positive failed")) as run:
            with self.assertRaisesRegex(RuntimeError, "positive failed"):
                smoke.main()
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.kwargs, {"strict": True})

    def test_strict_pair_uses_positive_then_negative(self):
        with patch.object(smoke.sys, "argv", ["run_smoke.py", "--strict", "--output", "unused"]), \
                patch.object(smoke.Path, "exists", return_value=False), \
                patch.object(smoke.signal, "signal"), \
                patch.object(smoke, "run", return_value={"status": "passed"}) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(smoke.main(), 0)
            self.assertEqual([call.kwargs for call in run.call_args_list],
                             [{"strict": True}, {"strict": True, "negative": True}])


class OwnedCleanupTests(unittest.TestCase):
    def test_orphan_is_stopped_after_owned_leader_exits(self):
        # Finite CPU-only child, using GPreempt's existing orphan regression case.
        code = "import os,time\npid=os.fork()\nif pid: os._exit(0)\ntime.sleep(15)\n"
        child = subprocess.Popen([sys.executable, "-c", code], start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            child.wait(timeout=2)
            self.assertTrue(smoke.group_members(child.pid))
            smoke.stop_owned(child)
            self.assertEqual(smoke.group_members(child.pid), [])
        finally:
            smoke.stop_owned(child)

    def test_only_recorded_regular_file_is_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "segment"
            path.touch()
            identity = smoke.segment_identity(path)
            with self.assertRaises(RuntimeError):
                smoke.unlink_owned_segment(path, None)
            self.assertTrue(path.exists())
            smoke.unlink_owned_segment(path, identity)
            self.assertFalse(path.exists())
            smoke.unlink_owned_segment(path, identity)  # Already removed by its owner.

    def test_replaced_file_and_dangling_symlink_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "segment"
            path.touch()
            identity = smoke.segment_identity(path)
            path.rename(Path(temporary) / "original")
            path.touch()
            with self.assertRaises(RuntimeError):
                smoke.unlink_owned_segment(path, identity)
            self.assertTrue(path.exists())
            path.unlink()
            path.symlink_to("absent-target")
            self.assertTrue(smoke.os.path.lexists(path))
            with self.assertRaises(RuntimeError):
                smoke.unlink_owned_segment(path, identity)
            self.assertTrue(path.is_symlink())

    def test_wrong_owner_or_file_type_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "segment"
            path.touch()
            with patch.object(smoke.os, "getuid", return_value=path.stat().st_uid + 1):
                with self.assertRaises(RuntimeError):
                    smoke.unlink_owned_segment(path, None)
            self.assertTrue(path.exists())
            with self.assertRaises(RuntimeError):
                smoke.segment_identity(Path(temporary))

    def test_preexisting_name_is_rejected_before_any_gpu_check(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(smoke, "runtime_configuration", return_value={}), \
                patch.object(smoke.os.path, "lexists", return_value=True), \
                patch.object(smoke.safety.LeaseSet, "acquire") as acquire, \
                patch.object(smoke.safety, "safety_snapshot") as snapshot:
            with self.assertRaisesRegex(RuntimeError, "unique shared-memory name"):
                smoke.run(Path(temporary) / "cell", Path("unused-build"))
            acquire.return_value.close.assert_called_once()
            snapshot.assert_not_called()


if __name__ == "__main__":
    unittest.main()
