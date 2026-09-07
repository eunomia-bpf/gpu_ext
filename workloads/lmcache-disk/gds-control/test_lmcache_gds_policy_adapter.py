"""Focused development tests for the BpfDecider ioctl buffer-reuse path.

These tests run against a mocked command-82 driver only; they prove
allocation reuse, input/output overwrite semantics, and legacy/reuse decision
equivalence. Root measures the real driver separately.
"""

import errno
import os
import struct
import tempfile
import threading
import unittest
from unittest import mock

import lmcache_gds_policy_adapter as policy
from lmcache_gds_policy_adapter import (
    ACTION_DEFER,
    ACTION_RECOMPUTE,
    ACTION_SUBMIT_NOW,
    FLAG_DEMAND,
    FLAG_RECOMPUTABLE,
    FLAG_SAFE_TO_DEFER,
    FLAG_SPECULATIVE,
    HINT_LIVE_DEMAND,
    IOCTL_CMD,
    MAX_DEFER_NS,
    OP_READ,
    OP_WRITE,
    BpfDecider,
    Decision,
    GdsDecisionError,
    PolicyRequest,
    native_decide,
)

REUSE_ENV = "LMCACHE_GDS_IOCTL_REUSE"

# Input region of the 136-byte ABI: 4I + 9Q + 4I + Q + I = 116 bytes.
_INPUT = struct.Struct("<4I9Q4IQI")
_OUTPUT = struct.Struct("IIQI4xQI")
_ACTION_OFFSET = 96


def _request_from_buffer(buffer):
    values = _INPUT.unpack_from(buffer, 0)
    return PolicyRequest(
        op=values[1],
        flags=values[2],
        priority=values[3],
        nbytes=values[6],
        tenant_id=values[7],
        caller_hint=values[8],
        deadline_ns=values[9],
        slack_ns=values[10],
        estimated_transfer_ns=values[11],
        recompute_ns=values[12],
        queue_depth=values[13],
        hbm_pressure_permille=values[14],
    )


class ScriptedDriver:
    """Fake command-82 driver that writes fixed outputs into the buffer."""

    def __init__(self, outputs, status=0):
        self.outputs = list(outputs)
        self.status = status
        self.calls = 0
        self.seen = []
        self.buffers = []

    def __call__(self, fd, command, buffer):
        self.calls += 1
        assert command == IOCTL_CMD
        self.seen.append(bytes(buffer))
        self.buffers.append(buffer)
        action, priority, defer_ns, batch = self.outputs.pop(0)
        _OUTPUT.pack_into(
            buffer, _ACTION_OFFSET,
            action, priority, defer_ns, batch, 777, self.status,
        )


class NativeDriver:
    """Fake command-82 driver that answers with the module's native policy."""

    def __init__(self):
        self.calls = 0
        self.seen = []
        self.buffers = []

    def __call__(self, fd, command, buffer):
        self.calls += 1
        assert command == IOCTL_CMD
        self.seen.append(bytes(buffer))
        self.buffers.append(buffer)
        decision = native_decide(_request_from_buffer(buffer))
        _OUTPUT.pack_into(
            buffer, _ACTION_OFFSET,
            decision.action, decision.priority, decision.defer_ns,
            decision.batch_target, 777, 0,
        )


def _sequence():
    return [
        # demand read: SUBMIT_NOW
        (PolicyRequest(op=OP_READ, flags=FLAG_DEMAND, priority=3, nbytes=4096), 0),
        # recompute is cheaper and fits the slack: RECOMPUTE
        (PolicyRequest(
            op=OP_READ, flags=FLAG_RECOMPUTABLE | FLAG_SPECULATIVE,
            priority=1, nbytes=8192, slack_ns=5_000_000,
            estimated_transfer_ns=9_000_000, recompute_ns=4_000_000,
        ), 1),
        # speculative read under pressure: DEFER with clamped deferNs
        (PolicyRequest(
            op=OP_READ, flags=FLAG_SPECULATIVE | FLAG_SAFE_TO_DEFER,
            priority=2, nbytes=4096, slack_ns=7_000_000,
            hbm_pressure_permille=900,
        ), 2),
        # live-demand write with queue and slack: DEFER one feedback step
        (PolicyRequest(
            op=OP_WRITE, flags=FLAG_SAFE_TO_DEFER, priority=5, nbytes=16384,
            caller_hint=HINT_LIVE_DEMAND, slack_ns=3_000_000, queue_depth=2,
        ), 3),
        # write under pressure: DEFER, clamped priority and batch target
        (PolicyRequest(
            op=OP_WRITE, flags=FLAG_SAFE_TO_DEFER, priority=8, nbytes=32768,
            slack_ns=MAX_DEFER_NS + 5, queue_depth=130,
            hbm_pressure_permille=650,
        ), 4),
        # plain write: SUBMIT_NOW with clamped priority
        (PolicyRequest(op=OP_WRITE, priority=9, nbytes=1024), 5),
    ]


class BpfDeciderReuseTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop(REUSE_ENV, None)

    def test_reuse_overwrites_inputs_and_outputs_each_request(self):
        requests = [
            PolicyRequest(op=OP_READ, flags=FLAG_DEMAND, priority=1, nbytes=4096),
            PolicyRequest(
                op=OP_WRITE, flags=FLAG_SAFE_TO_DEFER, priority=4, nbytes=9999,
                queue_depth=3, hbm_pressure_permille=610,
            ),
        ]
        driver = ScriptedDriver([
            (ACTION_DEFER, 1, 1_111_111, 4),
            (ACTION_RECOMPUTE, 4, 2_222_222, 7),
        ])
        with mock.patch.dict(os.environ, {REUSE_ENV: "1"}):
            decider = BpfDecider(ioctl_func=driver, uvm_fd=3)
        first = decider.decide(requests[0], 100)
        second = decider.decide(requests[1], 101)
        self.assertEqual(first, Decision(ACTION_DEFER, 1_111_111, 1, 4))
        self.assertEqual(second, Decision(ACTION_RECOMPUTE, 2_222_222, 4, 7))
        self.assertEqual(driver.calls, 2)
        # One shared buffer object served both ioctl calls.
        self.assertIs(driver.buffers[0], decider._shared)
        self.assertIs(driver.buffers[1], decider._shared)
        # Each request repacked every input and re-zeroed every output, so
        # the driver sees exactly what a fresh legacy pack would have held.
        self.assertEqual(
            driver.seen[0], requests[0].pack(request_id=100, object_id=100)
        )
        self.assertEqual(
            driver.seen[1], requests[1].pack(request_id=101, object_id=101)
        )
        # The decider parsed the outputs the driver wrote into the same buffer.
        written = _OUTPUT.unpack_from(decider._shared, _ACTION_OFFSET)
        self.assertEqual(written[0], ACTION_RECOMPUTE)
        self.assertEqual(written[2], 2_222_222)
        self.assertEqual(written[4], 777)

    def test_legacy_and_reuse_yield_same_decisions(self):
        sequence = _sequence()

        def run(reuse):
            env = {REUSE_ENV: "1"} if reuse else {}
            with mock.patch.dict(os.environ, env):
                driver = NativeDriver()
                decider = BpfDecider(ioctl_func=driver, uvm_fd=3)
            decisions = [
                decider.decide(request, request_id)
                for request, request_id in sequence
            ]
            return decisions, driver

        legacy_decisions, legacy_driver = run(False)
        reuse_decisions, reuse_driver = run(True)
        self.assertEqual(
            legacy_decisions,
            [
                Decision(ACTION_SUBMIT_NOW, 0, 3, 1),
                Decision(ACTION_RECOMPUTE, 0, 1, 1),
                Decision(ACTION_DEFER, 7_000_000, 2, 1),
                Decision(ACTION_DEFER, 1_000_000, 5, 1),
                Decision(ACTION_DEFER, MAX_DEFER_NS, 7, 64),
                Decision(ACTION_SUBMIT_NOW, 0, 7, 1),
            ],
        )
        self.assertEqual(reuse_decisions, legacy_decisions)
        # Byte-identical input buffers reached the (mock) driver on both paths.
        self.assertEqual(legacy_driver.seen, reuse_driver.seen)
        self.assertEqual(legacy_driver.calls, reuse_driver.calls)
        self.assertEqual(len(reuse_driver.seen), len(sequence))

    def test_reuse_is_opt_in_and_default_is_legacy(self):
        driver = ScriptedDriver([(ACTION_SUBMIT_NOW, 0, 0, 1)] * 2)
        legacy = BpfDecider(ioctl_func=driver, uvm_fd=3)
        legacy.decide(PolicyRequest(), 0)
        legacy.decide(PolicyRequest(), 1)
        self.assertIsNone(legacy._shared)
        # Legacy keeps allocating one fresh buffer per decision.
        self.assertIsNot(driver.buffers[0], driver.buffers[1])

        reuse_driver = ScriptedDriver([(ACTION_SUBMIT_NOW, 0, 0, 1)] * 2)
        with mock.patch.dict(os.environ, {REUSE_ENV: "1"}):
            reuse = BpfDecider(ioctl_func=reuse_driver, uvm_fd=3)
        self.assertIsNotNone(reuse._shared)
        reuse.decide(PolicyRequest(), 0)
        reuse.decide(PolicyRequest(), 1)
        self.assertIs(reuse_driver.buffers[0], reuse_driver.buffers[1])

    def test_reuse_error_handling_matches_legacy(self):
        for reuse in (False, True):
            env = {REUSE_ENV: "1"} if reuse else {}
            with mock.patch.dict(os.environ, env):
                driver = ScriptedDriver(
                    [(ACTION_SUBMIT_NOW, 0, 0, 1)], status=3
                )
                decider = BpfDecider(ioctl_func=driver, uvm_fd=3)
                with self.assertRaises(GdsDecisionError) as raised:
                    decider.decide(PolicyRequest(), 0)
                self.assertIn("rmStatus=3", str(raised.exception))

                bad_action = ScriptedDriver([(99, 0, 0, 1)])
                decider = BpfDecider(ioctl_func=bad_action, uvm_fd=3)
                with self.assertRaises(GdsDecisionError) as raised:
                    decider.decide(PolicyRequest(), 0)
                self.assertIn("unknown action 99", str(raised.exception))

    def test_reuse_propagates_ioctl_oserror(self):
        def failing(fd, command, buffer):
            del fd, buffer
            raise OSError(5, "Input/output error", f"UVM ioctl {command}")

        with mock.patch.dict(os.environ, {REUSE_ENV: "1"}):
            decider = BpfDecider(ioctl_func=failing, uvm_fd=3)
        with self.assertRaises(OSError) as raised:
            decider.decide(PolicyRequest(), 0)
        self.assertEqual(raised.exception.errno, 5)

    def test_reuse_default_ioctl_reports_errno(self):
        # No injected ioctl: exercises the bound-callable branch against a
        # real fd where command 82 must fail (regular file, no driver).
        with tempfile.NamedTemporaryFile() as handle:
            with mock.patch.dict(os.environ, {REUSE_ENV: "1"}):
                decider = BpfDecider(uvm_fd=handle.fileno())
            self.assertIsNotNone(decider._shared_ioctl)
            with self.assertRaises(OSError) as raised:
                decider.decide(PolicyRequest(), 0)
            self.assertIn(
                raised.exception.errno, (errno.ENOTTY, errno.EINVAL)
            )

    def test_reuse_concurrent_decisions_are_consistent(self):
        driver = NativeDriver()
        with mock.patch.dict(os.environ, {REUSE_ENV: "1"}):
            decider = BpfDecider(ioctl_func=driver, uvm_fd=3)

        def worker(thread_id):
            for request_id in range(25):
                request = PolicyRequest(
                    op=OP_READ, flags=FLAG_DEMAND,
                    priority=thread_id, nbytes=4096,
                )
                decision = decider.decide(request, thread_id * 100 + request_id)
                self.assertEqual(
                    decision, Decision(ACTION_SUBMIT_NOW, 0, thread_id, 1)
                )

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(driver.calls, 4 * 25)


if __name__ == "__main__":
    unittest.main()
