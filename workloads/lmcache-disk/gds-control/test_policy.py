"""Standard-library unit tests for the GDS control policy."""

import struct
import unittest

from policy import (
    BATCH_MAX,
    BATCH_MIN,
    DEFER_MAX_NS,
    FLAG_SAFE,
    MS_TO_NS,
    PRIORITY_MAX,
    PRIORITY_MIN,
    REQUEST_SIZE,
    VERSION,
    Action,
    DeadlineQueue,
    Kind,
    Op,
    Reply,
    Request,
    decide,
    fetch_level,
    pack_request,
    unpack_request,
)


def make(**overrides):
    defaults = dict(
        seq=0,
        op=Op.READ,
        kind=Kind.SPECULATIVE,
        deadline_ns=100_000,
        slack_ns=1_000,
        pressure=0,
        recompute_cost_ns=1 << 40,
        io_cost_ns=1_000_000,
        batch_hint=1,
        priority=5,
        delay_ms=2,
        flags=FLAG_SAFE,
        version=VERSION,
        epoch=1,
        reserved=0,
    )
    defaults.update(overrides)
    return Request(**defaults)


class DecideBranchTests(unittest.TestCase):
    def test_demand_read_submits_despite_pressure_and_recompute(self):
        req = make(
            kind=Kind.DEMAND,
            op=Op.READ,
            pressure=1000,
            recompute_cost_ns=1,
            io_cost_ns=100,
            slack_ns=100,
        )
        self.assertEqual(decide(req), Reply(Action.SUBMIT, 5, 0, 1))

    def test_recompute_when_cheaper_and_fits_slack(self):
        req = make(
            kind=Kind.DEMAND,
            op=Op.WRITE,
            recompute_cost_ns=5_000,
            io_cost_ns=10_000,
            slack_ns=5_000,
        )
        self.assertEqual(decide(req), Reply(Action.RECOMPUTE, 5, 0, 1))

    def test_speculative_recompute_wins_before_defer(self):
        req = make(
            kind=Kind.SPECULATIVE,
            op=Op.READ,
            pressure=900,
            recompute_cost_ns=5_000,
            io_cost_ns=10_000,
            slack_ns=5_000,
        )
        self.assertEqual(decide(req).action, Action.RECOMPUTE)

    def test_recompute_requires_strictly_cheaper(self):
        req = make(
            recompute_cost_ns=10_000,
            io_cost_ns=10_000,
            slack_ns=10_000,
        )
        self.assertEqual(decide(req).action, Action.SUBMIT)

    def test_recompute_requires_slack(self):
        req = make(
            recompute_cost_ns=5_001,
            io_cost_ns=10_000,
            slack_ns=5_000,
        )
        self.assertEqual(decide(req).action, Action.SUBMIT)

    def test_safe_speculative_read_defers_at_pressure_800(self):
        req = make(op=Op.READ, pressure=800, priority=3, delay_ms=4)
        self.assertEqual(decide(req), Reply(Action.DEFER, 3, 4 * MS_TO_NS, 1))

    def test_safe_speculative_read_submits_below_800(self):
        req = make(op=Op.READ, pressure=799)
        self.assertEqual(decide(req).action, Action.SUBMIT)

    def test_unsafe_speculative_read_submits_at_high_pressure(self):
        req = make(op=Op.READ, pressure=1000, flags=0)
        self.assertEqual(decide(req).action, Action.SUBMIT)

    def test_safe_speculative_write_batches_at_pressure_600(self):
        req = make(
            op=Op.WRITE,
            kind=Kind.SPECULATIVE,
            pressure=600,
            batch_hint=8,
        )
        self.assertEqual(
            decide(req), Reply(Action.BATCH, 5, 2 * MS_TO_NS, 8)
        )

    def test_safe_speculative_write_defers_with_batch_one(self):
        req = make(op=Op.WRITE, pressure=600, batch_hint=1)
        self.assertEqual(decide(req).action, Action.DEFER)

    def test_safe_speculative_write_submits_below_600(self):
        req = make(op=Op.WRITE, pressure=599)
        self.assertEqual(decide(req).action, Action.SUBMIT)

    def test_safe_speculative_write_submits_when_unsafe(self):
        req = make(op=Op.WRITE, pressure=1000, flags=0)
        self.assertEqual(decide(req).action, Action.SUBMIT)

    def test_submit_carries_zero_defer_and_unit_batch(self):
        req = make(kind=Kind.DEMAND, op=Op.READ)
        reply = decide(req)
        self.assertEqual((reply.defer_ns, reply.batch_target), (0, 1))

    def test_recompute_carries_zero_defer_and_unit_batch(self):
        req = make(op=Op.WRITE, recompute_cost_ns=1, io_cost_ns=2, slack_ns=2)
        reply = decide(req)
        self.assertEqual((reply.defer_ns, reply.batch_target), (0, 1))


class ClampTests(unittest.TestCase):
    def test_priority_clamp_high(self):
        req = make(kind=Kind.DEMAND, priority=99)
        self.assertEqual(decide(req).output_priority, PRIORITY_MAX)

    def test_priority_clamp_low(self):
        req = make(kind=Kind.DEMAND, priority=-3)
        self.assertEqual(decide(req).output_priority, PRIORITY_MIN)

    def test_priority_inside_range_unchanged(self):
        req = make(kind=Kind.DEMAND, priority=6)
        self.assertEqual(decide(req).output_priority, 6)

    def test_defer_clamp_over_max(self):
        req = make(op=Op.READ, pressure=900, delay_ms=99)
        self.assertEqual(decide(req).defer_ns, DEFER_MAX_NS)

    def test_defer_clamp_zero(self):
        req = make(op=Op.READ, pressure=900, delay_ms=0)
        self.assertEqual(decide(req).defer_ns, 0)

    def test_defer_inside_range_unchanged(self):
        req = make(op=Op.READ, pressure=900, delay_ms=7)
        self.assertEqual(decide(req).defer_ns, 7 * MS_TO_NS)

    def test_batch_clamp_over_max(self):
        req = make(op=Op.WRITE, pressure=700, batch_hint=999)
        self.assertEqual(decide(req), Reply(Action.BATCH, 5, 2_000_000, BATCH_MAX))

    def test_batch_clamp_below_min_defers(self):
        req = make(op=Op.WRITE, pressure=700, batch_hint=0)
        self.assertEqual(decide(req), Reply(Action.DEFER, 5, 2_000_000, BATCH_MIN))


class WireTests(unittest.TestCase):
    def test_packed_size_is_96_bytes(self):
        wire = pack_request(make())
        self.assertEqual(len(wire), REQUEST_SIZE)
        self.assertEqual(REQUEST_SIZE, 4 * 4 + 9 * 8 + 2 * 4)

    def test_round_trip(self):
        req = make(
            seq=7,
            op=Op.WRITE,
            kind=Kind.DEMAND,
            deadline_ns=123_456,
            slack_ns=1_000,
            pressure=650,
            recompute_cost_ns=9_876,
            io_cost_ns=54_321,
            batch_hint=17,
            priority=2,
            delay_ms=9,
            flags=FLAG_SAFE,
            version=VERSION,
            epoch=11,
            reserved=13,
        )
        out = unpack_request(pack_request(req))
        self.assertEqual(out, req)
        self.assertIsInstance(out.op, Op)
        self.assertIsInstance(out.kind, Kind)

    def test_header_is_little_endian_u32_block(self):
        req = make(priority=7, delay_ms=10)
        header = pack_request(req)[:16]
        self.assertEqual(
            header,
            struct.pack("<4I", VERSION, FLAG_SAFE, 7, 10),
        )

    def test_unpack_rejects_short_buffer(self):
        data = pack_request(make())[:-1]
        with self.assertRaises(ValueError):
            unpack_request(data)

    def test_unpack_rejects_long_buffer(self):
        data = pack_request(make()) + b"\x00"
        with self.assertRaises(ValueError):
            unpack_request(data)

    def test_unpack_rejects_unknown_version(self):
        data = bytearray(pack_request(make()))
        data[0] = 2
        with self.assertRaises(ValueError):
            unpack_request(bytes(data))


class FetchLevelTests(unittest.TestCase):
    def test_boundaries(self):
        table = [
            (-10, 0),
            (0, 0),
            (249, 0),
            (250, 25),
            (499, 25),
            (500, 50),
            (749, 50),
            (750, 75),
            (999, 75),
            (1000, 100),
            (5000, 100),
        ]
        for pressure, expected in table:
            with self.subTest(pressure=pressure):
                self.assertEqual(fetch_level(pressure), expected)


class DeadlineQueueTests(unittest.TestCase):
    def test_equal_deadlines_pop_in_push_order(self):
        queue = DeadlineQueue()
        for name in ("a", "b", "c", "d"):
            queue.push(name, 100)
        self.assertEqual(queue.pop_due(100), ["a", "b", "c", "d"])

    def test_interleaved_pushes_keep_stability(self):
        queue = DeadlineQueue()
        queue.push("late", 50)
        queue.push("first", 10)
        queue.push("second", 10)
        queue.push("middle", 30)
        queue.push("third", 10)
        self.assertEqual(queue.pop_due(10), ["first", "second", "third"])
        self.assertEqual(queue.pop_due(30), ["middle"])
        self.assertEqual(queue.pop_due(50), ["late"])

    def test_drain_orders_by_deadline_then_push(self):
        queue = DeadlineQueue()
        queue.push("d", 5)
        queue.push("a", 10)
        queue.push("b", 10)
        queue.push("z", 3)
        self.assertEqual(queue.drain(), ["z", "d", "a", "b"])
        self.assertEqual(len(queue), 0)
        self.assertEqual(queue.drain(), [])

    def test_future_items_stay_queued(self):
        queue = DeadlineQueue()
        queue.push("now", 10)
        queue.push("later", 20)
        due = queue.pop_due(15)
        self.assertEqual(due, ["now"])
        self.assertEqual(queue.peek(), "later")
        self.assertEqual(queue.peek_deadline(), 20)
        self.assertEqual(len(queue), 1)

    def test_pop_due_boundary_at_deadline(self):
        queue = DeadlineQueue()
        queue.push("edge", 42)
        self.assertEqual(queue.pop_due(41), [])
        self.assertEqual(queue.pop_due(42), ["edge"])

    def test_stability_holds_after_interleaved_pops(self):
        queue = DeadlineQueue()
        for round_no in range(3):
            for name in ("a", "b", "c"):
                queue.push(f"{name}{round_no}", 10)
            queue.push(f"early{round_no}", 5)
            self.assertEqual(
                queue.pop_due(10),
                [f"early{round_no}", f"a{round_no}", f"b{round_no}", f"c{round_no}"],
            )

    def test_peek_from_empty_raises(self):
        queue = DeadlineQueue()
        with self.assertRaises(IndexError):
            queue.peek()
        with self.assertRaises(IndexError):
            queue.peek_deadline()

    def test_bool_truthiness(self):
        queue = DeadlineQueue()
        self.assertFalse(queue)
        queue.push("x", 0)
        self.assertTrue(queue)


if __name__ == "__main__":
    unittest.main()
