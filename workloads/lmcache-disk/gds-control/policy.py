"""GDS control policy: decide the action for one I/O request.

Wire layout of a request (little-endian, 96 bytes total):

    header  4 x u32: version, flags, priority, delay_ms
    payload 9 x u64: seq, op, kind, deadline_ns, slack_ns, pressure,
                     recompute_cost_ns, io_cost_ns, batch_hint
    trailer 2 x u32: epoch, reserved

Decision precedence:

    1. demand read -> SUBMIT
    2. recompute cheaper than the I/O and within slack -> RECOMPUTE
    3. safe speculative read with pressure >= 800 -> DEFER
    4. safe speculative write with pressure >= 600 -> DEFER or BATCH
    5. otherwise -> SUBMIT

Outputs are clamped: priority 0..7, defer window 0..10ms, batch 1..64.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List
import heapq
import struct

MS_TO_NS = 1_000_000

PRIORITY_MIN = 0
PRIORITY_MAX = 7
DEFER_MIN_NS = 0
DEFER_MAX_NS = 10 * MS_TO_NS
BATCH_MIN = 1
BATCH_MAX = 64

READ_DEFER_PRESSURE = 800
WRITE_DEFER_PRESSURE = 600

FETCH_DIVISOR = 1000
FETCH_LEVELS = (0, 25, 50, 75, 100)

FLAG_SAFE = 0x1

VERSION = 1


class Action(IntEnum):
    SUBMIT = 0
    DEFER = 1
    BATCH = 2
    RECOMPUTE = 3


class Op(IntEnum):
    READ = 0
    WRITE = 1


class Kind(IntEnum):
    DEMAND = 0
    SPECULATIVE = 1


@dataclass(frozen=True)
class Request:
    seq: int = 0
    op: Op = Op.READ
    kind: Kind = Kind.DEMAND
    deadline_ns: int = 0
    slack_ns: int = 0
    pressure: int = 0
    recompute_cost_ns: int = 0
    io_cost_ns: int = 0
    batch_hint: int = 1
    priority: int = 5
    delay_ms: int = 0
    flags: int = 0
    version: int = VERSION
    epoch: int = 0
    reserved: int = 0


@dataclass(frozen=True)
class Reply:
    action: Action
    output_priority: int
    defer_ns: int
    batch_target: int


_HEADER = struct.Struct("<4I")   # version, flags, priority, delay_ms
_PAYLOAD = struct.Struct("<9Q")  # seq, kind fields as listed above
_TRAILER = struct.Struct("<2I")  # epoch, reserved
REQUEST_SIZE = _HEADER.size + _PAYLOAD.size + _TRAILER.size


def _clamp(value: int, low: int, high: int) -> int:
    if value < low:
        return low
    if value > high:
        return high
    return value


def pack_request(req: Request) -> bytes:
    return (
        _HEADER.pack(
            req.version,
            req.flags,
            req.priority,
            req.delay_ms,
        )
        + _PAYLOAD.pack(
            req.seq,
            int(req.op),
            int(req.kind),
            req.deadline_ns,
            req.slack_ns,
            req.pressure,
            req.recompute_cost_ns,
            req.io_cost_ns,
            req.batch_hint,
        )
        + _TRAILER.pack(req.epoch, req.reserved)
    )


def unpack_request(data: bytes) -> Request:
    if len(data) != REQUEST_SIZE:
        raise ValueError(
            f"request must be exactly {REQUEST_SIZE} bytes, got {len(data)}"
        )
    version, flags, priority, delay_ms = _HEADER.unpack_from(data, 0)
    if version != VERSION:
        raise ValueError(f"unsupported request version {version}")
    (
        seq,
        op,
        kind,
        deadline_ns,
        slack_ns,
        pressure,
        recompute_cost_ns,
        io_cost_ns,
        batch_hint,
    ) = _PAYLOAD.unpack_from(data, _HEADER.size)
    epoch, reserved = _TRAILER.unpack_from(data, _HEADER.size + _PAYLOAD.size)
    return Request(
        seq=seq,
        op=Op(op),
        kind=Kind(kind),
        deadline_ns=deadline_ns,
        slack_ns=slack_ns,
        pressure=pressure,
        recompute_cost_ns=recompute_cost_ns,
        io_cost_ns=io_cost_ns,
        batch_hint=batch_hint,
        priority=priority,
        delay_ms=delay_ms,
        flags=flags,
        version=version,
        epoch=epoch,
        reserved=reserved,
    )


def decide(req: Request) -> Reply:
    priority = _clamp(req.priority, PRIORITY_MIN, PRIORITY_MAX)
    defer_ns = _clamp(req.delay_ms * MS_TO_NS, DEFER_MIN_NS, DEFER_MAX_NS)
    batch = _clamp(req.batch_hint, BATCH_MIN, BATCH_MAX)

    if req.op == Op.READ and req.kind == Kind.DEMAND:
        return Reply(Action.SUBMIT, priority, 0, BATCH_MIN)

    if req.recompute_cost_ns < req.io_cost_ns and req.recompute_cost_ns <= req.slack_ns:
        return Reply(Action.RECOMPUTE, priority, 0, BATCH_MIN)

    if req.kind == Kind.SPECULATIVE and req.flags & FLAG_SAFE:
        if req.op == Op.READ and req.pressure >= READ_DEFER_PRESSURE:
            return Reply(Action.DEFER, priority, defer_ns, BATCH_MIN)
        if req.op == Op.WRITE and req.pressure >= WRITE_DEFER_PRESSURE:
            if batch > BATCH_MIN:
                return Reply(Action.BATCH, priority, defer_ns, batch)
            return Reply(Action.DEFER, priority, defer_ns, BATCH_MIN)

    return Reply(Action.SUBMIT, priority, 0, BATCH_MIN)


def fetch_level(pressure: int) -> int:
    pct = _clamp(pressure * 100 // FETCH_DIVISOR, 0, 100)
    if pct >= 100:
        return 100
    if pct >= 75:
        return 75
    if pct >= 50:
        return 50
    if pct >= 25:
        return 25
    return 0


class DeadlineQueue:
    """Stable queue ordered by deadline, then by push order."""

    def __init__(self) -> None:
        self._entries: List[tuple] = []
        self._seq = 0

    def push(self, item, deadline_ns: int) -> None:
        heapq.heappush(self._entries, (deadline_ns, self._seq, item))
        self._seq += 1

    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def peek_deadline(self) -> int:
        if not self._entries:
            raise IndexError("peek from empty deadline queue")
        return self._entries[0][0]

    def peek(self):
        if not self._entries:
            raise IndexError("peek from empty deadline queue")
        return self._entries[0][2]

    def pop_due(self, now_ns: int) -> List:
        due = []
        while self._entries and self._entries[0][0] <= now_ns:
            _, _, item = heapq.heappop(self._entries)
            due.append(item)
        return due

    def drain(self) -> List:
        out = []
        while self._entries:
            out.append(heapq.heappop(self._entries)[2])
        return out
