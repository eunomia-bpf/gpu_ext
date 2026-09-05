#!/usr/bin/env python3
"""Offline entry point for the frozen stale-state experiment boundary."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

import coordinator
import protocol


BRIDGE_PREFLIGHT_SCHEMA = "stale-state-bridge-preflight-v1"


class _FastAdvancingClock:
    """Deterministic monotonic clock for a zero-wall-time contract preflight."""

    def __init__(self, start_ns: int = 10_000_000_000):
        self.now_ns = start_ns

    def clock_ns(self) -> int:
        self.now_ns += 1_000
        return self.now_ns

    def sleep(self, seconds: float) -> None:
        self.now_ns += max(1, int(seconds * 1.0e9))


def run_bridge_preflight(
    *,
    implementation: str | None = None,
    delay_ms: int | None = None,
    clock_factory: Callable[[], _FastAdvancingClock] = _FastAdvancingClock,
    bridge_factory: Callable[
        [Callable[[], int]], coordinator.Bridge
    ] = lambda clock_ns: coordinator.InMemoryContractBridge(clock_ns=clock_ns),
) -> dict[str, Any]:
    """Exercise the bridge ABI model without claiming live experiment evidence."""

    if (implementation is None) != (delay_ms is None):
        raise protocol.ValidationError(
            "bridge-preflight implementation and delay must be selected together"
        )
    if implementation is None:
        conditions = [
            (mode, delay)
            for mode in protocol.IMPLEMENTATIONS
            for delay in protocol.DELAYS_MS
        ]
    else:
        if implementation not in protocol.IMPLEMENTATIONS:
            raise protocol.ValidationError(
                f"unsupported bridge-preflight implementation: {implementation!r}"
            )
        if type(delay_ms) is not int or delay_ms not in protocol.DELAYS_MS:
            raise protocol.ValidationError(
                f"unsupported bridge-preflight delay: {delay_ms!r}"
            )
        conditions = [(implementation, delay_ms)]

    rows = []
    for ordinal, (mode, delay) in enumerate(conditions, 1):
        clock = clock_factory()
        bridge = bridge_factory(clock.clock_ns)
        if bridge.live or bridge.experiment_evidence:
            raise protocol.ValidationError(
                "bridge-preflight accepts only a non-live, non-evidence backend"
            )
        replay = coordinator.Coordinator(
            bridge, clock_ns=clock.clock_ns, sleep=clock.sleep
        ).replay(
            implementation=mode,
            generation=202609040000 + ordinal,
            delay_ms=delay,
        )
        if (
            replay.get("live_bridge") is not False
            or replay.get("experiment_evidence") is not False
            or replay.get("synthetic_source") is not True
        ):
            raise protocol.ValidationError(
                "bridge-preflight replay changed its non-live synthetic evidence boundary"
            )
        rows.append(replay)

    return {
        "schema": BRIDGE_PREFLIGHT_SCHEMA,
        "cpu_only": True,
        "live_bridge": any(row["live_bridge"] for row in rows),
        "experiment_evidence": any(row["experiment_evidence"] for row in rows),
        "synthetic_source": all(row["synthetic_source"] for row in rows),
        "condition_count": len(rows),
        "conditions": rows,
    }


def _emit(value: dict[str, Any], stream: Any | None = None) -> None:
    if stream is None:
        stream = sys.stdout
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    """Create one result atomically, while refusing to replace prior evidence."""

    path = protocol.lexical_absolute(path)
    if path.exists():
        raise protocol.ValidationError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            _emit(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise protocol.ValidationError(f"refusing to overwrite output: {path}")
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan and validate the stale-state study. Live execution is "
            "intentionally unavailable until the declared driver interface exists."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    dry_run = commands.add_parser(
        "dry-run", help="print a side-effect-free frozen execution plan"
    )
    dry_run.add_argument("stage", choices=("preflight", "full"))
    dry_run.add_argument("--output", required=True, type=Path)
    dry_run.add_argument("--preflight", type=Path)

    cpu = commands.add_parser(
        "cpu-preflight", help="exercise real fixed-delay timing without a GPU"
    )
    cpu.add_argument("--output", type=Path)
    cpu.add_argument("--samples-per-delay", type=int, default=3)

    bridge = commands.add_parser(
        "bridge-preflight",
        help="exercise the non-live in-memory driver bridge contract",
    )
    bridge.add_argument("--output", type=Path)
    bridge.add_argument("--implementation", choices=protocol.IMPLEMENTATIONS)
    bridge.add_argument("--delay-ms", type=int, choices=protocol.DELAYS_MS)

    analyze = commands.add_parser(
        "analyze", help="fail-closed validation and paired analysis of raw records"
    )
    analyze.add_argument("--input", required=True, type=Path)

    cell = commands.add_parser(
        "validate-cell", help="validate one raw cell against its matrix identity"
    )
    cell.add_argument("--input", required=True, type=Path)
    cell.add_argument("--block", required=True, type=int)
    cell.add_argument("--arm", required=True)

    live = commands.add_parser(
        "live", help="fail closed at the missing shared-snapshot interface"
    )
    live.add_argument("--output", required=True, type=Path)
    return parser


def _find_cell(block: int, arm: str) -> protocol.MatrixCell:
    matches = [
        cell
        for cell in protocol.matrix("full")
        if cell.block == block and cell.arm == arm
    ]
    if len(matches) != 1:
        raise protocol.ValidationError(
            f"no unique frozen cell for block={block}, arm={arm!r}"
        )
    return matches[0]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "dry-run":
            _emit(protocol.dry_run_plan(args.stage, args.output, args.preflight))
            return 0
        if args.command == "cpu-preflight":
            result = protocol.run_cpu_delay_preflight(
                samples_per_delay=args.samples_per_delay
            )
            if args.output is not None:
                _write_new_json(args.output, result)
            _emit(result)
            return 0
        if args.command == "bridge-preflight":
            result = run_bridge_preflight(
                implementation=args.implementation, delay_ms=args.delay_ms
            )
            if args.output is not None:
                _write_new_json(args.output, result)
            _emit(result)
            return 0
        if args.command == "analyze":
            _emit(protocol.validate_campaign(args.input))
            return 0
        if args.command == "validate-cell":
            _emit(protocol.validate_cell(args.input, _find_cell(args.block, args.arm)))
            return 0
        if args.command == "live":
            # Deliberately before path inspection, directory creation, leases,
            # process launch, device access, or any driver state change.
            raise protocol.ValidationError(f"live execution blocked: {protocol.LIVE_BLOCKER}")
        raise AssertionError(f"unhandled command: {args.command}")
    except protocol.ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())
