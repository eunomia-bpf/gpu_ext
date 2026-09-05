#!/usr/bin/env python3
"""Coordinator and test double for the stale-state driver bridge v1.

The in-memory bridge models the proc ABI for CPU tests.  Its output is never
live experiment evidence.  The proc bridge performs only versioned control
and status I/O; workload ownership and experiment validation remain the
responsibility of the future live runner.
"""

from __future__ import annotations

import errno
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import protocol


PROC_PATH = Path("/proc/uvm_stale_state_v1")
ABI_VERSION = 1
PHASE_IDS = {"dense": 1, "sparse": 2}
MODES = {"off", "native", "bpf"}
UINT64_MAX = (1 << 64) - 1
INT32_MAX = (1 << 31) - 1

COUNTER_FIELDS = (
    "snapshot_updates",
    "snapshot_rejections",
    "callback_invocations",
    "snapshot_read_attempts",
    "snapshot_read_successes",
    "missing_snapshot_decisions",
    "invalid_snapshot_decisions",
    "native_callback_invocations",
    "bpf_callback_invocations",
    "decision_requests",
    "decisions",
    "decision_records",
    "effect_requests",
    "effect_records",
    "dense_prefetch_decisions",
    "discarded_prefetch_decisions",
    "request_errors",
    "effect_errors",
    "selected_diagnostics",
    "finished_diagnostics",
)

STATUS_FIELDS = (
    "abi_version",
    "mode",
    "generation",
    "snapshot_present",
    "snapshot_sequence",
    "snapshot_phase",
    "source_mono_ns",
    "published_mono_ns",
    *COUNTER_FIELDS,
    "active_callbacks",
)


@dataclass(frozen=True)
class BridgeStatus:
    abi_version: int
    mode: str
    generation: int
    snapshot_present: int
    snapshot_sequence: int
    snapshot_phase: int
    source_mono_ns: int
    published_mono_ns: int
    snapshot_updates: int
    snapshot_rejections: int
    callback_invocations: int
    snapshot_read_attempts: int
    snapshot_read_successes: int
    missing_snapshot_decisions: int
    invalid_snapshot_decisions: int
    native_callback_invocations: int
    bpf_callback_invocations: int
    decision_requests: int
    decisions: int
    decision_records: int
    effect_requests: int
    effect_records: int
    dense_prefetch_decisions: int
    discarded_prefetch_decisions: int
    request_errors: int
    effect_errors: int
    selected_diagnostics: int
    finished_diagnostics: int
    active_callbacks: int

    def as_record(self) -> dict[str, int | str]:
        return asdict(self)


def parse_status(text: str) -> BridgeStatus:
    """Parse one complete, exact `/proc/uvm_stale_state_v1` status record."""

    values: dict[str, str] = {}
    for token in text.split():
        if token.count("=") != 1:
            raise protocol.ValidationError(f"malformed bridge status token: {token!r}")
        key, value = token.split("=", 1)
        if not key or not value:
            raise protocol.ValidationError(f"malformed bridge status token: {token!r}")
        if key in values:
            raise protocol.ValidationError(f"duplicate bridge status field: {key}")
        values[key] = value

    expected = set(STATUS_FIELDS)
    missing = sorted(expected - values.keys())
    extra = sorted(values.keys() - expected)
    if missing or extra:
        raise protocol.ValidationError(
            f"bridge status schema mismatch: missing={missing}, extra={extra}"
        )

    mode = values["mode"]
    if mode not in MODES:
        raise protocol.ValidationError(f"invalid bridge mode: {mode!r}")

    parsed: dict[str, int | str] = {"mode": mode}
    for field in STATUS_FIELDS:
        if field == "mode":
            continue
        raw = values[field]
        try:
            number = int(raw, 10)
        except ValueError as exc:
            raise protocol.ValidationError(
                f"non-integer bridge status field {field}: {raw!r}"
            ) from exc
        if number < 0:
            raise protocol.ValidationError(
                f"negative bridge status field {field}: {number}"
            )
        maximum = INT32_MAX if field == "active_callbacks" else UINT64_MAX
        if number > maximum:
            raise protocol.ValidationError(
                f"out-of-range bridge status field {field}: {number}"
            )
        parsed[field] = number

    if parsed["abi_version"] != ABI_VERSION:
        raise protocol.ValidationError(
            f"unsupported bridge ABI version: {parsed['abi_version']}"
        )
    if parsed["snapshot_present"] not in {0, 1}:
        raise protocol.ValidationError("snapshot_present is not boolean")
    if parsed["snapshot_phase"] not in {0, *PHASE_IDS.values()}:
        raise protocol.ValidationError("snapshot_phase is invalid")

    snapshot_fields = (
        parsed["snapshot_sequence"],
        parsed["snapshot_phase"],
        parsed["source_mono_ns"],
        parsed["published_mono_ns"],
    )
    if parsed["snapshot_present"] == 0 and any(snapshot_fields):
        raise protocol.ValidationError("absent snapshot has nonzero fields")
    if parsed["snapshot_present"] == 1:
        if any(value == 0 for value in snapshot_fields):
            raise protocol.ValidationError("present snapshot has a zero field")
        if parsed["published_mono_ns"] < parsed["source_mono_ns"]:
            raise protocol.ValidationError("snapshot publication predates its source")

    return BridgeStatus(**parsed)


class Bridge(Protocol):
    live: bool
    experiment_evidence: bool

    def configure(self, mode: str, generation: int) -> None: ...

    def publish(
        self, generation: int, sequence: int, phase: int, source_mono_ns: int
    ) -> None: ...

    def disable(self, generation: int) -> None: ...

    def status(self) -> BridgeStatus: ...


class ProcBridge:
    """Thin client for the root-only version-1 proc control endpoint."""

    live = True
    experiment_evidence = False

    def __init__(self, path: Path = PROC_PATH):
        self.path = Path(path)

    def _write(self, command: str) -> None:
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CLOEXEC)
        try:
            payload = command.encode("ascii")
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError(errno.EIO, "short write to stale-state proc endpoint")
        finally:
            os.close(descriptor)

    def configure(self, mode: str, generation: int) -> None:
        self._write(f"configure {mode} {generation}\n")

    def publish(
        self, generation: int, sequence: int, phase: int, source_mono_ns: int
    ) -> None:
        self._write(f"publish {generation} {sequence} {phase} {source_mono_ns}\n")

    def disable(self, generation: int) -> None:
        self._write(f"disable {generation}\n")

    def status(self) -> BridgeStatus:
        with self.path.open("r", encoding="ascii") as stream:
            return parse_status(stream.read())


class InMemoryContractBridge:
    """Deterministic CPU model of the driver control-plane contract."""

    live = False
    experiment_evidence = False

    def __init__(self, *, clock_ns: Callable[[], int] = time.monotonic_ns):
        self._clock_ns = clock_ns
        self._mode = "off"
        self._generation = 0
        self._snapshot: tuple[int, int, int, int] | None = None
        self._counters = {field: 0 for field in COUNTER_FIELDS}
        self._active_callbacks = 0

    @staticmethod
    def _error(code: int, message: str) -> None:
        raise OSError(code, message)

    def configure(self, mode: str, generation: int) -> None:
        if mode not in {"native", "bpf"} or not _valid_u64(generation):
            self._error(errno.EINVAL, "invalid stale-state configuration")
        self._mode = "off"
        self._active_callbacks = 0
        self._snapshot = None
        self._counters = {field: 0 for field in COUNTER_FIELDS}
        self._generation = generation
        self._mode = mode

    def publish(
        self, generation: int, sequence: int, phase: int, source_mono_ns: int
    ) -> None:
        if (
            self._mode == "off"
            or not _valid_u64(generation)
            or generation != self._generation
        ):
            self._reject(errno.ESTALE, "stale bridge generation")
        if (
            not _valid_u64(sequence)
            or type(phase) is not int
            or phase not in PHASE_IDS.values()
            or not _valid_u64(source_mono_ns)
        ):
            self._reject(errno.EINVAL, "invalid snapshot fields")

        if self._snapshot is None:
            follows = sequence == 1
        else:
            current_sequence, _, current_source_ns, _ = self._snapshot
            follows = (
                current_sequence != UINT64_MAX
                and sequence == current_sequence + 1
                and source_mono_ns > current_source_ns
            )
        if not follows:
            self._reject(errno.ERANGE, "snapshot does not follow current publication")

        published_mono_ns = self._clock_ns()
        if source_mono_ns > published_mono_ns:
            self._reject(errno.ERANGE, "snapshot source is in the future")

        self._snapshot = (sequence, phase, source_mono_ns, published_mono_ns)
        self._counters["snapshot_updates"] += 1

    def _reject(self, code: int, message: str) -> None:
        self._counters["snapshot_rejections"] += 1
        self._error(code, message)

    def disable(self, generation: int) -> None:
        if not _valid_u64(generation) or generation != self._generation:
            self._error(errno.ESTALE, "stale bridge generation")
        self._mode = "off"
        self._active_callbacks = 0
        self._snapshot = None

    def status(self) -> BridgeStatus:
        if self._snapshot is None:
            sequence = phase = source_mono_ns = published_mono_ns = 0
            present = 0
        else:
            sequence, phase, source_mono_ns, published_mono_ns = self._snapshot
            present = 1
        return BridgeStatus(
            abi_version=ABI_VERSION,
            mode=self._mode,
            generation=self._generation,
            snapshot_present=present,
            snapshot_sequence=sequence,
            snapshot_phase=phase,
            source_mono_ns=source_mono_ns,
            published_mono_ns=published_mono_ns,
            **self._counters,
            active_callbacks=self._active_callbacks,
        )


class Coordinator:
    """Replay the frozen seven-publication timeline against one bridge mode."""

    def __init__(
        self,
        bridge: Bridge,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.bridge = bridge
        self._clock_ns = clock_ns
        self._sleep = sleep

    def _wait_until(self, target_ns: int) -> int:
        now_ns = self._clock_ns()
        while now_ns < target_ns:
            self._sleep((target_ns - now_ns) / 1.0e9)
            next_ns = self._clock_ns()
            if next_ns <= now_ns:
                raise protocol.ValidationError("coordinator monotonic clock did not advance")
            now_ns = next_ns
        return now_ns

    def replay(
        self,
        *,
        implementation: str,
        generation: int,
        delay_ms: int,
        epoch_mono_ns: int | None = None,
    ) -> dict[str, Any]:
        if implementation not in protocol.IMPLEMENTATIONS:
            raise protocol.ValidationError(
                f"unsupported coordinator implementation: {implementation!r}"
            )
        if delay_ms not in protocol.DELAYS_MS or type(delay_ms) is not int:
            raise protocol.ValidationError(f"unsupported coordinator delay: {delay_ms!r}")
        if not _valid_u64(generation):
            raise protocol.ValidationError(f"invalid coordinator generation: {generation!r}")

        epoch_ns = self._clock_ns() if epoch_mono_ns is None else epoch_mono_ns
        if not _valid_u64(epoch_ns):
            raise protocol.ValidationError(f"invalid coordinator epoch: {epoch_ns!r}")

        delay_ns = delay_ms * 1_000_000
        publications: list[dict[str, Any]] = []
        failure: Exception | None = None
        disabled_status: BridgeStatus | None = None
        result: dict[str, Any] | None = None
        successful_updates = 0
        initial_status = self.bridge.status()
        _require_idle(initial_status)
        try:
            self.bridge.configure(implementation, generation)
            configured_status = self.bridge.status()
            _require_configured(configured_status, implementation, generation)

            for sequence in range(1, protocol.MEASURED_PHASES + 2):
                expected = protocol.expected_phase(sequence)
                scheduled_source_ns = epoch_ns + expected["scheduled_offset_ns"]
                observed_source_ns = self._wait_until(scheduled_source_ns)
                if (
                    observed_source_ns - scheduled_source_ns
                    > protocol.MAXIMUM_BOUNDARY_OVERRUN_NS
                ):
                    raise protocol.ValidationError(
                        "coordinator source boundary exceeded the frozen overrun limit"
                    )
                eligible_ns = observed_source_ns + delay_ns
                self._wait_until(eligible_ns)
                write_started_ns = self._clock_ns()
                self.bridge.publish(
                    generation,
                    sequence,
                    PHASE_IDS[expected["phase"]],
                    observed_source_ns,
                )
                successful_updates = sequence
                write_finished_ns = self._clock_ns()
                status = self.bridge.status()
                _require_publication(
                    status,
                    implementation=implementation,
                    generation=generation,
                    sequence=sequence,
                    phase=PHASE_IDS[expected["phase"]],
                    source_mono_ns=observed_source_ns,
                    write_started_mono_ns=write_started_ns,
                    write_finished_mono_ns=write_finished_ns,
                )
                publications.append(
                    {
                        "event": "snapshot_published",
                        "sequence": sequence,
                        "phase": expected["phase"],
                        "scheduled_offset_ns": expected["scheduled_offset_ns"],
                        "scheduled_source_mono_ns": scheduled_source_ns,
                        "source_mono_ns": observed_source_ns,
                        "eligible_mono_ns": eligible_ns,
                        "write_started_mono_ns": write_started_ns,
                        "write_finished_mono_ns": write_finished_ns,
                        "published_mono_ns": status.published_mono_ns,
                        "delay_ns": delay_ns,
                    }
                )

            final_enabled_status = self.bridge.status()
            _require_publication(
                final_enabled_status,
                implementation=implementation,
                generation=generation,
                sequence=protocol.MEASURED_PHASES + 1,
                phase=PHASE_IDS[expected["phase"]],
                source_mono_ns=observed_source_ns,
                write_started_mono_ns=write_started_ns,
                write_finished_mono_ns=write_finished_ns,
            )
            result = {
                "backend": type(self.bridge).__name__,
                "live_bridge": self.bridge.live,
                "experiment_evidence": self.bridge.experiment_evidence,
                "synthetic_source": True,
                "implementation": implementation,
                "generation": generation,
                "delay_ms": delay_ms,
                "epoch_mono_ns": epoch_ns,
                "configured_status": configured_status.as_record(),
                "publications": publications,
                "final_enabled_status": final_enabled_status.as_record(),
            }
        except Exception as exc:
            failure = exc
        finally:
            cleanup_failure: Exception | None = None
            try:
                self.bridge.disable(generation)
            except Exception as cleanup_exc:
                cleanup_failure = cleanup_exc
            else:
                try:
                    disabled_status = self.bridge.status()
                    _require_disabled(
                        disabled_status,
                        generation,
                        expected_updates=successful_updates,
                    )
                except Exception as cleanup_exc:
                    cleanup_failure = cleanup_exc
            if cleanup_failure is not None:
                if failure is not None:
                    raise ExceptionGroup(
                        "coordinator replay and cleanup validation both failed",
                        [failure, cleanup_failure],
                    )
                raise cleanup_failure

        if failure is not None:
            raise failure
        if result is None:
            raise AssertionError("coordinator replay completed without a result")
        if disabled_status is None:
            raise AssertionError("coordinator cleanup completed without status")
        result["disabled_status"] = disabled_status.as_record()
        return result


def _valid_u64(value: object) -> bool:
    return type(value) is int and 0 < value <= UINT64_MAX


def _require_configured(status: BridgeStatus, mode: str, generation: int) -> None:
    if (
        status.mode != mode
        or status.generation != generation
        or status.snapshot_present != 0
        or status.active_callbacks != 0
    ):
        raise protocol.ValidationError("bridge configure acknowledgement is inconsistent")
    _require_bridge_only_counters(status, expected_updates=0)


def _require_idle(status: BridgeStatus) -> None:
    if (
        status.mode != "off"
        or status.snapshot_present != 0
        or status.active_callbacks != 0
    ):
        raise protocol.ValidationError(
            "stale-state endpoint is not idle; refusing to overwrite live state"
        )


def _require_publication(
    status: BridgeStatus,
    *,
    implementation: str,
    generation: int,
    sequence: int,
    phase: int,
    source_mono_ns: int,
    write_started_mono_ns: int,
    write_finished_mono_ns: int,
) -> None:
    if (
        status.mode != implementation
        or status.generation != generation
        or status.snapshot_present != 1
        or status.snapshot_sequence != sequence
        or status.snapshot_phase != phase
        or status.source_mono_ns != source_mono_ns
        or status.published_mono_ns < source_mono_ns
        or write_started_mono_ns > write_finished_mono_ns
        or status.published_mono_ns < write_started_mono_ns
        or status.published_mono_ns > write_finished_mono_ns
    ):
        raise protocol.ValidationError("bridge publication acknowledgement is inconsistent")
    _require_bridge_only_counters(status, expected_updates=sequence)


def _require_disabled(
    status: BridgeStatus, generation: int, *, expected_updates: int
) -> None:
    if (
        status.mode != "off"
        or status.generation != generation
        or status.snapshot_present != 0
        or status.snapshot_sequence != 0
        or status.snapshot_phase != 0
        or status.source_mono_ns != 0
        or status.published_mono_ns != 0
    ):
        raise protocol.ValidationError("bridge did not reach clean disabled state")
    _require_bridge_only_counters(status, expected_updates=expected_updates)


def _require_bridge_only_counters(
    status: BridgeStatus, *, expected_updates: int
) -> None:
    if status.snapshot_updates != expected_updates or status.snapshot_rejections != 0:
        raise protocol.ValidationError("bridge snapshot counters do not close")
    unexpected = {
        field: getattr(status, field)
        for field in COUNTER_FIELDS
        if field not in {"snapshot_updates", "snapshot_rejections"}
        and getattr(status, field) != 0
    }
    if unexpected or status.active_callbacks != 0:
        raise protocol.ValidationError(
            "bridge-only preflight observed callback or decision activity: "
            f"counters={unexpected}, active_callbacks={status.active_callbacks}"
        )
