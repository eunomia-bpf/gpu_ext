#!/usr/bin/env python3
"""Coordinator and test double for the stale-state driver bridge v1.

The in-memory bridge models the proc ABI for CPU tests.  Its output is never
live experiment evidence.  The proc bridge performs only versioned control
and status I/O; workload ownership and experiment validation remain the
responsibility of the future live runner.
"""

from __future__ import annotations

import errno
import json
import os
import select
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
TRUTH_RECORD_MAX_BYTES = 1024
TRUTH_RECORD_TIMEOUT_SECONDS = 30.0
LIVE_COORDINATOR_SCHEMA = "stale-state-live-truth-coordinator-v1"

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


class TruthFDCoordinator:
    """Relay workload-authored phase truth to the versioned driver endpoint.

    This component closes only the live truth-source/control-plane boundary.  A
    returned record is not a complete experiment cell: policy diagnostics, UVM
    events, safety monitors, leases, and workload correctness remain the live
    runner's responsibility.
    """

    def __init__(
        self,
        bridge: Bridge,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
        truth_timeout_seconds: float = TRUTH_RECORD_TIMEOUT_SECONDS,
    ):
        if (
            isinstance(truth_timeout_seconds, bool)
            or not isinstance(truth_timeout_seconds, (int, float))
            or truth_timeout_seconds <= 0
        ):
            raise protocol.ValidationError("truth record timeout must be positive")
        self.bridge = bridge
        self._clock_ns = clock_ns
        self._sleep = sleep
        self._truth_timeout_seconds = float(truth_timeout_seconds)

    def _wait_until(self, target_ns: int) -> int:
        now_ns = self._clock_ns()
        while now_ns < target_ns:
            self._sleep((target_ns - now_ns) / 1.0e9)
            next_ns = self._clock_ns()
            if next_ns <= now_ns:
                raise protocol.ValidationError(
                    "live truth coordinator monotonic clock did not advance"
                )
            now_ns = next_ns
        return now_ns

    def run(
        self,
        *,
        truth_fd: int,
        expected_pid: int,
        release: Callable[[], None],
        implementation: str | None,
        generation: int | None,
        delay_ms: int | None,
    ) -> dict[str, Any]:
        """Consume one workload truth pipe and always leave owned policy state off."""

        if type(truth_fd) is not int or truth_fd < 0:
            raise protocol.ValidationError("truth_fd must be a non-negative integer")
        if type(expected_pid) is not int or expected_pid <= 0:
            raise protocol.ValidationError("expected_pid must be a positive integer")
        if not callable(release):
            raise protocol.ValidationError("release must be callable")
        baseline = implementation is None
        if baseline:
            if generation is not None or delay_ms is not None:
                raise protocol.ValidationError(
                    "baseline truth relay must not receive policy configuration"
                )
        else:
            if implementation not in protocol.IMPLEMENTATIONS:
                raise protocol.ValidationError(
                    f"unsupported live implementation: {implementation!r}"
                )
            if not _valid_u64(generation):
                raise protocol.ValidationError(
                    f"invalid live generation: {generation!r}"
                )
            if type(delay_ms) is not int or delay_ms not in protocol.DELAYS_MS:
                raise protocol.ValidationError(
                    f"unsupported live delay: {delay_ms!r}"
                )

        duplicate = os.dup(truth_fd)
        configured = False
        successful_updates = 0
        failure: Exception | None = None
        disabled_status: BridgeStatus | None = None
        final_enabled_status: BridgeStatus | None = None
        configured_status: BridgeStatus | None = None
        result: dict[str, Any] | None = None
        publications: list[dict[str, Any]] = []

        try:
            initial_status = self.bridge.status()
            _require_idle(initial_status)
            reader = _TruthFDReader(duplicate, self._truth_timeout_seconds)
            try:
                ready = reader.read_required()
                _require_workload_ready(ready, expected_pid)

                if not baseline:
                    assert implementation is not None
                    assert generation is not None
                    assert delay_ms is not None
                    self.bridge.configure(implementation, generation)
                    configured = True
                    configured_status = self.bridge.status()
                    _require_configured(
                        configured_status, implementation, generation
                    )
                else:
                    _require_idle(self.bridge.status())

                release()
                delay_ns = 0 if delay_ms is None else delay_ms * 1_000_000
                epoch_ns: int | None = None
                previous_end_ns = 0

                for sequence in range(1, protocol.MEASURED_PHASES + 2):
                    expected = protocol.expected_phase(sequence)
                    start = reader.read_required()
                    received_start_ns = self._clock_ns()
                    _require_phase_record(start, "phase_start", expected)
                    source_ns = start["mono_ns"]
                    if epoch_ns is None:
                        epoch_ns = source_ns
                    scheduled_start_ns = epoch_ns + expected["scheduled_offset_ns"]
                    if (
                        source_ns < scheduled_start_ns
                        or source_ns
                        > scheduled_start_ns + protocol.MAXIMUM_BOUNDARY_OVERRUN_NS
                        or source_ns < previous_end_ns
                    ):
                        raise protocol.ValidationError(
                            f"phase {sequence} start is outside the frozen live timeline"
                        )
                    _require_prompt_truth_delivery(source_ns, received_start_ns)

                    if baseline:
                        _require_idle(self.bridge.status())
                    else:
                        assert implementation is not None
                        assert generation is not None
                        eligible_ns = source_ns + delay_ns
                        publish_ready_ns = self._wait_until(eligible_ns)
                        if (
                            publish_ready_ns
                            > eligible_ns + protocol.MAXIMUM_BOUNDARY_OVERRUN_NS
                        ):
                            raise protocol.ValidationError(
                                "snapshot publication exceeded the frozen overrun limit"
                            )
                        write_started_ns = self._clock_ns()
                        self.bridge.publish(
                            generation,
                            sequence,
                            PHASE_IDS[expected["phase"]],
                            source_ns,
                        )
                        successful_updates = sequence
                        write_finished_ns = self._clock_ns()
                        status = self.bridge.status()
                        status_observed_ns = self._clock_ns()
                        _require_live_publication(
                            status,
                            implementation=implementation,
                            generation=generation,
                            sequence=sequence,
                            phase=PHASE_IDS[expected["phase"]],
                            source_mono_ns=source_ns,
                            eligible_mono_ns=eligible_ns,
                            write_started_mono_ns=write_started_ns,
                            write_finished_mono_ns=write_finished_ns,
                            status_observed_mono_ns=status_observed_ns,
                        )
                        publications.append(
                            {
                                "event": "snapshot_published",
                                "publisher": "shared_driver_snapshot",
                                "consumer_implementation": implementation,
                                "sequence": sequence,
                                "phase": expected["phase"],
                                "scheduled_offset_ns": expected[
                                    "scheduled_offset_ns"
                                ],
                                "source_mono_ns": source_ns,
                                "eligible_mono_ns": eligible_ns,
                                "write_started_mono_ns": write_started_ns,
                                "write_finished_mono_ns": write_finished_ns,
                                "published_mono_ns": status.published_mono_ns,
                                "status_observed_mono_ns": status_observed_ns,
                                "delay_ns": delay_ns,
                            }
                        )

                    end = reader.read_required()
                    received_end_ns = self._clock_ns()
                    _require_phase_record(end, "phase_end", expected)
                    end_ns = end["mono_ns"]
                    duration_ns = (
                        protocol.BOOTSTRAP_NS
                        if sequence == 1
                        else protocol.PHASE_NS
                    )
                    scheduled_end_ns = scheduled_start_ns + duration_ns
                    if (
                        end_ns < scheduled_end_ns
                        or end_ns
                        > scheduled_end_ns + protocol.MAXIMUM_BOUNDARY_OVERRUN_NS
                        or end_ns <= source_ns
                    ):
                        raise protocol.ValidationError(
                            f"phase {sequence} end is outside the frozen live timeline"
                        )
                    _require_prompt_truth_delivery(end_ns, received_end_ns)
                    previous_end_ns = end_ns
                    if baseline:
                        _require_idle(self.bridge.status())

                if reader.read_optional() is not None:
                    raise protocol.ValidationError(
                        "workload truth pipe contains trailing records"
                    )

                if baseline:
                    _require_idle(self.bridge.status())
                else:
                    assert implementation is not None
                    assert generation is not None
                    final_enabled_status = self.bridge.status()
                    _require_live_final(
                        final_enabled_status,
                        implementation=implementation,
                        generation=generation,
                        expected_updates=successful_updates,
                    )

                result = {
                    "schema": LIVE_COORDINATOR_SCHEMA,
                    "truth_source": "workload_phase_fd",
                    "synthetic_source": False,
                    "live_bridge": self.bridge.live,
                    "experiment_evidence": False,
                    "evidence_scope": "coordinator_only_not_complete_cell",
                    "target_pid": expected_pid,
                    "implementation": implementation,
                    "generation": generation,
                    "delay_ms": delay_ms,
                    "policy_configured": not baseline,
                    "baseline_policy_artifacts": False,
                    "truth_record_count": 15,
                    "publications": publications,
                    "configured_status": (
                        None
                        if configured_status is None
                        else configured_status.as_record()
                    ),
                    "final_enabled_status": (
                        None
                        if final_enabled_status is None
                        else final_enabled_status.as_record()
                    ),
                }
            finally:
                reader.close()
                duplicate = -1
        except Exception as exc:
            failure = exc
        finally:
            if duplicate >= 0:
                os.close(duplicate)
            cleanup_failure: Exception | None = None
            if configured:
                assert generation is not None
                try:
                    self.bridge.disable(generation)
                except Exception as cleanup_exc:
                    cleanup_failure = cleanup_exc
                else:
                    try:
                        disabled_status = self.bridge.status()
                        _require_live_disabled(
                            disabled_status,
                            generation=generation,
                            expected_updates=successful_updates,
                            final_enabled_status=final_enabled_status,
                        )
                    except Exception as cleanup_exc:
                        cleanup_failure = cleanup_exc
            if cleanup_failure is not None:
                if failure is not None:
                    raise ExceptionGroup(
                        "live truth relay and cleanup validation both failed",
                        [failure, cleanup_failure],
                    )
                raise cleanup_failure

        if failure is not None:
            raise failure
        if result is None:
            raise AssertionError("live truth relay completed without a result")
        if configured:
            if disabled_status is None:
                raise AssertionError("live truth relay cleanup has no status")
            result["disabled_status"] = disabled_status.as_record()
        else:
            result["disabled_status"] = None
        return result


def _decode_truth_record(payload: bytes) -> dict[str, Any]:
    if not payload or payload == b"\n" or not payload.endswith(b"\n"):
        raise protocol.ValidationError("workload truth record is empty or unterminated")
    if len(payload) > TRUTH_RECORD_MAX_BYTES:
        raise protocol.ValidationError("workload truth record exceeds the size limit")

    def reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise protocol.ValidationError(
                    f"duplicate workload truth field: {key}"
                )
            value[key] = item
        return value

    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(text, object_pairs_hook=reject_duplicate_fields)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise protocol.ValidationError(
            f"malformed workload truth JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise protocol.ValidationError("workload truth record is not an object")
    return value


class _TruthFDReader:
    """Deadline-aware newline reader over one duplicated workload pipe fd."""

    def __init__(self, descriptor: int, timeout_seconds: float):
        self._descriptor = descriptor
        self._timeout_seconds = timeout_seconds
        self._buffer = bytearray()
        self._eof = False

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1

    def _payload(self) -> bytes | None:
        deadline = time.monotonic() + self._timeout_seconds
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                length = newline + 1
                payload = bytes(self._buffer[:length])
                del self._buffer[:length]
                if length > TRUTH_RECORD_MAX_BYTES:
                    raise protocol.ValidationError(
                        "workload truth record exceeds the size limit"
                    )
                return payload
            if len(self._buffer) >= TRUTH_RECORD_MAX_BYTES:
                raise protocol.ValidationError(
                    "workload truth record exceeds the size limit"
                )
            if self._eof:
                if self._buffer:
                    payload = bytes(self._buffer)
                    self._buffer.clear()
                    return payload
                return None

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise protocol.ValidationError("workload truth pipe timed out")
            try:
                ready, _, _ = select.select(
                    [self._descriptor], [], [], remaining
                )
            except (OSError, ValueError) as exc:
                raise protocol.ValidationError(
                    f"cannot wait for workload truth pipe: {exc}"
                ) from exc
            if not ready:
                raise protocol.ValidationError("workload truth pipe timed out")
            try:
                chunk = os.read(self._descriptor, 4096)
            except OSError as exc:
                raise protocol.ValidationError(
                    f"cannot read workload truth pipe: {exc}"
                ) from exc
            if chunk:
                self._buffer.extend(chunk)
            else:
                self._eof = True

    def read_required(self) -> dict[str, Any]:
        payload = self._payload()
        if payload is None:
            raise protocol.ValidationError("workload truth pipe ended early")
        return _decode_truth_record(payload)

    def read_optional(self) -> dict[str, Any] | None:
        payload = self._payload()
        if payload is None:
            return None
        return _decode_truth_record(payload)


def _require_exact_fields(record: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(record)
    if actual != expected:
        raise protocol.ValidationError(
            f"{label} schema mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_workload_ready(record: dict[str, Any], expected_pid: int) -> None:
    _require_exact_fields(
        record,
        {"event", "pid", "protocol", "timeline", "allocation_bytes", "regions"},
        "workload_ready",
    )
    expected = {
        "event": "workload_ready",
        "pid": expected_pid,
        "protocol": protocol.PROTOCOL,
        "timeline": protocol.TIMELINE,
        "allocation_bytes": protocol.ALLOCATION_BYTES,
        "regions": protocol.REGIONS,
    }
    if (
        type(record.get("pid")) is not int
        or type(record.get("allocation_bytes")) is not int
        or type(record.get("regions")) is not int
        or record != expected
    ):
        raise protocol.ValidationError("workload_ready identity differs")


def _require_phase_record(
    record: dict[str, Any], event: str, expected: dict[str, Any]
) -> None:
    _require_exact_fields(
        record,
        {"event", "sequence", "phase", "measured", "scheduled_offset_ns", "mono_ns"},
        event,
    )
    if (
        type(record.get("sequence")) is not int
        or type(record.get("phase")) is not str
        or type(record.get("measured")) is not bool
        or type(record.get("scheduled_offset_ns")) is not int
    ):
        raise protocol.ValidationError(f"{event} field types are invalid")
    for field, value in (
        ("event", event),
        ("sequence", expected["sequence"]),
        ("phase", expected["phase"]),
        ("measured", expected["measured"]),
        ("scheduled_offset_ns", expected["scheduled_offset_ns"]),
    ):
        if record.get(field) != value:
            raise protocol.ValidationError(
                f"{event} {field} differs for sequence {expected['sequence']}"
            )
    if not _valid_u64(record.get("mono_ns")):
        raise protocol.ValidationError(f"{event} monotonic timestamp is invalid")


def _require_prompt_truth_delivery(event_ns: int, received_ns: int) -> None:
    if (
        received_ns < event_ns
        or received_ns > event_ns + protocol.MAXIMUM_BOUNDARY_OVERRUN_NS
    ):
        raise protocol.ValidationError(
            "workload truth event was not delivered within the frozen overrun limit"
        )


def _require_no_live_errors(status: BridgeStatus) -> None:
    errors = {
        field: getattr(status, field)
        for field in (
            "snapshot_rejections",
            "missing_snapshot_decisions",
            "invalid_snapshot_decisions",
            "request_errors",
            "effect_errors",
        )
        if getattr(status, field) != 0
    }
    if errors:
        raise protocol.ValidationError(
            f"live bridge reported an invalid action: {errors}"
        )


def _require_live_publication(
    status: BridgeStatus,
    *,
    implementation: str,
    generation: int,
    sequence: int,
    phase: int,
    source_mono_ns: int,
    eligible_mono_ns: int,
    write_started_mono_ns: int,
    write_finished_mono_ns: int,
    status_observed_mono_ns: int,
) -> None:
    if (
        status.mode != implementation
        or status.generation != generation
        or status.snapshot_present != 1
        or status.snapshot_sequence != sequence
        or status.snapshot_phase != phase
        or status.source_mono_ns != source_mono_ns
        or status.snapshot_updates != sequence
        or status.published_mono_ns < source_mono_ns
        or status.published_mono_ns < eligible_mono_ns
        or status.published_mono_ns
        > eligible_mono_ns + protocol.MAXIMUM_BOUNDARY_OVERRUN_NS
        or write_started_mono_ns > write_finished_mono_ns
        or status.published_mono_ns < write_started_mono_ns
        or status.published_mono_ns > write_finished_mono_ns
        or status_observed_mono_ns < write_finished_mono_ns
        or status_observed_mono_ns
        > eligible_mono_ns + protocol.MAXIMUM_BOUNDARY_OVERRUN_NS
    ):
        raise protocol.ValidationError(
            "live bridge publication acknowledgement is inconsistent"
        )
    _require_no_live_errors(status)
    inactive_count = (
        status.bpf_callback_invocations
        if implementation == "native"
        else status.native_callback_invocations
    )
    if inactive_count != 0:
        raise protocol.ValidationError("inactive live policy consumer was invoked")


def _require_live_final(
    status: BridgeStatus,
    *,
    implementation: str,
    generation: int,
    expected_updates: int,
) -> None:
    if (
        status.mode != implementation
        or status.generation != generation
        or status.snapshot_present != 1
        or status.snapshot_sequence != expected_updates
        or status.snapshot_updates != expected_updates
        or status.active_callbacks != 0
    ):
        raise protocol.ValidationError("live bridge final identity is inconsistent")
    _require_no_live_errors(status)
    callbacks = status.callback_invocations
    active_mode_callbacks = (
        status.native_callback_invocations
        if implementation == "native"
        else status.bpf_callback_invocations
    )
    inactive_mode_callbacks = (
        status.bpf_callback_invocations
        if implementation == "native"
        else status.native_callback_invocations
    )
    closure = (
        status.snapshot_read_attempts,
        status.snapshot_read_successes,
        active_mode_callbacks,
        status.decision_requests,
        status.decisions,
        status.decision_records,
        status.effect_requests,
        status.effect_records,
        status.selected_diagnostics,
        status.finished_diagnostics,
        status.dense_prefetch_decisions + status.discarded_prefetch_decisions,
    )
    if (
        callbacks == 0
        or inactive_mode_callbacks != 0
        or any(value != callbacks for value in closure)
        or status.dense_prefetch_decisions == 0
        or status.discarded_prefetch_decisions == 0
    ):
        raise protocol.ValidationError("live bridge callback counters do not close")


def _require_live_disabled(
    status: BridgeStatus,
    *,
    generation: int,
    expected_updates: int,
    final_enabled_status: BridgeStatus | None,
) -> None:
    if (
        status.mode != "off"
        or status.generation != generation
        or status.snapshot_present != 0
        or status.snapshot_sequence != 0
        or status.snapshot_phase != 0
        or status.source_mono_ns != 0
        or status.published_mono_ns != 0
        or status.snapshot_updates != expected_updates
        or status.active_callbacks != 0
    ):
        raise protocol.ValidationError("live bridge did not reach clean disabled state")
    _require_no_live_errors(status)
    if final_enabled_status is not None:
        for field in COUNTER_FIELDS:
            if getattr(status, field) != getattr(final_enabled_status, field):
                raise protocol.ValidationError(
                    f"live bridge cleanup changed counter {field}"
                )


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
