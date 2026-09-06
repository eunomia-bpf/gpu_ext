"""Scalar gpu-storage policy ABI and decision backends.

This module contains no LMCache hooks. It is the lean shared policy layer for
the standalone executor and :mod:`lmcache_gds_backend_adapter`: the exact
136-byte ``UVM_GPU_STORAGE_DECIDE`` command-82 ABI, plus fifo, native, and live
BPF deciders. Trusted callers remain solely responsible for storage handles,
GPU pointers, CUDA streams, and execution of the returned action.
"""

from __future__ import annotations

import ctypes
import logging
import os
import struct
import threading
from dataclasses import dataclass
from typing import Callable, Optional

__all__ = [
    "ABI_VERSION", "IOCTL_CMD", "PARAMS_SIZE", "PARAM_FIELD_OFFSETS",
    "OP_READ", "OP_WRITE", "FLAG_DEMAND", "FLAG_SPECULATIVE",
    "FLAG_RECOMPUTABLE", "FLAG_SAFE_TO_DEFER", "ACTION_SUBMIT_NOW",
    "ACTION_DEFER", "ACTION_RECOMPUTE", "MAX_PRIORITY", "MAX_DEFER_NS",
    "MIN_BATCH", "MAX_BATCH", "READ_DEFER_PRESSURE_PERMILLE",
    "WRITE_DEFER_PRESSURE_PERMILLE", "PolicyRequest", "Decision", "Decider",
    "FifoDecider", "NativeDecider", "BpfDecider", "GdsDecisionError",
    "native_decide", "build_decider",
]

logger = logging.getLogger(__name__)

ABI_VERSION = 1
IOCTL_CMD = 82
OP_READ = 0
OP_WRITE = 1
FLAG_DEMAND = 0x1
FLAG_SPECULATIVE = 0x2
FLAG_RECOMPUTABLE = 0x4
FLAG_SAFE_TO_DEFER = 0x8
ACTION_SUBMIT_NOW = 0
ACTION_DEFER = 1
ACTION_RECOMPUTE = 2
MAX_PRIORITY = 7
MAX_DEFER_NS = 10_000_000
MIN_BATCH = 1
MAX_BATCH = 64
READ_DEFER_PRESSURE_PERMILLE = 800
WRITE_DEFER_PRESSURE_PERMILLE = 600

# Natural alignment is part of the userspace/kernel interface. The explicit
# pads are at offsets 116 and 132.
_PARAM_FORMAT = "<4I9Q4IQI4xQI4x"
_PARAM_UNITS = (
    "I", "I", "I", "I", "Q", "Q", "Q", "Q", "Q", "Q", "Q", "Q", "Q",
    "I", "I", "I", "I", "Q", "I", "4x", "Q", "I", "4x",
)
_PARAM_FIELDS = (
    "abiVersion", "op", "requestFlags", "inputPriority", "requestId",
    "objectId", "bytes", "tenantId", "callerHint", "deadlineNs", "slackNs",
    "estimatedTransferNs", "recomputeNs", "queueDepth",
    "hbmPressurePermille", "action", "outputPriority", "deferNs",
    "batchTarget", "pad1", "callerTgid", "rmStatus", "pad2",
)


def _field_offsets() -> dict[str, int]:
    offsets: dict[str, int] = {}
    position = 0
    for unit, name in zip(_PARAM_UNITS, _PARAM_FIELDS, strict=True):
        offsets[name] = position
        position += struct.calcsize("<" + unit)
    return offsets


_PARAM = struct.Struct(_PARAM_FORMAT)
_PARAM_OUT = struct.Struct("IIQI4xQI")
PARAMS_SIZE = _PARAM.size
PARAM_FIELD_OFFSETS = _field_offsets()

if PARAMS_SIZE != 136:
    raise RuntimeError(f"UVM gpu-storage params must be 136 bytes, got {PARAMS_SIZE}")
for _name, _expected in {
    "requestId": 16,
    "action": 96,
    "deferNs": 104,
    "callerTgid": 120,
    "rmStatus": 128,
}.items():
    if PARAM_FIELD_OFFSETS[_name] != _expected:
        raise RuntimeError(
            f"UVM gpu-storage offset drift for {_name}: "
            f"{PARAM_FIELD_OFFSETS[_name]} != {_expected}"
        )


def _clamp(value: int, low: int, high: int) -> int:
    return low if value < low else high if value > high else value


@dataclass(frozen=True)
class PolicyRequest:
    op: int = OP_READ
    flags: int = 0
    priority: int = 0
    nbytes: int = 0
    tenant_id: int = 0
    caller_hint: int = 0
    deadline_ns: int = 0
    slack_ns: int = 0
    estimated_transfer_ns: int = 0
    recompute_ns: int = 0
    queue_depth: int = 0
    hbm_pressure_permille: int = 0

    def pack(self, request_id: int = 0, object_id: int = 0) -> bytearray:
        """Pack inputs and zero outputs in the exact 136-byte ioctl buffer."""
        return bytearray(
            _PARAM.pack(
                ABI_VERSION, self.op, self.flags, self.priority, request_id,
                object_id, self.nbytes, self.tenant_id, self.caller_hint,
                self.deadline_ns, self.slack_ns, self.estimated_transfer_ns,
                self.recompute_ns, self.queue_depth, self.hbm_pressure_permille,
                0, 0, 0, 0, 0, 0,
            )
        )


@dataclass(frozen=True)
class Decision:
    action: int
    defer_ns: int
    priority: int
    batch_target: int


class GdsDecisionError(RuntimeError):
    """The live ioctl failed or returned an unusable decision."""


class Decider:
    def decide(self, request: PolicyRequest, request_id: int) -> Decision:
        raise NotImplementedError

    def close(self) -> None:
        return None


class FifoDecider(Decider):
    def decide(self, request: PolicyRequest, request_id: int) -> Decision:
        del request_id
        return Decision(
            ACTION_SUBMIT_NOW, 0,
            _clamp(request.priority, 0, MAX_PRIORITY), MIN_BATCH,
        )


def native_decide(request: PolicyRequest) -> Decision:
    """Mirror ``gds_policy.bpf.c`` precedence and clamps exactly."""
    priority = _clamp(request.priority, 0, MAX_PRIORITY)
    if request.op == OP_READ and request.flags & FLAG_DEMAND:
        return Decision(ACTION_SUBMIT_NOW, 0, priority, MIN_BATCH)
    if (
        request.op == OP_READ
        and request.flags & FLAG_RECOMPUTABLE
        and request.recompute_ns < request.estimated_transfer_ns
        and request.recompute_ns <= request.slack_ns
    ):
        return Decision(ACTION_RECOMPUTE, 0, priority, MIN_BATCH)
    if (
        request.op == OP_READ
        and request.flags & FLAG_SPECULATIVE
        and request.flags & FLAG_SAFE_TO_DEFER
        and request.hbm_pressure_permille >= READ_DEFER_PRESSURE_PERMILLE
    ):
        return Decision(
            ACTION_DEFER,
            _clamp(request.slack_ns, 0, MAX_DEFER_NS), priority, MIN_BATCH,
        )
    if (
        request.op == OP_WRITE
        and request.flags & FLAG_SAFE_TO_DEFER
        and request.hbm_pressure_permille >= WRITE_DEFER_PRESSURE_PERMILLE
    ):
        return Decision(
            ACTION_DEFER,
            _clamp(request.slack_ns, 0, MAX_DEFER_NS), priority,
            _clamp(request.queue_depth, MIN_BATCH, MAX_BATCH),
        )
    return Decision(ACTION_SUBMIT_NOW, 0, priority, MIN_BATCH)


class NativeDecider(Decider):
    def decide(self, request: PolicyRequest, request_id: int) -> Decision:
        del request_id
        return native_decide(request)


_LIBC: Optional[ctypes.CDLL] = None


def _libc() -> ctypes.CDLL:
    global _LIBC
    if _LIBC is None:
        _LIBC = ctypes.CDLL(None, use_errno=True)
    return _LIBC


def _uvm_ioctl(fd: int, command: int, buffer: bytearray) -> None:
    pointer = (ctypes.c_char * PARAMS_SIZE).from_buffer(buffer)
    result = _libc().ioctl(fd, command, pointer)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), f"UVM ioctl {command}")


class BpfDecider(Decider):
    """Issue command 82; the driver invokes the attached BPF policy."""

    def __init__(
        self,
        uvm_device: str = "/dev/nvidia-uvm",
        uvm_fd: Optional[int] = None,
        open_func: Callable[[str, int], int] = os.open,
        close_func: Callable[[int], None] = os.close,
        ioctl_func: Optional[Callable[[int, int, bytearray], None]] = None,
    ) -> None:
        self._device = uvm_device
        self._fd = uvm_fd
        self._opened = uvm_fd is not None
        self._open = open_func
        self._close = close_func
        self._ioctl = ioctl_func or _uvm_ioctl
        self._lock = threading.Lock()

    def decide(self, request: PolicyRequest, request_id: int) -> Decision:
        with self._lock:
            fd = self._fd_or_open()
            buffer = request.pack(request_id=request_id, object_id=request_id)
            self._ioctl(fd, IOCTL_CMD, buffer)
        action, priority, defer_ns, batch_target, _tgid, rm_status = (
            _PARAM_OUT.unpack_from(buffer, PARAM_FIELD_OFFSETS["action"])
        )
        if rm_status != 0:
            raise GdsDecisionError(
                f"UVM_GPU_STORAGE_DECIDE returned rmStatus={rm_status}"
            )
        if action not in {ACTION_SUBMIT_NOW, ACTION_DEFER, ACTION_RECOMPUTE}:
            raise GdsDecisionError(
                f"UVM_GPU_STORAGE_DECIDE returned unknown action {action}"
            )
        return Decision(action, defer_ns, priority, batch_target)

    def _fd_or_open(self) -> int:
        if self._fd is None:
            try:
                self._fd = self._open(self._device, os.O_RDWR | os.O_CLOEXEC)
            except OSError as error:
                raise GdsDecisionError(
                    f"cannot open UVM device {self._device}: {error}"
                ) from error
            self._opened = True
        return self._fd

    def close(self) -> None:
        with self._lock:
            if self._opened and self._fd is not None:
                try:
                    self._close(self._fd)
                except OSError as error:
                    logger.warning("BpfDecider close failed: %s", error)
                self._fd = None
                self._opened = False


def build_decider(mode: str, *, uvm_device: str = "/dev/nvidia-uvm") -> Decider:
    if mode == "fifo":
        return FifoDecider()
    if mode == "native":
        return NativeDecider()
    if mode == "bpf":
        return BpfDecider(uvm_device=uvm_device)
    raise ValueError(f"unknown policy mode {mode!r}; expected fifo, native, or bpf")
