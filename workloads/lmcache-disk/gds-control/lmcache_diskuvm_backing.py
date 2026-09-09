"""Disk-backed UVM GPU-promotion backing for completed LMCache 0.5.4 KV chunks.

Userspace seam that drives the sealed disk-backed managed-range mechanism of
the nvidia-uvm driver built from ``gpu_ext-kernel-575-gds`` (driver
``dea1fefc``) against **actual completed and immutable LMCache KV chunks** the
installed GDS backend has already durably written.  It is not a new benchmark:
the five completed full-read primitive cells in
``results-disk-uvm-gpu-promotion-20260908.md`` already characterize the
mechanism; this module points it at the real on-disk KV file of finished cache
objects instead of synthetic pattern pages, and wires it into the serving
put/get path.

Design (resolved scope)
-----------------------
- put -> immutable completion -> offload, then later get -> restore.  A chunk
  is prepared **once** at the GDS write-completion callback and the retained
  backing is **reused** on later demand reads. A demand get does not rebuild
  or re-offload that backing; the driver reads disk when fault hydration is
  needed.
- The managed range registers **exactly ``kv_bytes``** when ``kv_bytes`` is a
  multiple of the 2 MiB UVM VA block.  A chunk whose region is not a whole VA
  block is left untouched (the demand get falls back to the stock GDS read,
  unchanged). No file layout is extended or truncated; OFFLOAD rewrites the
  existing KV data region with the same bytes. No new driver ABI is added.
- The managed range is allocated with ``cudaMallocManaged`` at the **exact**
  ``kv_bytes`` (no overallocation, no interior sub-range): the driver attaches
  a disk backing to the **whole** managed range, so the start must already be
  2 MiB block-aligned.  When the exact allocation is not block-aligned the
  chunk is left untouched (the demand get falls back to the stock GDS read).
  No sub-range registration, no file extension, no new driver ABI.
- Transport is **explicit CPU staging**: on a GPU fault for a page that is
  durably on disk but resident nowhere, the driver hydrates the durable bytes
  into a transient CPU staging chunk and uses its normal CPU->GPU copy-engine
  pass.  Not NVMe-to-GPU P2P, and not a transparent/automatic offload.
- No per-chunk clock/timer/performance gate is added; the offload path runs
  ioctls and memory moves only.  The OFFLOAD ioctl (85) is asynchronous per
  the driver header, so preparation polls QUERY until the span reports no
  pending pages (matching the existing primitive) with **no arbitrary short
  timeout**.  There is no whole-buffer readback/comparison gate: the durable
  bytes are populated into the about-to-be-sealed range before ``OFFLOAD`` so
  the writeback is idempotent.

Driver contract (authoritative source ``uvm_ioctl.h``, ABI v1)
-------------------------------------------------------------
::

    UVM_DISK_BACKING_REGISTER            = ioctl 84
    UVM_DISK_BACKING_OFFLOAD             = ioctl 85   (asynchronous)
    UVM_DISK_BACKING_QUERY               = ioctl 86
    UVM_DISK_BACKING_SET_GPU_PROMOTION   = ioctl 87

Registration alone does not mark any page disk-backed and does not import an
existing file.  ``OFFLOAD`` performs real writeback and, on success, sets the
durable ``on_disk`` page bits and releases the in-memory copies.  Only then is
a page eligible for GPU-promotion hydration, which is why preparation always
runs ``OFFLOAD`` and awaits completion before it will restore.

CPU-importable
--------------
Importing this module does not import CUDA, torch, or the GDS backend; those
are resolved lazily. The injectable allocator and ioctl interfaces permit
CPU-only tests, but do not constitute a real disk/GPU run. Root owns the
GPU build/run.
"""

from __future__ import annotations

import atexit
import ctypes
import fcntl
import json
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DISK_UVM_ABI_VERSION",
    "UVM_DISK_BACKING_REGISTER",
    "UVM_DISK_BACKING_OFFLOAD",
    "UVM_DISK_BACKING_QUERY",
    "UVM_DISK_BACKING_SET_GPU_PROMOTION",
    "UVM_VA_BLOCK_BYTES",
    "UVM_PAGE_BYTES",
    "LMCACHE_GDS_METADATA_BYTES",
    "NV_OK",
    "NV_ERR_ILLEGAL_ACTION",
    "NV_ERR_INVALID_ARGUMENT",
    "NV_ERR_INVALID_STATE",
    "nv_status_str",
    "RegisterParams",
    "OffloadParams",
    "QueryParams",
    "GpuPromotionParams",
    "abi_layout_self_check",
    "DiskUvmError",
    "DiskUvmChunkNotFitError",
    "DiskUvmNotOnDiskError",
    "DiskUvmOffloadError",
    "ManagedRange",
    "ByteStubManagedRange",
    "IoctlDispatch",
    "find_uvm_fd",
    "ChunkBacking",
    "DiskUvmBacking",
    "DiskUvmStore",
    "build_chunk_from_key",
    "install_backend",
    "backing_diagnostics",
    "write_diagnostics",
    "MODE_ENV",
    "DIAG_OUT_ENV",
    "bootstrap_from_env",
]

# --------------------------------------------------------------------- ABI --
DISK_UVM_ABI_VERSION = 1
UVM_DISK_BACKING_REGISTER = 84
UVM_DISK_BACKING_OFFLOAD = 85
UVM_DISK_BACKING_QUERY = 86
UVM_DISK_BACKING_SET_GPU_PROMOTION = 87

UVM_VA_BLOCK_BYTES = 1 << 21  # 2 MiB: managed range size and alignment.
UVM_PAGE_BYTES = 4096  # driver page granularity used by the QUERY counters.

# The GDS on-disk layout reserves a fixed metadata prefix before the KV bytes
# (installed ``gds_backend._METADATA_MAX_SIZE``).  The KV region of a chunk is
# ``[LMCACHE_GDS_METADATA_BYTES, file_size)``.
LMCACHE_GDS_METADATA_BYTES = 4096

_NV_OK = 0x00000000
NV_OK = _NV_OK
NV_ERR_ILLEGAL_ACTION = 0x00000016
NV_ERR_INVALID_ARGUMENT = 0x0000001F
NV_ERR_INVALID_STATE = 0x00000040
_NV_STATUS_NAMES = {
    0x00000000: "NV_OK",
    0x00000003: "NV_ERR_BUSY_RETRY",
    0x00000016: "NV_ERR_ILLEGAL_ACTION",
    0x0000001E: "NV_ERR_INVALID_ADDRESS",
    0x0000001F: "NV_ERR_INVALID_ARGUMENT",
    0x00000038: "NV_ERR_INVALID_OPERATION",
    0x00000040: "NV_ERR_INVALID_STATE",
    0x00000051: "NV_ERR_NO_MEMORY",
    0x00000057: "NV_ERR_OBJECT_NOT_FOUND",
    0x00000063: "NV_ERR_STATE_IN_USE",
}


def nv_status_str(status: int) -> str:
    """Human name for an NV_STATUS value, or ``"?0x...?"`` when unknown."""
    return _NV_STATUS_NAMES.get(status, "?0x%08x?" % status)


# Userspace mirrors of UVM_DISK_BACKING_*_PARAMS (uvm_ioctl.h, ABI v1).  The
# ctypes layout reproduces the C struct including the 8-byte alignment pad the
# compiler adds after the final 32-bit field; ``abi_layout_self_check`` asserts
# the sizes and key offsets exactly like the C static_asserts.
class RegisterParams(ctypes.Structure):
    _fields_ = [
        ("abiVersion", ctypes.c_uint32),
        ("pad0", ctypes.c_uint32),
        ("rangeStart", ctypes.c_uint64),
        ("rangeEnd", ctypes.c_uint64),
        ("fileOffset", ctypes.c_uint64),
        ("fileFd", ctypes.c_int32),
        ("pad1", ctypes.c_uint32),
        ("rmStatus", ctypes.c_uint32),
    ]


class OffloadParams(ctypes.Structure):
    _fields_ = [
        ("abiVersion", ctypes.c_uint32),
        ("pad0", ctypes.c_uint32),
        ("rangeStart", ctypes.c_uint64),
        ("rangeEnd", ctypes.c_uint64),
        ("rmStatus", ctypes.c_uint32),
    ]


class QueryParams(ctypes.Structure):
    _fields_ = [
        ("abiVersion", ctypes.c_uint32),
        ("pad0", ctypes.c_uint32),
        ("rangeStart", ctypes.c_uint64),
        ("rangeEnd", ctypes.c_uint64),
        ("totalNumPages", ctypes.c_uint32),
        ("onDiskPages", ctypes.c_uint32),
        ("pendingPages", ctypes.c_uint32),
        ("errorPages", ctypes.c_uint32),
        ("rmStatus", ctypes.c_uint32),
    ]


class GpuPromotionParams(ctypes.Structure):
    _fields_ = [
        ("abiVersion", ctypes.c_uint32),
        ("pad0", ctypes.c_uint32),
        ("rangeStart", ctypes.c_uint64),
        ("rangeEnd", ctypes.c_uint64),
        ("enable", ctypes.c_uint32),
        ("pad1", ctypes.c_uint32),
        ("rmStatus", ctypes.c_uint32),
    ]


def _field_offset(struct_type: type, name: str) -> int:
    """Byte offset of ``name`` (little-endian host layout), located by a
    sentinel so it is exact for multi-byte fields."""
    sentinel = 0x12345678
    fields = dict(struct_type._fields_)
    instance = struct_type()
    for fname, _ftype in struct_type._fields_:
        setattr(instance, fname, 0)
    setattr(instance, name, sentinel)
    blob = bytes(
        ctypes.string_at(ctypes.addressof(instance), ctypes.sizeof(struct_type))
    )
    ctype = fields[name]
    if ctype in (ctypes.c_uint64, ctypes.c_int64):
        pattern = struct.pack("<Q", sentinel)
    else:
        pattern = struct.pack("<I", sentinel)
    offset = blob.find(pattern)
    assert offset >= 0, f"field {name} not located in {struct_type.__name__}"
    return offset


def abi_layout_self_check() -> None:
    """Assert the userspace mirrors match the driver ABI byte-for-byte.

    Mirrors the C ``static_assert``/``offsetof`` checks in ``disk_uvm_perf.cu``
    and adds the two that file omitted (the GPU-promotion size and its
    ``rmStatus`` offset).  Raises ``AssertionError`` on drift.
    """
    assert ctypes.sizeof(RegisterParams) == 48, "REGISTER params size"
    assert _field_offset(RegisterParams, "fileFd") == 32, "REGISTER.fileFd"
    assert ctypes.sizeof(OffloadParams) == 32, "OFFLOAD params size"
    assert ctypes.sizeof(QueryParams) == 48, "QUERY params size"
    assert _field_offset(QueryParams, "totalNumPages") == 24, "QUERY.totalNumPages"
    assert ctypes.sizeof(GpuPromotionParams) == 40, "GPU-promotion params size"
    assert _field_offset(GpuPromotionParams, "rmStatus") == 32, "GPU-promotion.rmStatus"


# ------------------------------------------------------------------ errors --
class DiskUvmError(RuntimeError):
    """Base class for disk-UVM backing errors."""


class DiskUvmChunkNotFitError(DiskUvmError):
    """The on-disk KV region is not a whole 2 MiB UVM VA block; the chunk is
    not disk-UVM-capable in this driver revision (no sub-range registration).
    The demand get falls back to the stock GDS read, unchanged."""


class DiskUvmNotOnDiskError(DiskUvmError):
    """The named on-disk KV file does not exist or is smaller than the prefix."""


class DiskUvmOffloadError(DiskUvmError):
    """A register/offload/query/promotion ioctl failed or the span never became
    durably on disk after the asynchronous writeback."""


def _classify_alloc_error(exc: BaseException) -> str:
    """Map a managed ``alloc`` failure to a prepare reason.

    The CUDA seam raises ``DiskUvmAlignmentError`` when the exact-size
    allocation is not 2 MiB block-aligned; any other alloc failure (OOM, CUDA
    error) is a generic error.  Matched by class name so this CPU-only module
    never imports the CUDA seam.
    """
    if type(exc).__name__ == "DiskUvmAlignmentError":
        return "not_aligned"
    return "error"


# ------------------------------------------------------- managed range seam --
class ManagedRange:
    """The managed-memory seam.  ``DiskUvmBacking`` only programs ioctls and
    drives this seam; the real GPU implementation (root) supplies
    ``cudaMallocManaged`` + a fault-inducing read + a device-to-device copy,
    and CPU tests supply :class:`ByteStubManagedRange`.

    ``copy_to(va, size, dest)`` copies ``[va, va+size)`` into ``dest``.  In the
    CUDA implementation ``dest`` is a device pointer (int) and this is a GPU
    managed-range -> real-destination device-to-device copy; in the stub it is a
    writable byte buffer.  The CPU->GPU copy-engine pass is the driver's
    CPU-staged fault hydration (driven by ``gpu_fault_read``), not this D2D copy.
    """

    def alloc(self, size: int) -> int:
        raise NotImplementedError

    def free(self, va: int) -> None:
        raise NotImplementedError

    def snapshot(self, va: int, size: int) -> bytes:
        raise NotImplementedError

    def populate(self, va: int, src_bytes: bytes) -> None:
        raise NotImplementedError

    def gpu_fault_read(self, va: int, size: int) -> int:
        raise NotImplementedError

    def copy_to(self, va: int, size: int, dest: Any) -> int:
        raise NotImplementedError


class ByteStubManagedRange(ManagedRange):
    """CPU-only managed-range stand-in backed by a ``bytearray``.

    ``alloc`` returns a synthetic 2 MiB-aligned VA; ``populate`` writes the
    region; ``gpu_fault_read`` records the fault (the observable a test asserts
    on); ``copy_to`` performs a plain byte copy, mirroring the driver's
    CPU->GPU copy-engine pass.  No CUDA.
    """

    def __init__(self) -> None:
        self._store: bytearray = bytearray()
        self._va_base = UVM_VA_BLOCK_BYTES  # synthetic, already block-aligned
        self._alloc_size = 0
        self.fault_reads: List[Tuple[int, int]] = []

    @property
    def alloc_size(self) -> int:
        return self._alloc_size

    def alloc(self, size: int) -> int:
        if size <= 0 or size % UVM_VA_BLOCK_BYTES != 0:
            raise ValueError(f"stub managed alloc must be a VA-block multiple: {size}")
        pad = size - len(self._store)
        if pad > 0:
            self._store.extend(b"\x00" * pad)
        self._alloc_size = size
        return self._va_base

    def free(self, va: int) -> None:
        self._alloc_size = 0

    def snapshot(self, va: int, size: int) -> bytes:
        return bytes(self._store[len(self._store) - size:])

    def populate(self, va: int, src_bytes: bytes) -> None:
        size = len(src_bytes)
        self._store[len(self._store) - size:] = src_bytes

    def gpu_fault_read(self, va: int, size: int) -> int:
        self.fault_reads.append((va, size))
        return size

    def copy_to(self, va: int, size: int, dest: bytearray) -> int:
        src = self._store[len(self._store) - size:]
        dest[:size] = src
        return size


# ------------------------------------------------------- ioctl dispatch seam --
# A dispatch callable mirrors ``fcntl.ioctl``: ``(fd, command, buf)`` where
# ``buf`` is a mutable ``bytearray`` of exactly the params size; the kernel
# writes OUT fields (notably ``rmStatus``) back into it.  The default is the
# real system ioctl; tests inject a recording driver.
IoctlDispatch = Callable[[int, int, bytearray], None]


def _default_ioctl_dispatch(fd: int, command: int, buf: bytearray) -> None:
    fcntl.ioctl(fd, command, buf)


# ----------------------------------------------------------------- fd probe --
def _list_nvidia_uvm_fds() -> List[int]:
    """All fds in this process that point at /dev/nvidia-uvm.

    Port of the .cu traversal: libcuda may open several /dev/nvidia-uvm fds;
    only one carries the process UVM va space owning a given managed range.
    """
    fds: List[int] = []
    for name in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{name}")
        except OSError:
            continue
        if target == "/dev/nvidia-uvm":
            fds.append(int(name))
    return sorted(fds)


def _query_probe(dispatch: IoctlDispatch, fd: int, start: int, end: int) -> int:
    """QUERY on ``[start, end]`` against ``fd``; return rmStatus.

    Per the driver the answer distinguishes the owner: NV_OK /
    NV_ERR_INVALID_STATE means the fd's va space contains the range (the latter
    being the pre-REGISTER state); NV_ERR_INVALID_ARGUMENT means the range is
    absent from this fd's va space; NV_ERR_ILLEGAL_ACTION means the fd was never
    initialized to a va space.
    """
    p = QueryParams()
    p.abiVersion = DISK_UVM_ABI_VERSION
    p.rangeStart = start
    p.rangeEnd = end
    p.rmStatus = 0
    buf = bytearray(ctypes.string_at(ctypes.addressof(p), ctypes.sizeof(p)))
    dispatch(fd, UVM_DISK_BACKING_QUERY, buf)
    ctypes.memmove(ctypes.addressof(p), buf, ctypes.sizeof(p))
    return int(p.rmStatus)


def find_uvm_fd(
    start: int,
    end: int,
    dispatch: Optional[IoctlDispatch] = None,
    candidate_fds: Optional[Sequence[int]] = None,
) -> int:
    """Select the /dev/nvidia-uvm fd whose va space owns ``[start, end]``.

    Returns the owning fd, or the .cu sentinels: -1 when no nvidia-uvm fd
    exists, -2 when no candidate owns the range, -3 when more than one does.
    ``candidate_fds`` overrides the /proc scan (injection point for tests).
    """
    disp = dispatch or _default_ioctl_dispatch
    cands = list(candidate_fds) if candidate_fds is not None else _list_nvidia_uvm_fds()
    if not cands:
        return -1
    owners: List[int] = []
    for fd in cands:
        st = _query_probe(disp, fd, start, end)
        if st in (NV_OK, NV_ERR_INVALID_STATE):
            owners.append(fd)
    if len(owners) == 0:
        return -2
    if len(owners) > 1:
        return -3
    return owners[0]


# --------------------------------------------------------------- offload/restore --
@dataclass
class ChunkBacking:
    """The result of preparing one completed chunk for GPU-promotion restore."""

    path: str
    file_offset: int
    kv_bytes: int
    total_pages: int
    managed_va: int
    uvm_fd: int
    backing_fd: int
    on_disk_pages: int
    pending_pages: int
    error_pages: int
    gpu_promotion_enabled: bool = False
    ioctl_trace: List[Tuple[int, int]] = field(default_factory=list)


class DiskUvmBacking:
    """Drives the disk-UVM ABI for one completed immutable LMCache KV chunk.

    All GPU/OS interaction is injected so the offload/restore sequencing is
    testable on a plain CPU.  Default arguments select the real system
    implementations for a root GPU run.  Preparation (``try_prepare``) runs
    once; ``restore`` reuses the retained backing.
    """

    def __init__(
        self,
        path: str,
        file_offset: int = LMCACHE_GDS_METADATA_BYTES,
        kv_bytes: Optional[int] = None,
        *,
        managed: ManagedRange,
        dispatch: Optional[IoctlDispatch] = None,
        open_backing: Optional[Callable[[str], int]] = None,
        candidate_fds: Optional[Sequence[int]] = None,
        poll_us: int = 1000,
    ) -> None:
        self.path = path
        self.file_offset = int(file_offset)
        self.managed = managed
        self._dispatch = dispatch or _default_ioctl_dispatch
        self._open_backing = open_backing or _open_backing_direct
        self._candidate_fds = candidate_fds
        self._poll_us = int(poll_us)
        if kv_bytes is None:
            kv_bytes = self._measure_kv_bytes()
        self.kv_bytes = int(kv_bytes)
        self.backing: Optional[ChunkBacking] = None
        self.last_reason = "ok"

    # ----------------------------------------------------------- measurement --
    def _measure_kv_bytes(self) -> int:
        """On-disk KV region length = file size minus the metadata prefix."""
        try:
            size = os.path.getsize(self.path)
        except OSError as error:
            raise DiskUvmNotOnDiskError(
                f"on-disk KV file {self.path} is not readable: {error}"
            ) from error
        if size <= self.file_offset:
            raise DiskUvmNotOnDiskError(
                f"on-disk KV file {self.path} is {size} bytes, which is not "
                f"larger than the {self.file_offset}-byte metadata prefix"
            )
        return size - self.file_offset

    @property
    def total_pages(self) -> int:
        return self.kv_bytes // UVM_PAGE_BYTES

    @property
    def fits(self) -> bool:
        """True when the on-disk region is exactly a whole number of UVM VA
        blocks and can be registered exactly without padding the file."""
        return self.kv_bytes > 0 and self.kv_bytes % UVM_VA_BLOCK_BYTES == 0

    # --------------------------------------------------------------- ioctls --
    def _ioctl(self, fd: int, command: int, params: ctypes.Structure,
               trace: Optional[List[Tuple[int, int]]] = None) -> int:
        buf = bytearray(
            ctypes.string_at(ctypes.addressof(params), ctypes.sizeof(params))
        )
        self._dispatch(fd, command, buf)
        ctypes.memmove(ctypes.addressof(params), buf, ctypes.sizeof(params))
        if trace is not None:
            trace.append((fd, command))
        return int(params.rmStatus)

    def _register(self, uvm_fd: int, start: int, end: int, backing_fd: int,
                  trace: List[Tuple[int, int]]) -> None:
        p = RegisterParams()
        p.abiVersion = DISK_UVM_ABI_VERSION
        p.rangeStart = start
        p.rangeEnd = end
        p.fileOffset = self.file_offset
        p.fileFd = backing_fd
        p.pad1 = 0
        p.rmStatus = 0
        st = self._ioctl(uvm_fd, UVM_DISK_BACKING_REGISTER, p, trace)
        if st != NV_OK:
            raise DiskUvmOffloadError(
                f"REGISTER rmStatus=0x{st:08x} ({nv_status_str(st)}) for "
                f"{self.path}[{self.file_offset}+{self.kv_bytes}]"
            )

    def _offload(self, uvm_fd: int, start: int, end: int,
                 trace: List[Tuple[int, int]]) -> None:
        p = OffloadParams()
        p.abiVersion = DISK_UVM_ABI_VERSION
        p.rangeStart = start
        p.rangeEnd = end
        p.rmStatus = 0
        st = self._ioctl(uvm_fd, UVM_DISK_BACKING_OFFLOAD, p, trace)
        if st != NV_OK:
            raise DiskUvmOffloadError(
                f"OFFLOAD rmStatus=0x{st:08x} ({nv_status_str(st)}) for {self.path}"
            )

    def _query(self, uvm_fd: int, start: int, end: int,
               trace: Optional[List[Tuple[int, int]]] = None) -> QueryParams:
        p = QueryParams()
        p.abiVersion = DISK_UVM_ABI_VERSION
        p.rangeStart = start
        p.rangeEnd = end
        p.rmStatus = 0
        st = self._ioctl(uvm_fd, UVM_DISK_BACKING_QUERY, p, trace)
        if st != NV_OK:
            raise DiskUvmOffloadError(
                f"QUERY rmStatus=0x{st:08x} ({nv_status_str(st)}) for {self.path}"
            )
        return p

    def _wait_offload_done(self, uvm_fd: int, start: int, end: int,
                           trace: List[Tuple[int, int]]) -> QueryParams:
        """Poll QUERY until the offloaded span reports no pending pages.

        Matches the existing primitive's ``wait_offload_done``: there is no
        arbitrary short timeout; the loop runs until the asynchronous writeback
        completes.
        """
        while True:
            q = self._query(uvm_fd, start, end, trace)
            if q.pendingPages == 0:
                return q
            time.sleep(self._poll_us / 1e6)

    def _enable_gpu_promotion(self, uvm_fd: int, start: int, end: int,
                              trace: List[Tuple[int, int]]) -> None:
        p = GpuPromotionParams()
        p.abiVersion = DISK_UVM_ABI_VERSION
        p.rangeStart = start
        p.rangeEnd = end
        p.enable = 1
        p.pad1 = 0
        p.rmStatus = 0
        st = self._ioctl(uvm_fd, UVM_DISK_BACKING_SET_GPU_PROMOTION, p, trace)
        if st != NV_OK:
            raise DiskUvmOffloadError(
                f"SET_GPU_PROMOTION rmStatus=0x{st:08x} ({nv_status_str(st)}) "
                f"for {self.path}"
            )

    # -------------------------------------------------------------- lifecycle --
    def try_prepare(self) -> Tuple[bool, Optional[ChunkBacking]]:
        """Prepare the chunk for GPU-promotion restore, running once.

        Populates the about-to-be-sealed managed range with the durable bytes
        (CPU-staged), opens the O_DIRECT backing file, selects the owning UVM
        fd, then runs REGISTER -> OFFLOAD (awaited) -> SET_GPU_PROMOTION and
        retains the backing.  Returns ``(False, None)`` when the region is not
        a whole VA block or any real ioctl/OS error occurs; in that case
        nothing is written and the on-disk file is left untouched, so the
        demand get falls back to the stock GDS read.
        """
        self.last_reason = "ok"
        if not self.fits:
            self.last_reason = "not_fitted"
            return (False, None)
        total_pages = self.total_pages
        trace: List[Tuple[int, int]] = []
        managed_va: Optional[int] = None
        backing_fd: Optional[int] = None
        success = False
        try:
            # Allocate the exact range first: the start must be block-aligned,
            # and an unaligned exact allocation is a cheap fallback (no read).
            try:
                managed_va = self.managed.alloc(self.kv_bytes)
            except Exception as alloc_exc:
                self.last_reason = _classify_alloc_error(alloc_exc)
                return (False, None)
            # Populate the durable bytes before sealing so the asynchronous
            # OFFLOAD writeback is idempotent.  No whole-buffer readback gate.
            on_disk = _read_on_disk_region(self.path, self.file_offset, self.kv_bytes)
            self.managed.populate(managed_va, on_disk)
            backing_fd = self._open_backing(self.path)
            start = managed_va
            end = managed_va + self.kv_bytes - 1
            uvm_fd = find_uvm_fd(start, end, self._dispatch, self._candidate_fds)
            if uvm_fd < 0:
                self.last_reason = "no_uvm_fd"
                return (False, None)
            self._register(uvm_fd, start, end, backing_fd, trace)
            self._offload(uvm_fd, start, end, trace)  # asynchronous
            q = self._wait_offload_done(uvm_fd, start, end, trace)
            if q.onDiskPages != total_pages or q.errorPages != 0:
                self.last_reason = "not_on_disk"
                return (False, None)
            self._enable_gpu_promotion(uvm_fd, start, end, trace)
            self.backing = ChunkBacking(
                path=self.path,
                file_offset=self.file_offset,
                kv_bytes=self.kv_bytes,
                total_pages=total_pages,
                managed_va=managed_va,
                uvm_fd=uvm_fd,
                backing_fd=backing_fd,
                on_disk_pages=int(q.onDiskPages),
                pending_pages=int(q.pendingPages),
                error_pages=int(q.errorPages),
                gpu_promotion_enabled=True,
                ioctl_trace=trace,
            )
            success = True
            return (True, self.backing)
        except Exception:
            self.last_reason = "error"
            return (False, None)
        finally:
            if not success:
                if managed_va is not None:
                    try:
                        self.managed.free(managed_va)
                    except Exception:
                        pass
                if backing_fd is not None:
                    try:
                        os.close(backing_fd)
                    except OSError:
                        pass

    def restore(self, dest_device_ptr: int) -> int:
        """Restore the prepared chunk into the destination device region.

        Reuses the retained backing: a single GPU fault over the sealed,
        on-disk, GPU-promotion range (the driver's CPU-staged disk->GPU
        copy-engine restore) followed by a device-to-device copy into
        ``dest_device_ptr``.  No re-read of disk, re-write, or re-offload.
        """
        if self.backing is None:
            raise DiskUvmError(
                "restore() before a successful try_prepare(); no retained backing"
            )
        self.managed.gpu_fault_read(self.backing.managed_va, self.kv_bytes)
        return self.managed.copy_to(self.backing.managed_va, self.kv_bytes,
                                    dest_device_ptr)

    def release(self) -> None:
        """Tear down: free the managed range and close the backing fd.  Per the
        driver, freeing the managed range unregisters the backing view."""
        if self.backing is None:
            return
        try:
            self.managed.free(self.backing.managed_va)
        except Exception:
            pass
        try:
            os.close(self.backing.backing_fd)
        except OSError:
            pass
        self.backing = None


# ---------------------------------------------------------- real plumbing --
def _open_backing_direct(path: str) -> int:
    """Open the on-disk KV file for direct I/O, mirroring the .cu contract.

    O_DIRECT is required; when the kernel does not honor it (fdinfo flags),
    buffered I/O is refused by design and an error is raised so the chunk falls
    back to the stock GDS read.
    """
    fd = os.open(path, os.O_RDWR | os.O_DIRECT)
    flags = _read_fdinfo_flags(fd)
    if flags and not _is_direct_flag(flags):
        os.close(fd)
        raise OSError(
            f"O_DIRECT not honored for {path}; kernel fdinfo flags={flags!r}. "
            f"The disk arm requires direct I/O; buffered I/O fallback is refused."
        )
    return fd


def _is_direct_flag(flags_field: str) -> bool:
    # O_DIRECT is 020000 octal; the .cu recorded 0140002 as the honored value.
    try:
        value = int(flags_field, 8)
    except ValueError:
        return False
    return bool(value & 0o20000)


def _read_fdinfo_flags(fd: int) -> str:
    try:
        with open(f"/proc/self/fdinfo/{fd}", "re") as handle:
            for line in handle:
                if line.startswith("flags:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return ""
    return ""


def _read_on_disk_region(path: str, file_offset: int, kv_bytes: int) -> bytes:
    """CPU-staged read of the on-disk KV region into a host buffer."""
    with open(path, "rb") as handle:
        handle.seek(file_offset)
        data = handle.read(kv_bytes)
    if len(data) != kv_bytes:
        raise DiskUvmNotOnDiskError(
            f"short read of {len(data)}/{kv_bytes} bytes from {path}"
        )
    return data


_log = logging.getLogger("lmcache.diskuvm")
# Process-wide registry of installed stores so the (one-line) diagnostics dump
# and atexit JSON cover every GDS backend instance in this serving process.
_ALL_STORES: "set[Any]" = set()


# ------------------------------------------------------------ retained store --
class DiskUvmStore:
    """Per-backend retained disk-UVM backings, keyed by ``CacheEngineKey``.

    Created at GDS write-completion (immutable completion) and reused on later
    demand reads.  A key with no entry simply means the demand get takes the
    stock GDS read path.

    ``counters`` distinguishes the transport outcomes so a silent stock
    fallback is never mistaken for working disk-UVM:
      - ``prepared``       chunks sealed + offloaded + GPU-promotion-enabled
      - ``restored``       demand reads served from a retained backing
      - ``stock_fallback`` demand reads that took the stock GDS read (no
                           prepared backing, unfit/unaligned chunk, or a
                           restore error)
      - ``not_fitted``     put-time: ``kv_bytes`` not a whole 2 MiB VA block
      - ``not_aligned``    put-time: exact allocation not 2 MiB block-aligned
      - ``error``          any other prepare/restore failure
    """

    def __init__(
        self,
        backend: Any,
        *,
        managed: ManagedRange,
        dispatch: Optional[IoctlDispatch] = None,
        candidate_fds: Optional[Sequence[int]] = None,
        poll_us: int = 1000,
        file_offset: int = LMCACHE_GDS_METADATA_BYTES,
    ) -> None:
        self._backend = backend
        self._managed = managed
        self._dispatch = dispatch
        self._candidate_fds = candidate_fds
        self._poll_us = int(poll_us)
        self._file_offset = int(file_offset)
        self._lock = threading.Lock()
        self._backings: Dict[Any, DiskUvmBacking] = {}
        self.counters: Dict[str, int] = {
            "prepared": 0,
            "restored": 0,
            "stock_fallback": 0,
            "not_fitted": 0,
            "not_aligned": 0,
            "error": 0,
        }
        _ALL_STORES.add(self)

    def lookup(self, key: Any) -> Optional[DiskUvmBacking]:
        with self._lock:
            return self._backings.get(key)

    def count(self, name: str) -> None:
        """Increment a counter (used on the read path, outside the store lock)."""
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + 1

    def prepare(self, key: Any, kv_bytes: int) -> Optional[DiskUvmBacking]:
        """Prepare ``key`` once at immutable completion; retain when it fits."""
        with self._lock:
            existing = self._backings.get(key)
            if existing is not None:
                return existing
            path = self._backend._key_to_path(key)[0]
            backing = DiskUvmBacking(
                path,
                file_offset=self._file_offset,
                kv_bytes=kv_bytes,
                managed=self._managed,
                dispatch=self._dispatch,
                candidate_fds=self._candidate_fds,
                poll_us=self._poll_us,
            )
            ok, chunk = backing.try_prepare()
            if ok and chunk is not None:
                self._backings[key] = backing
                self.counters["prepared"] += 1
                _log.debug(
                    "disk-uvm prepared key=%s kv_bytes=%d total_pages=%d "
                    "on_disk=%d va=0x%x fd=%d",
                    key, chunk.kv_bytes, chunk.total_pages,
                    chunk.on_disk_pages, chunk.managed_va, chunk.uvm_fd,
                )
                return backing
            reason = getattr(backing, "last_reason", "error")
            if reason == "not_fitted":
                self.counters["not_fitted"] = self.counters.get("not_fitted", 0) + 1
            elif reason == "not_aligned":
                self.counters["not_aligned"] = self.counters.get("not_aligned", 0) + 1
            else:
                self.counters["error"] = self.counters.get("error", 0) + 1
            _log.debug("disk-uvm prepare skipped key=%s reason=%s", key, reason)
            return None

    def describe(self) -> Dict[str, Any]:
        with self._lock:
            return {"retained": len(self._backings), **dict(self.counters)}

    def release_all(self) -> int:
        released = 0
        with self._lock:
            for backing in self._backings.values():
                backing.release()
                released += 1
            self._backings.clear()
        return released


# ----------------------------------------------------------- serving hooks --
_THREAD_TLS = threading.local()
_BACKEND_ATTR = "_disk_uvm_store"


def _gds_metadata_bytes(backend: Any) -> int:
    value = getattr(backend, "_metadata_max_size", None)
    if isinstance(value, int) and value > 0:
        return value
    return LMCACHE_GDS_METADATA_BYTES


def _chain_prepare(
    original_callback: Optional[Callable[[Any], None]],
    store: DiskUvmStore,
    kv_bytes: int,
) -> Callable[[Any], None]:
    """Wrap a GDS write-completion callback to prepare the backing.

    The chunk size (``kv_bytes``) is captured at ``submit_put_task`` time,
    BEFORE the installed GdsBackend runs ``memory_obj.ref_count_down()`` and the
    object may be recycled; the completion callback only ever sees that int.
    Preparation happens only after the original callback succeeds (the sole
    write-completion evidence).  Preparation errors are swallowed so they never
    abort serving; the demand get then simply falls back to the stock read.
    """

    def _prepare(completed_key: Any) -> None:
        try:
            store.prepare(completed_key, kv_bytes)
        except Exception:
            pass

    if original_callback is None:
        return _prepare

    def chained(completed_key: Any) -> None:
        original_callback(completed_key)  # propagate if the write is not complete
        _prepare(completed_key)

    return chained


def install_backend(
    backend: Any,
    *,
    managed: Optional[ManagedRange] = None,
    dispatch: Optional[IoctlDispatch] = None,
    candidate_fds: Optional[Sequence[int]] = None,
    poll_us: int = 1000,
) -> DiskUvmStore:
    """Additively wire the disk-UVM backing into an existing GDS backend.

    - put -> immutable completion -> offload: ``submit_put_task``'s completion
      callback prepares the backing once and caches it.
    - get -> restore: ``_load_gds`` (the single real GDS read seam) is routed
      through the retained backing when the correlated key has a prepared one,
      into the actual device destination; otherwise the stock GDS read runs
      unchanged.  The key is correlated thread-locally by the
      ``_load_bytes_from_disk_with_memory`` wrapper (the real read caller).

    The existing put/get and the stock/native/bpf admission path are untouched,
    so the native/BPF comparison is preserved.  Idempotent per backend.
    """
    existing = getattr(backend, _BACKEND_ATTR, None)
    if existing is not None:
        return existing

    managed = managed or _real_managed_range()
    store = DiskUvmStore(
        backend,
        managed=managed,
        dispatch=dispatch,
        candidate_fds=candidate_fds,
        poll_us=poll_us,
        file_offset=_gds_metadata_bytes(backend),
    )

    original_load_mem = backend._load_bytes_from_disk_with_memory
    original_load_gds = backend._load_gds
    original_submit = backend.submit_put_task

    def load_bytes_with_memory(key: Any, path: str, memory_obj: Any) -> Any:
        _THREAD_TLS.key = key
        try:
            return original_load_mem(key, path, memory_obj)
        finally:
            _THREAD_TLS.key = None

    def load_gds(gds_path: str, file_offset: int, gpu_pointer: Any,
                 size_in_bytes: int, dev_offset: int) -> int:
        key = getattr(_THREAD_TLS, "key", None)
        if key is not None:
            backing = store.lookup(key)
            if backing is not None and backing.kv_bytes == size_in_bytes:
                dest = int(gpu_pointer.value) + dev_offset
                try:
                    backing.restore(dest)
                    store.count("restored")
                    return size_in_bytes
                except Exception:
                    store.count("error")  # fall back to the stock GDS read
        store.count("stock_fallback")
        return original_load_gds(gds_path, file_offset, gpu_pointer,
                                 size_in_bytes, dev_offset)

    def submit_put_task(key: Any, memory_obj: Any,
                        on_complete_callback: Optional[Callable[[Any], None]] = None) -> Any:
        # Capture the logical KV byte size now, while ``memory_obj`` is still
        # live: the installed GdsBackend ref_count_downs it before the
        # completion callback fires, so the callback must never re-read it.
        try:
            kv_bytes = int(memory_obj.get_size())
        except Exception:
            kv_bytes = 0
        if kv_bytes > 0:
            chained = _chain_prepare(on_complete_callback, store, kv_bytes)
        else:
            chained = on_complete_callback
        return original_submit(key, memory_obj, on_complete_callback=chained)

    backend._load_bytes_from_disk_with_memory = load_bytes_with_memory
    backend._load_gds = load_gds
    backend.submit_put_task = submit_put_task
    setattr(backend, _BACKEND_ATTR, store)
    return store


def build_chunk_from_key(
    backend: Any,
    key: Any,
    *,
    managed: Optional[ManagedRange] = None,
    dispatch: Optional[IoctlDispatch] = None,
    candidate_fds: Optional[Sequence[int]] = None,
) -> DiskUvmBacking:
    """Build a standalone :class:`DiskUvmBacking` for one real completed chunk.

    Resolves the on-disk path from the installed GDS backend (``_key_to_path``)
    and the KV region length from the real file size.  For explicit, non-serving
    use; the serving path uses :func:`install_backend`.
    """
    path = backend._key_to_path(key)[0]
    return DiskUvmBacking(
        path,
        file_offset=_gds_metadata_bytes(backend),
        managed=managed or _real_managed_range(),
        dispatch=dispatch,
        candidate_fds=candidate_fds,
    )


def _real_managed_range() -> ManagedRange:
    """The CUDA managed-range implementation, imported only when actually run on
    a GPU host; never imported at module import time."""
    from lmcache_diskuvm_backing_cuda import CudaManagedRange

    return CudaManagedRange()


MODE_ENV = "LMCACHE_DISK_UVM_PROMOTION"
DIAG_OUT_ENV = "LMCACHE_DISK_UVM_DIAG_OUT"
_EXPECTED_LMCACHE_VERSION = "0.5.4"


def bootstrap_from_env(
    environ: Optional[Dict[str, str]] = None,
) -> Optional[Callable[..., None]]:
    """Patch future GDS backend instances when the opt-in env is set.

    Default off.  When ``LMCACHE_DISK_UVM_PROMOTION`` is an opt-in value, each
    created ``GdsBackend`` gets the additive disk-UVM put-completion preparation
    and cached-key read routing.  This never changes the existing put/get or the
    stock/native/bpf admission path.  Returns the installed class hook, or
    ``None`` when disabled.
    """
    import functools

    env = os.environ if environ is None else environ
    mode = env.get(MODE_ENV, "").strip().lower()
    if mode in {"", "off", "0", "disabled", "none"}:
        return None
    if mode not in {"1", "on", "true", "yes"}:
        raise ValueError(f"{MODE_ENV} must be off or an opt-in value, got {mode!r}")

    try:
        from importlib.metadata import version as _pkg_version

        installed = _pkg_version("lmcache")
    except Exception:
        installed = ""
    if installed and installed != _EXPECTED_LMCACHE_VERSION:
        raise RuntimeError(
            f"disk-UVM backing requires LMCache {_EXPECTED_LMCACHE_VERSION}, "
            f"found {installed}"
        )

    # Surface the transport counters at process exit when a diagnostics path is
    # configured, so a silent stock fallback is never mistaken for working UVM.
    if DIAG_OUT_ENV in os.environ:
        atexit.register(_diagnostics_atexit)

    from lmcache.v1.storage_backend.gds_backend import GdsBackend

    hook_attr = "_disk_uvm_class_hook"
    prior = getattr(GdsBackend, hook_attr, None)
    if prior is not None:
        return prior[1]

    original_init = GdsBackend.__init__

    @functools.wraps(original_init)
    def hooked_init(instance: Any, *args: Any, **kwargs: Any) -> None:
        original_init(instance, *args, **kwargs)
        install_backend(instance)

    GdsBackend.__init__ = hooked_init
    setattr(GdsBackend, hook_attr, (mode, hooked_init))
    return hooked_init


def backing_diagnostics() -> Dict[str, Any]:
    """Aggregate the transport counters across every installed store in this
    process.  Cheap and safe to call from the runner after teardown; it is the
    one-line signal that disk-UVM actually restored reads rather than silently
    taking the stock GDS read."""
    aggregate: Dict[str, int] = {}
    retained_total = 0
    for store in list(_ALL_STORES):
        desc = store.describe()
        retained_total += int(desc.pop("retained", 0))
        for name, value in desc.items():
            aggregate[name] = aggregate.get(name, 0) + int(value)
    aggregate["retained_total"] = retained_total
    return aggregate


def write_diagnostics(path: str) -> str:
    """Write the aggregate counters as a small JSON document; returns the path."""
    doc = {
        "schema": 1,
        "kind": "lmcache-disk-uvm-promotion",
        "counters": backing_diagnostics(),
    }
    with open(path, "w") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def _diagnostics_atexit() -> None:
    """Atexit dump of the aggregate counters (registered by
    :func:`bootstrap_from_env` when ``LMCACHE_DISK_UVM_DIAG_OUT`` is set)."""
    path = os.environ.get(DIAG_OUT_ENV)
    if not path:
        return
    try:
        counters = backing_diagnostics()
        write_diagnostics(path)
    except Exception:  # never let the dump break a clean serving exit
        return
    _log.info(
        "disk-uvm counters: prepared=%s restored=%s stock_fallback=%s "
        "not_fitted=%s not_aligned=%s error=%s retained=%s",
        counters.get("prepared", 0), counters.get("restored", 0),
        counters.get("stock_fallback", 0), counters.get("not_fitted", 0),
        counters.get("not_aligned", 0), counters.get("error", 0),
        counters.get("retained_total", 0),
    )


abi_layout_self_check()

# With gds-control on PYTHONPATH, importing this module can be the sole opt-in.
if MODE_ENV in os.environ:
    bootstrap_from_env()
