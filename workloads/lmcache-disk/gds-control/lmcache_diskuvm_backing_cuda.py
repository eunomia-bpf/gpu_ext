"""CUDA managed-range implementation for :mod:`lmcache_diskuvm_backing`.

Imported only when actually running on a GPU host (via
``lmcache_diskuvm_backing._real_managed_range``); never imported at module
import of the CPU-only backing module.  Root owns the GPU build/run.

Allocation contract (matches the tested primitive ``disk-uvm/disk_uvm_perf.cu``)
-------------------------------------------------------------------------------
The driver attaches a disk backing to the **whole** managed range
(``uvm_api_disk_backing_register``); there is no sub-range registration in this
revision.  The start VA must therefore be 2 MiB-aligned.  We call
``cudaMallocManaged`` with the **exact** ``kv_bytes`` (no overallocation), check
the returned start is block-aligned, and when it is not we fall back to the
stock GDS read (no new ABI, no file extension, no interior-range registration).
The 256 MiB primitive allocation received an aligned start; smaller
allocations are not guaranteed to be, hence the check.
"""

from __future__ import annotations

import ctypes
import os
from typing import Any, Dict, Optional, Tuple

_UVM_VA_BLOCK = 1 << 21
_CUDA_MEM_ATTACH_GLOBAL = 1
_CUDA_COPY_H2D = 1
_CUDA_COPY_D2H = 2
_CUDA_COPY_D2D = 3


class DiskUvmAlignmentError(RuntimeError):
    """The exact-size managed allocation is not 2 MiB block-aligned, so this
    driver revision cannot attach a disk backing to it; the chunk falls back to
    the stock GDS read."""


class DiskUvmFaultLibError(RuntimeError):
    """The fault-inducing kernel shared object (``libdiskuvm_fault.so``) is not
    available; the demand get degrades to the stock GDS read."""


def _load_cudart() -> ctypes.CDLL:
    candidates = ("libcudart.so", "libcudart.so.13", "libcudart.so.12")
    last_error: Optional[OSError] = None
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError as error:
            last_error = error
    raise OSError(f"cannot load libcudart: {last_error}")


def _load_fault_lib() -> ctypes.CDLL:
    here = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.get("LMCACHE_DISK_UVM_FAULT_LIB")
    candidates = [env] if env else []
    candidates.append(os.path.join(here, "libdiskuvm_fault.so"))
    candidates.append("libdiskuvm_fault.so")
    last_error: Optional[OSError] = None
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
        except OSError as error:
            last_error = error
            continue
        lib.diskuvm_fault_read.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        lib.diskuvm_fault_read.restype = ctypes.c_int
        return lib
    raise DiskUvmFaultLibError(
        f"cannot load libdiskuvm_fault.so ({last_error}). Build it with: "
        f"nvcc -shared -arch=sm_120 -Xcompiler -fPIC -o libdiskuvm_fault.so "
        f"diskuvm_fault.cu -lcudart"
    )


def _check(error_code: int, what: str) -> None:
    if error_code != 0:
        raise RuntimeError(f"{what}: CUDA error {error_code}")


class CudaManagedRange:
    """Real CUDA managed-range allocator + CPU-staged restore primitives."""

    def __init__(self, device: int = 0) -> None:
        self._cudart = _load_cudart()
        self._cudart.cudaSetDevice.argtypes = [ctypes.c_int]
        self._cudart.cudaMallocManaged.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        self._cudart.cudaFree.argtypes = [ctypes.c_void_p]
        self._cudart.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._device = int(device)
        self._allocs: Dict[int, int] = {}  # aligned start -> exact allocation base
        self._alloc_size: Dict[int, int] = {}  # aligned start -> allocated size
        self._fault_lib: Optional[ctypes.CDLL] = None
        _check(self._cudart.cudaSetDevice(self._device), "cudaSetDevice")

    # -------------------------------------------------------------- lifecycle --
    def alloc(self, size: int) -> int:
        """Allocate an **exact** ``size``-byte managed range and require it to be
        2 MiB block-aligned; raise :class:`DiskUvmAlignmentError` otherwise."""
        if size <= 0 or size % _UVM_VA_BLOCK != 0:
            raise ValueError(f"managed alloc must be a VA-block multiple: {size}")
        ptr = ctypes.c_void_p()
        _check(
            self._cudart.cudaMallocManaged(
                ctypes.byref(ptr), ctypes.c_size_t(size),
                ctypes.c_uint(_CUDA_MEM_ATTACH_GLOBAL),
            ),
            "cudaMallocManaged",
        )
        base = int(ptr.value)
        if base % _UVM_VA_BLOCK != 0:
            _check(self._cudart.cudaFree(ptr), "cudaFree (unaligned rollback)")
            raise DiskUvmAlignmentError(
                f"cudaMallocManaged({size}) returned 0x{base:x}, not aligned to "
                f"the {_UVM_VA_BLOCK}-byte UVM VA block; no sub-range "
                f"registration exists in this driver revision"
            )
        self._allocs[base] = base
        self._alloc_size[base] = size
        return base

    def free(self, va: int) -> None:
        base = self._allocs.pop(va, None)
        if base is None:
            return
        self._alloc_size.pop(va, None)
        _check(self._cudart.cudaFree(ctypes.c_void_p(base)), "cudaFree")

    # ----------------------------------------------------------------- moves --
    def populate(self, va: int, src_bytes: bytes) -> None:
        """Host->managed copy of the durable bytes (explicit CPU staging)."""
        host = ctypes.create_string_buffer(src_bytes)
        _check(
            self._cudart.cudaMemcpy(ctypes.c_void_p(va), host,
                                    ctypes.c_size_t(len(src_bytes)),
                                    ctypes.c_int(_CUDA_COPY_H2D)),
            "cudaMemcpy (populate H2D)",
        )

    def snapshot(self, va: int, size: int) -> bytes:
        host = (ctypes.c_uint8 * size)()
        _check(
            self._cudart.cudaMemcpy(host, ctypes.c_void_p(va),
                                    ctypes.c_size_t(size),
                                    ctypes.c_int(_CUDA_COPY_D2H)),
            "cudaMemcpy (snapshot D2H)",
        )
        return bytes(host)

    def gpu_fault_read(self, va: int, size: int) -> int:
        """Trigger a GPU fault over ``[va, va+size)`` via the companion kernel.

        While the range is sealed, on-disk, and GPU-promotion-enabled, this
        drives the driver's CPU-staged disk->GPU restore.  The fault library is
        loaded lazily so the put/offload path never requires it.
        """
        if self._fault_lib is None:
            self._fault_lib = _load_fault_lib()
        rc = int(self._fault_lib.diskuvm_fault_read(
            ctypes.c_size_t(va), ctypes.c_size_t(size),
            ctypes.c_int(self._device)))
        if rc != 0:
            raise RuntimeError(f"diskuvm_fault_read failed with code {rc}")
        return size

    def copy_to(self, va: int, size: int, dest: int) -> int:
        """Device-to-device copy of ``[va, va+size)`` into ``dest`` (a device
        pointer). This follows the driver's CPU-staged hydration; the copy
        itself is D2D, not an NVMe-to-GPU P2P transfer."""
        _check(
            self._cudart.cudaMemcpy(ctypes.c_void_p(int(dest)),
                                    ctypes.c_void_p(va),
                                    ctypes.c_size_t(size),
                                    ctypes.c_int(_CUDA_COPY_D2D)),
            "cudaMemcpy (restore D2D)",
        )
        return size
