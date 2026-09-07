# SPDX-License-Identifier: GPL-2.0
"""Thin ctypes binding for the real KV reclaim candidate selector.

Exposes the three decision arms of the shared bounded-integer policy
(kv_reclaim_abi.h) for later serving integration, without any benchmark:

  KvReclaimNative.choose(...)   native C entrypoint (kv_reclaim_native.so),
                                the stock/native arm;
  KvReclaimKernel.choose(...)   UVM ioctl 83 (UVM_KV_RECLAIM_CHOOSE) against
                                an open /dev/nvidia-uvm fd; returns the stock
                                victim when no gpu_kv_reclaim_ops BPF policy
                                is attached.

A candidate is a dict with keys cookie, freeable_bytes, computed_tokens,
disk_backed_tokens, disk_backed_bytes, priority, flags, or the same fields
as a 7-tuple. Semantics:
  - priority keeps vLLM integer semantics: LOWER value is MORE important;
    the worst class (the only eligible one) is the maximum value;
  - disk_backed_bytes is the provider's actual backed transfer bytes, not
    clamped to freeable_bytes (freeable blocks exclude shared/referenced
    blocks while the disk transfer may restore full backing objects);
  - FLAG_COVERAGE_KNOWN marks trusted disk-backed counts; without it the
    disk-backed prefix is priced as zero (unknown coverage is NOT a fully
    backed request).
The kernel returns the stock victim (route STOCK) on unknown inputs; only
fixed-width scalar metadata crosses the boundary - no fd, file offset, GPU
pointer, stream, or completion. The kernel kfunc validates index/cookie/
eligible class and route range only; the shared bounded-integer algorithm
belongs to the matched native/BPF policies, not to kernel validation.
"""

import ctypes
import fcntl
from pathlib import Path

ABI_VERSION = 1
MAX_CANDIDATES = 8
MAX_PRIORITY = 7
FLAG_COVERAGE_KNOWN = 0x1
FLAGS_ALL = FLAG_COVERAGE_KNOWN

ROUTE_STOCK = 0
ROUTE_FULL_RECOMPUTE = 1
ROUTE_DISK_PREFIX = 2
_ROUTE_NAMES = {
    ROUTE_STOCK: "stock",
    ROUTE_FULL_RECOMPUTE: "full_recompute",
    ROUTE_DISK_PREFIX: "disk_prefix",
}

NV_OK = 0
UVM_KV_RECLAIM_CHOOSE = 83  # UVM_IOCTL_BASE(83)


def _route_name(route):
    return _ROUTE_NAMES.get(route, "route_%d" % route)


def _as_fields(candidate):
    if isinstance(candidate, dict):
        return (
            candidate.get("cookie", 0),
            candidate.get("freeable_bytes", 0),
            candidate.get("computed_tokens", 0),
            candidate.get("disk_backed_tokens", 0),
            candidate.get("disk_backed_bytes", 0),
            candidate.get("priority", 0),
            candidate.get("flags", 0),
        )
    return tuple(candidate)


class KvReclaimNative:
    """Native decision entrypoint (kv_reclaim_native.so), same algorithm
    as the BPF policy."""

    def __init__(self, lib_path=None):
        here = Path(__file__).resolve().parent
        path = lib_path or (here / "kv_reclaim_native.so")
        self._lib = ctypes.CDLL(str(path))
        self._lib.kv_reclaim_native_choose.restype = ctypes.c_int
        self._lib.kv_reclaim_native_choose.argtypes = [
            ctypes.c_uint32,  # abi_version
            ctypes.c_uint32,  # n_candidates
            ctypes.c_uint32,  # stock_index
            ctypes.c_uint64,  # disk_read_ns_per_kib
            ctypes.c_uint64,  # recompute_ns_per_token
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]

    def choose(self, candidates, disk_read_ns_per_kib, recompute_ns_per_token,
               stock_index=0):
        """Run the shared selection. Returns dict(status, index, route,
        route_name, cookie, estimated_ns); status 1 means hard-invalid
        input and the outputs carry the stock defaults."""
        n = len(candidates)
        fields = [_as_fields(c) for c in candidates]
        cookie = (ctypes.c_uint64 * MAX_CANDIDATES)(
            *[f[0] for f in fields] + [0] * (MAX_CANDIDATES - n))
        freeable = (ctypes.c_uint64 * MAX_CANDIDATES)(
            *[f[1] for f in fields] + [0] * (MAX_CANDIDATES - n))
        computed = (ctypes.c_uint64 * MAX_CANDIDATES)(
            *[f[2] for f in fields] + [0] * (MAX_CANDIDATES - n))
        backed_tok = (ctypes.c_uint64 * MAX_CANDIDATES)(
            *[f[3] for f in fields] + [0] * (MAX_CANDIDATES - n))
        backed_bytes = (ctypes.c_uint64 * MAX_CANDIDATES)(
            *[f[4] for f in fields] + [0] * (MAX_CANDIDATES - n))
        priority = (ctypes.c_uint32 * MAX_CANDIDATES)(
            *[f[5] for f in fields] + [0] * (MAX_CANDIDATES - n))
        flags = (ctypes.c_uint32 * MAX_CANDIDATES)(
            *[f[6] for f in fields] + [0] * (MAX_CANDIDATES - n))
        out_index = ctypes.c_uint32(0)
        out_route = ctypes.c_uint32(0)
        out_cookie = ctypes.c_uint64(0)
        out_est = ctypes.c_uint64(0)
        status = self._lib.kv_reclaim_native_choose(
            ABI_VERSION, n, stock_index,
            int(disk_read_ns_per_kib), int(recompute_ns_per_token),
            cookie, freeable, computed, backed_tok, backed_bytes,
            priority, flags,
            ctypes.byref(out_index), ctypes.byref(out_route),
            ctypes.byref(out_cookie), ctypes.byref(out_est))
        return {
            "status": status,
            "index": out_index.value,
            "route": out_route.value,
            "route_name": _route_name(out_route.value),
            "cookie": out_cookie.value,
            "estimated_ns": out_est.value,
        }


class _KvReclaimParams(ctypes.Structure):
    """Mirror of UVM_KV_RECLAIM_CHOOSE_PARAMS (uvm_ioctl.h)."""

    _fields_ = [
        ("abiVersion", ctypes.c_uint32),
        ("nCandidates", ctypes.c_uint32),
        ("stockIndex", ctypes.c_uint32),
        ("pad0", ctypes.c_uint32),
        ("diskReadNsPerKib", ctypes.c_uint64),
        ("recomputeNsPerToken", ctypes.c_uint64),
        ("cookie", ctypes.c_uint64 * MAX_CANDIDATES),
        ("freeableBytes", ctypes.c_uint64 * MAX_CANDIDATES),
        ("computedTokens", ctypes.c_uint64 * MAX_CANDIDATES),
        ("diskBackedTokens", ctypes.c_uint64 * MAX_CANDIDATES),
        ("diskBackedBytes", ctypes.c_uint64 * MAX_CANDIDATES),
        ("priority", ctypes.c_uint32 * MAX_CANDIDATES),
        ("flags", ctypes.c_uint32 * MAX_CANDIDATES),
        ("selectedIndex", ctypes.c_uint32),
        ("selectedRoute", ctypes.c_uint32),
        ("selectedCookie", ctypes.c_uint64),
        ("estimatedRecoveryNs", ctypes.c_uint64),
        ("rmStatus", ctypes.c_uint32),
    ]


class KvReclaimKernel:
    """UVM ioctl 83 arm. Takes an already-open /dev/nvidia-uvm fd; the
    kernel returns the stock victim when no gpu_kv_reclaim_ops policy is
    attached or the input is unknown/insufficient."""

    def __init__(self, uvm_fd):
        self._fd = int(uvm_fd)

    def choose(self, candidates, disk_read_ns_per_kib, recompute_ns_per_token,
               stock_index=0):
        n = len(candidates)
        p = _KvReclaimParams()
        p.abiVersion = ABI_VERSION
        p.nCandidates = n
        p.stockIndex = stock_index
        p.pad0 = 0
        p.diskReadNsPerKib = int(disk_read_ns_per_kib)
        p.recomputeNsPerToken = int(recompute_ns_per_token)
        for i, c in enumerate(candidates):
            f = _as_fields(c)
            p.cookie[i] = f[0]
            p.freeableBytes[i] = f[1]
            p.computedTokens[i] = f[2]
            p.diskBackedTokens[i] = f[3]
            p.diskBackedBytes[i] = f[4]
            p.priority[i] = f[5]
            p.flags[i] = f[6]
        fcntl.ioctl(self._fd, UVM_KV_RECLAIM_CHOOSE, p)
        return {
            "status": p.rmStatus,
            "index": p.selectedIndex,
            "route": p.selectedRoute,
            "route_name": _route_name(p.selectedRoute),
            "cookie": p.selectedCookie,
            "estimated_ns": p.estimatedRecoveryNs,
        }
