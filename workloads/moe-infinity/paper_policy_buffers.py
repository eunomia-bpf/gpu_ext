"""Batch-copy the unchanged rank/match ABI; selection still executes in BPF."""
import ctypes

import numpy as np


class RankEntry(ctypes.Structure):
    _fields_ = [("identity", ctypes.c_uint64), ("score_bits", ctypes.c_uint64),
                ("ordinal", ctypes.c_uint32), ("reserved", ctypes.c_uint32)]


ENTRY_DTYPE = np.dtype([("identity", np.uint64), ("score_bits", np.uint64),
                       ("ordinal", np.uint32), ("reserved", np.uint32)], align=True)
if ENTRY_DTYPE.itemsize != ctypes.sizeof(RankEntry) or any(
    ENTRY_DTYPE.fields[name][1] != getattr(RankEntry, name).offset for name in ENTRY_DTYPE.names
):
    raise RuntimeError("NumPy/ctypes rank ABI layout disagreement")


def packed_entries(identities, scores):
    scores = np.ascontiguousarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(identities) != len(scores):
        raise ValueError("rank identity/score length mismatch")
    entries = np.zeros(len(scores), dtype=ENTRY_DTYPE)
    entries["identity"] = identities
    # Bitwise copy, not numeric float->integer conversion. Preserves NaN
    # payloads, infinities, signed zero and original input ordinal exactly.
    entries["score_bits"] = scores.view(np.uint64)
    entries["ordinal"] = np.arange(len(scores), dtype=np.uint32)
    return entries


def packed_select(run, identities, scores):
    entries = packed_entries(identities, scores)
    size = len(entries)
    indices = (ctypes.c_uint32 * size)()
    count = ctypes.c_uint32()
    # entries remains strongly owned throughout this synchronous native/JIT
    # call, including while ctypes releases the GIL. No pointer is retained.
    rc = run(entries.ctypes.data_as(ctypes.POINTER(RankEntry)), size,
             indices, size, ctypes.byref(count))
    if rc != 0 or count.value > size:
        raise RuntimeError("BPF rank execution failed; no fallback")
    result = list(indices[:count.value])
    if len(set(result)) != len(result) or any(i >= size for i in result):
        raise RuntimeError("BPF rank returned invalid indices")
    return result
