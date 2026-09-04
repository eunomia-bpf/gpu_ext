# RTX 5090 Table 1 preflight 575-04: retained build failure

Date: 2026-09-03 (America/Vancouver)

This attempt used the committed revision runner at `a5ad340` and the bpftime
runtime branch through `b1bf699`.  Admission passed on the RTX 5090 with NVIDIA
575.57.08, an idle GPU, and no compute applications.  The NVBit tool built for
`sm_120`.

The campaign stopped before its first correctness or performance cell while
building the freshly copied gpubpf `kernelretsnoop` tool.  Its copied Makefile
retained the source-tree-relative include `../../../runtime/include`, which no
longer resolves from the per-attempt output directory.  Compilation therefore
failed at:

```text
kernelretsnoop.c:18:10: fatal error: bpftime_gpu_ringbuf.h: No such file or directory
```

No Table 1 measurement is claimed from this attempt.  The ignored generated
tool binaries remain in the local `raw/preflight-575-04` directory.  The
admission record, NVBit build log, and failing gpubpf build log are retained in
Git.  The next attempt must fix and test path freezing, use a new output
directory, and rebuild from committed sources.
