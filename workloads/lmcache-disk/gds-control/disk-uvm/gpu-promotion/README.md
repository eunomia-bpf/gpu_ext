# Disk-UVM GPU-promotion implementation and retained candidate

The tested implementation is driver `dea1fefc`, client `e07b4d69`.
[Five completed runs](../../../results-disk-uvm-gpu-promotion-20260908.md)
report repeated GPU-read median 0.250566 ms versus the earlier CPU-first
5.407625 ms, but first restore is slower (210.570854 vs 170.601143 ms).
These separate campaigns do not establish an end-to-end LMCache speedup.

- `integrated-driver.patch`: the actual driver changes from `c8e2831d` to
  `dea1fefc`, including the no-resident early-return repair and ioctl bounds.
  This is the patch to apply to a clean `c8e2831d` source when reproducing it.
- `gpu-promotion.patch` and `initial-candidate-notes.md`: retained initial
  local-model candidate, **not the tested implementation**. Its disk staging
  was placed after an existing early return and could therefore be bypassed.
  Do not use that historical candidate for a live run.

The opt-in ioctl 87 is issued on the existing range-owning UVM descriptor
following registration, outside the restore timings. It keeps GPU residency
as the GPU fault target, hydrates disk pages into CPU staging, then uses the
normal CPU-to-GPU copy path. The flag defaults off; CPU restoration stays on
its existing path. This is explicit sealed/read-only same-VA restoration,
not automatic pressure-triggered offload, a new native/BPF policy comparison,
or established NVMe-to-GPU P2P. No extra clock or performance gate is added.

The local model authored the driver algorithm and staging repair. Root
integrated it, made the bounded client flag/ioctl edit, built, measured and
published. All old measurements and the first candidate are retained.
