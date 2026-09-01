# Preflight 1: PASS

The approved two-cell semantic preflight completed on the RTX 5090 with the
custom NVIDIA 610.43.02 UVM module. Monitor and tracer timing is excluded from
all performance analysis.

## Native no-prefetch

- Module parameter: `uvm_perf_prefetch_enable=0`; no memory struct_ops attached.
- Workload: 8 GiB allocation, 64 KiB regions, 131,072 unique demand addresses.
- Result: 439.912 ms, zero mismatches.
- Final UVM events: 129,832 migrations, 531,791,872 migrated bytes, zero
  prefetch migrations, zero prefetch bytes, zero dropped migration events.

## gpubpf no-prefetch

- Module parameter: `uvm_perf_prefetch_enable=1`; owned empty-region `BYPASS`
  policy emitted a ready record before workload release.
- Workload: the same 8 GiB allocation, region layout, and kernel.
- Result: 408.164 ms, zero mismatches.
- Hook coverage, counted only after CPU initialization and monitor readiness:
  131,072 wrapper calls and 131,072 helper calls.
- Final UVM events: 129,929 migrations, 532,189,184 migrated bytes, zero
  prefetch migrations, zero prefetch bytes, zero dropped migration events.
- Cleanup: the owned loader emitted its detaching record and exited zero; no
  memory struct_ops remained; UVM refcount was zero.

## Admission decision

PASS. Both cells generated real migration activity while producing the same
scoped no-prefetch outcome and exact numerical output. The BPF cell covered all
131,072 expected fault regions with matching wrapper/helper counts. This single
instrumented pair is a semantic and engagement control, not a retained timing
sample.

No file/content hashes, checksums, or digests were generated, refreshed,
compared, or recorded for this preflight.
