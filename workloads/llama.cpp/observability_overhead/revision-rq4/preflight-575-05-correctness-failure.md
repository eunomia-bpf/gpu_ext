# RTX 5090 Table 1 preflight 575-05: retained correctness failure

Date: 2026-09-03 (America/Vancouver)

This attempt used `gpu_ext` at `37ebc90` and the bpftime Table 1 branch at
`b1bf699`.  It passed admission and built NVBit plus all three freshly copied
gpubpf tools.  All seven correctness arms executed on the RTX 5090 with NVIDIA
575.57.08, and every application arm produced the same expected 47-byte
output.  Because three probe gates failed, the runner correctly skipped every
performance cell.

## Valid controls

- Baseline correctness passed.
- NVBit kernel-exit collection observed 720,896 events and 220 launches.
- NVBit and gpubpf thread histograms each observed 720,896 events and 22,528
  nonzero threads.  The gpubpf arm completely read back 1,048,576 entries
  (8,388,608 bytes).

## Failed gates

- gpubpf kernel-exit collection retained all 720,896 events with zero drops and
  the exact expected multiplicity totals, but its order-sensitive coordinate
  segmentation check reported 3,808 mismatches.  The loader therefore exited
  1.  This must be resolved by proving the event-order contract or replacing
  the invalid ordering assumption with an equally strict order-independent
  oracle; the count gate must not be weakened.
- gpubpf launch latency observed 220 device entries but zero host entries.  It
  reported 220 queue underflows and failed pairing.  The selected ELF uprobe
  address did not observe the corresponding host launch path.
- NVBit launch latency selected 220 launches but classified no samples and
  reported 220 clock errors.  Its current host/device timestamp comparison is
  not a valid calibrated latency measurement on this run.
- The new gpubpf clock calibration also found that the start/end offset
  intervals did not overlap.  A robust latency implementation must account for
  host/device clock-rate drift rather than treating one constant offset as a
  foregone result.

The correctness process was launched from a PTY and received EOF for each
interactive `llama-cli` arm; no timing result was collected or inferred from
those wall-clock durations.  Final checks found no compute applications, no
private `rq4_*` shared-memory objects, UVM reference count zero, empty
`struct_ops`, an idle 15 MiB GPU, and the original 400 W power limit.

The next attempt must use a fresh output directory after source fixes, retain
all seven arms, and pass these frozen correctness/engagement gates before any
performance measurement.
