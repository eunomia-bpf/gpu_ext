# Reviewer-A raw-state evidence plan

## Existing implementation boundary

The sibling bpftime runtime already provides a device-visible per-thread GPU
ring buffer (`BPF_MAP_TYPE_GPU_RINGBUF_MAP`, type 1527), host polling, and exact
ring statistics for committed, collected, pending, out-of-range, full,
bad-size, other-drop, and dirty-slot states.  It also provides a per-GPU-thread
array (`BPF_MAP_TYPE_PERGPUTD_ARRAY_MAP`, type 1502).  Existing examples print
sampled coordinates or counters, but the revision checklist contains no
multi-scale exact comparison of raw device records against CUDA-generated
truth and no live overflow rejection gate.

## Evidence decision

Use the ring buffer for raw, non-aggregated coordinate/sequence tuples and the
array for a separate aggregated control.  Compare both against a finite CUDA
truth array in two positive geometries.  Add a deterministic capacity overflow
as a negative evidence gate, and repeat all three cells in five randomized
formal blocks after one complete preflight block.

This route tests the actual current ABI and does not require a new map type.
It also keeps the limit explicit: the host can consume arbitrary fixed-size
records within the configured capacity, while overflow is observable and must
invalidate completeness claims.

## Publication interpretation

If all positive cells pass, the paper may say that gpubpf supports raw
host-authoritative state in addition to composable reductions, with exact
bounded readback demonstrated at 256 and 2,048 active threads.  It must also
say that fixed-capacity queues can drop records and that the tested runtime
reports those drops fail-closed.  This result does not justify claims about
automatic hierarchy placement, on-chip shards, throughput, strict admission,
or arbitrary/unbounded data.

