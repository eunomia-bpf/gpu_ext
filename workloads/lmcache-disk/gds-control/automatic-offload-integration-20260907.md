# Automatic disk-backed KV offload: implementation target

This is the September 7 implementation direction, not a completed feature or
new performance result. It extends the existing real LMCache/cuFile storage
path; it does not restart completed baseline campaigns or change the paper.

## Target and ownership

Manage KV residency, backing copies, and pending storage operations together.
After registration, the runtime should automatically coordinate writeback,
reclaim, prefetch, and restore. BPF selects actions; LMCache/cuFile transports
bytes; vLLM owns live KV blocks and the point at which consumers may use them.
The current driver does not provide transparent same-address SSD paging.
See `disk-uvm-source-boundary-20260907.md` for the inspected 575 source boundary.

Residency and backing validity are separate facts: an object can have both
an HBM copy and a valid disk copy. A pending write is not a valid disk copy.
For the non-recompute route, release follows successful write completion and
the end of active source use. Restore must complete before its consumer runs.
A recomputable object may instead be discarded with an explicit recomputation
route; this must not be mislabeled as disk restoration.

## Current parallel implementation

- The vLLM victim-selection seam is committed in `e76a03aa`, but not installed.
  It keeps actual freeing and request rescheduling in upstream vLLM.
- Qwen 27B is implementing a bounded native/BPF candidate-selection interface,
  with a driver patch artifact and bindings. These files are still in progress.
- Qwen Next session `ses_f82a511f5ffeEc2docfRXrY3kU` repairs the real request-to-disk
  registry. Its predecessor's normal completion did not establish usability:
  `enable()` recursively acquires a non-reentrant lock, and the warm capture
  incorrectly interprets the boolean `retrieve()` return as KV chunks.
- GLM session `ses_f83256701ffek3qZigZd1Fj2qL` repairs completed-event ownership
  after async consumption. Existing attempt-02 data and its negative reference
  count warnings remain retained; no repaired-version result exists yet.

## Integration details that affect the algorithm

Use actual contiguous disk-backed prefixes, not the assumption that every
computed token is saved. Capture the engine's internal processed-chunk tuples,
and confirm disk backing through successful GDS writes or actual backend
presence. CPU-cache hits alone do not establish disk backing.

Price recovery using measured read cost and computed-token recompute cost.
Write duration is not a read estimate. Actual transfer bytes and freeable KV
bytes are distinct: shared KV blocks may not be reclaimable even when the
disk object contains their data. Do not clamp read bytes to freeable bytes.

Keep the stock victim's priority class; vLLM's priority convention must not
be silently inverted. The kernel validates the returned candidate and allowed
action, but must not hard-code the candidate-cost algorithm into its enforcer.
That algorithm belongs to the matched native and BPF policy implementations.

## Next real serving comparison

Compare stock victim selection, native disk-aware selection, and the same BPF
selection under identical arrivals, generation lengths, and KV capacity that
actually causes preemption. Attempt 02 had an 8192-token KV pool and at most
two short running sequences; repeating it unchanged cannot measure reclaim.

Report serving throughput, TTFT and end-to-end latency alongside preemptions,
disk bytes and recomputation. Retain adverse results. A selector-only result
does not establish automatic offload: request association, actual block reclaim,
write completion, and the selected recovery route still need to be connected.
Freeing a staging buffer or returning KV blocks to a fixed pool is not evidence
of releasing physical HBM to other processes.
