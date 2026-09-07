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

## Execution update, September 7, 19:58 UTC

The consumed-event repair and its complete 20-cell comparison are published
in `8987a14b` / `c3c6d42e`; the warning disappears but the deadline-prefetch
policy still loses to demand FIFO. Those completed cells will not be repeated.
Root installed the committed vLLM selection seam after that campaign ended;
installation and the preserved original are recorded in `8d7493e5`.

The registry's Qwen Next request ended with an actual gateway API 524 after
automatic retries. Root confirmed the terminal CLI and absent live session,
then resumed the same task/session with GLM. This was not termination for
silence or an artificial execution deadline. The selector, registry repair,
and serving integration remain the three live local tasks; the last two now
use GLM because the Next request failed. No new reclaim performance is claimed.

Root's first ordinary selector build passed for the native shared library but
failed for BPF. The current `kv_reclaim_abi.h` overflow check in
`uvm_kv_reclaim_mul_hi` is optimized into unsupported `__multi3` calls by the
BPF compiler. The BPF file also places its `preserve_access_index` pragma
after the shared record definitions, producing an unused-attribute warning.
Both diagnostics were returned to the selector owner; no driver load or
performance claim follows from the native-only build. Build commands from
the repository root (outputs stay in the existing temporary task directory):

```sh
cc -O2 -g -Wall -Wextra -fPIC -shared workloads/lmcache-disk/gds-control/kv_reclaim_native.c -o /tmp/opencode/kv_reclaim_native-build-check.so
clang -g -O2 -target bpf -D__TARGET_ARCH_x86 -Ivmlinux/x86 -Ilibbpf/src -c workloads/lmcache-disk/gds-control/kv_reclaim_policy.bpf.c -o /tmp/opencode/kv_reclaim_policy-build-check.bpf.o
```

## Selector build follow-up, September 7

The local implementation replaced the overflow helper with explicit 32-bit
limb multiplication and updated its comparison call sites. Root rebuilt the
actual `kv_reclaim_policy.bpf.c` above successfully, with no compiler output.
The native shared library also builds with
`-Werror=implicit-function-declaration -Wl,-z,defs`, so the previously missing
helper is no longer left as an unresolved symbol. These are compilation and
link results, not a loaded-driver or serving-performance result.

The backing registry now captures the installed GDS read path separately from
write completion; ordinary Python compilation of the registry and selector
binding passes. Complete backing coverage and actual object-transfer byte
accounting are still being integrated with the serving consumer. All three
local OpenCode sessions remain live; no extra session or execution timeout
was introduced. The driver patch and real reclaim runner are not yet ready
for a new performance campaign. No paper files were changed.
