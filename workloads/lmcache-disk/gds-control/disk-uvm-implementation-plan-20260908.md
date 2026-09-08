# Registered disk-backed UVM implementation

The user's target is automatic disk offload tied to UVM residency and
gpubpf asynchronous decisions, with demand restoration at the original
virtual address. Existing LMCache fixed-pool recycling and command-82 I/O
decisions do not implement that target.

## Current implementation assignment

Update at 08:07 UTC: the GLM request ended with an actual HTTP 524 after
automatic retries; its CLI exited naturally. Root resumed the same saved
session through cluster-direct Qwen 27B on coordinator port 40081, preserving
all partial source. The new request is busy and has executed source-inspection
tools. This was not a cancellation for silence or elapsed time. Its private
provider configuration disables the separate overall, response-header and
stream-chunk timeouts; it does not change global credentials or claim to
remove upstream errors. There remain three inference sessions, not four.
Heavy builds remain deferred while the original Fig.13 GPU campaign runs.

Historical update at 06:17 UTC: GLM through local OpenCode owns driver development in
the existing `gpu_ext-kernel-575-gds` checkout, branch
`revision/gpu-storage-decision-575`, session `ses_f80ce8d77ffexXOG7cehG0hvgg`.
Its previous turn ended naturally with `finish=stop` and CLI exit zero after
writing only a partial C file; root resumed the same unfinished task rather
than treating that exit as completion. It has now added registration,
reference release, bitmap status queries and real `kernel_read/kernel_write`
helpers with contiguous-page transfer coalescing. The initial lock expression,
reference-count API mismatch and header complete-type issue were corrected.
This is still unbuilt, unwired implementation work: asynchronous offload,
physical-copy release, same-VA fault restoration and ioctl/build integration
are not established. No new driver has been installed or loaded.

The previous LMCache performance campaign is complete; ordinary scoped CPU
builds are allowed, while root coordinates all GPU experiments and module
changes. The other two live tasks are automatic warp execution (GLM) and the
original Fig.13 runner (cluster-direct Qwen 27B). At most three local OpenCode
sessions run; no native subagents are used.

Historical update at 03:28 UTC: Qwen Next's CPU-path request ended with actual HTTP524
after its automatic retries, without analysis. A fresh GLM plan session
`ses_f80f29bf9ffeT5CaGizKZC3rJk` takes that unfinished read-only task; its
session permissions explicitly deny edit, shell execution and subagent
tasks. The previous LMCache analysis session was terminated for editing
and rebuilding the frozen control despite its read-only assignment;
its patch and the affected measurement are preserved separately in the
total-cost result report. Qwen 27B remains the sole driver-code owner.
There are now two live local sessions, not three; neither is stopped for
silence or elapsed time. Module builds remain deferred during GPU timing.

The smallest useful slice is an explicitly registered managed range with
a retained backing file, real asynchronous write/read transport, removable
physical memory copies after successful writeback, and same-address demand
restore. An explicitly sealed/read-only range may be the first supported
contract, but the implementation must say how mutation is handled; that
slice is not a claim of general writable-memory support. Registration or
write-only sidecar code alone does not complete this slice.

The initial transport uses CPU staging. Direct managed-memory GDS P2P is
not assumed. BPF offload/prefetch decisions can use this backing mechanism
once it exists; BPF itself does not perform filesystem I/O. Code, a build,
or this plan is not evidence of a functioning disk-backed pager.

## Source-supported starting points and corrections

The managed eviction branch of `uvm_va_block_evict_chunks` already moves
GPU-resident data to CPU memory and transfers copy dependencies into the
eviction tracker before marking GPU chunks evicted. The existing deferred
eviction-mapping queue shows a block retain/release pattern, but is not a
ready-made disk I/O engine.

Root inspected `uvm_gpu_replayable_faults.c:2333`: the
`NV_WARN_MORE_PROCESSING_REQUIRED` branch releases VA-space locks and
continues without advancing the current fault index. Contrary to the first
local analysis, this is not by itself an asynchronous fault parking/wakeup
API. Blindly returning that status can repeatedly retry the same fault.
The implementation must provide a real in-flight completion path outside
UVM locks and must also cover CPU faults and migration entry points, or
explicitly reject unsupported operations without returning zero-filled data.

CPU-chunk references provide lifetime, not an immutable snapshot. Existing
CPU dirty bits are not automatically disk-version validity. Writeback
completion must account for concurrent access, range splitting, teardown,
and ownership before removing the last memory copy. Physical GPU chunk
release may precede disk completion when an intact CPU staging copy remains;
disk-only state may not be published before the disk copy is valid.

## Handoff and next measurement

Root will review the actual diff and build output, commit and push each
completed implementation step, then run the real managed-range workload
after the current GPU campaign releases its locks. The workload must
exercise offload and access the same VA again, report disk transfer and
memory-footprint observations with their actual scope, and retain failures.
No new papers, paper edits, artificial timeouts, clock tests, or additional
preflight/audit campaigns are part of this work.
