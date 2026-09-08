# Registered disk-backed UVM implementation

The user's target is automatic disk offload tied to UVM residency and
gpubpf asynchronous decisions, with demand restoration at the original
virtual address. Existing LMCache fixed-pool recycling and command-82 I/O
decisions do not implement that target.

## Current implementation assignment

Qwen 27B through local OpenCode owns driver development in the existing
`gpu_ext-kernel-575-gds` checkout, branch
`revision/gpu-storage-decision-575`. It may build but must not install or
reload the driver while the independent LMCache performance campaign runs.
Qwen Next has a separate read-only CPU-fault integration assignment.
The existing GLM session is finishing the previous victim-policy analysis.
At most three local OpenCode sessions run; no native subagents are used.

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
