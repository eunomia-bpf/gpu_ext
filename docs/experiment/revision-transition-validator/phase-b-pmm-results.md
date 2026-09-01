# Transition validator Phase B PMM results

Date: 2026-08-31

Disposition: `PASS` for offline production integration, kernel-native test
construction, BPF fixture construction, and builds. Phase B remains `PARTIAL`
because neither the kernel-native ioctl nor the seven verifier fixtures has
run against a safely loaded custom module.

Independent review: [phase-b-pmm-review.md](phase-b-pmm-review.md).

## Production integration

Every PMM list mutation that can alias a root chunk now passes through one
lock-asserting production helper. A root records explicit driver-owned
membership and a generation in the same `list_lock` critical section as the
list operation. Cross-list and ownership transitions advance the generation;
same-list head/tail reorders do not. Proven subchunk-only mutations leave root
metadata unchanged. Policy validation never uses `list_empty()` as membership
evidence.

The former raw-list move interface is replaced by a callback-local 72-byte
decision context and
`bpf_gpu_request_reorder(decision_ctx, destination_u64, position_u64)`. The
setter records the first raw request, including an invalid one, and latches a
conflict on any different second request. It never mutates a list. The
post-callback production helper validates PMM/root identity, source,
generation, request width/range, repetition/conflict, and raw callback action
while `list_lock` remains held.

The access path implements the frozen action table: only a true no-request
`DEFAULT` reaches the native tail move; no-request `BYPASS` preserves entry
state; a legal request with `DEFAULT` or `BYPASS` commits exactly once and
suppresses native fallback; invalid action, invalid request, stale identity, or
conflict preserves entry state. Activate validation runs after the native move
and performs no second native mutation.

The 610 port maps the same contract onto its `UNUSED`, `DISCARDED`, and `USED`
allocation lists. `DISCARDED` is explicit driver state but is neither a legal
policy source nor destination. Its external lazy-free enqueue path is routed
through the same helper.

## Kernel-native executable test

`uvm_test_pmm_bpf_transition()` is registered through the existing NVIDIA UVM
test-ioctl path on both driver lines. It allocates isolated real `uvm_gpu_t`,
`uvm_pmm_gpu_t`, and `uvm_gpu_root_chunk_t` objects, initializes real
`uvm_spinlock_t` and Linux list heads, and calls the exact production list and
commit helpers while holding the production lock.

The test covers root and proven-subchunk aliases; no-request action routing;
same-list generation stability; cross-list generation advance; identical
repeat; cross-callback HEAD/TAIL reversal; invalid-only, valid-to-invalid,
invalid-to-valid, and identical-invalid-repeat under both `DEFAULT` and
`BYPASS`; invalid action with a legal request; stale generation/source;
foreign PMM/root identity; `FREE`, `EVICTION`, `NONE`, and `LAZY_FREE`; and
post-native activate behavior. The 610 version additionally exercises
`DISCARDED` as a rejected source.

This is not a host mock, but it has only been compiled into the modules. The
ioctl has not run on a loaded custom stack, so no runtime pass is claimed.

## BPF verifier-load fixtures

The focused loader now contains seven exact-ABI fixtures. Four positive
controls run first: immutable scheduler read, scheduler timeslice setter,
scheduler explicit-low interleave setter, and PMM reorder setter. Only if all
four load does it attempt the three direct-write negatives: scheduler input,
scheduler hidden state, and PMM hidden decision state.

The PMM negative object emits a direct 64-bit store at callback-context offset
56, inside the driver-owned request. The positive object passes the received
decision pointer to `bpf_gpu_request_reorder()` with `USED` and `HEAD` raw
values. Object BTF reports the decision structure at 72 bytes, and
disassembly shows the expected direct store or kfunc relocation.

The runner calls `bpf_object__load()` without attaching. It accepts a negative
outcome only when the load error is exactly `-EACCES` after all four positive
controls have loaded, preserves one raw verifier log per attempted fixture,
and requires exactly seven attempts, four admissions, three rejections, and
seven matching outcomes.

The local runner exits at its root precondition, and the running official
module does not export `/sys/kernel/btf/nvidia`. The display-owned module stack
was not replaced. Therefore the seven expected outcomes remain executable but
unobserved, and the live portion is `PARTIAL`.

## Offline evidence

- The production-header host test passes 12 cases and 145 assertions.
- All five 575 modules build against Linux 6.14 headers; its independently
  reviewed native-test batch is pushed on `test-sched`.
- All five 610 modules build against Linux 7.1.12 headers; its independently
  reviewed production and native-test batches are pushed on
  `port/nvidia-610.43.02`.
- The extension's complete active set builds. Source audit confirms all 60 PMM
  callback definitions use the decision-context ABI. The seven focused BPF
  objects and userspace loader build through the normal extension toolchain.
- Earlier fresh 610 module-BTF inspection confirmed the 72-byte decision
  context, typed reorder kfunc, explicit root state/generation, and absence of
  the old raw head/tail kfuncs. This is ABI evidence, not a live-load result.
