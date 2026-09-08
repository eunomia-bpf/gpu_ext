# XSched Level-2 sm_120 source checkpoint

Status: the device BPF guardian, sm_120 cubin and NVBit tool library are
**built**, but end-to-end Level-2 execution is **not measured**. The existing
Level-1 measurements and dependency checkout are unchanged. This is not a
completed reproduction of the original Level-2 system.

The 2026-09-08 component build compiled 49 eBPF instruction words; the
48-byte-context exporter accepted them with its existing GPU checks, emitted
device-callable PTX, and ptxas generated the sm_120 cubin. The tool library
linked successfully. See `../level2-build/build-20260908.kMcZTl/README.md`
and `branchless-barrier-components.log`. No verifier rule was disabled.
The native LDC adapter and HAL/end-to-end replay still require completion.

The source includes a device BPF guardian, shared native-C/BPF trampoline,
NVBit launch-context adapter, native probe/guardian/restore stubs, LDC
patcher, and an opt-in HAL patch. Component build commands are in
`../level2-build/`. The HAL patch passed a non-mutating `git apply --check`
against the local XSched source; it has not been applied to that checkout.

Integration status: the shared decision context is now a bounded 6 x u64
scalar snapshot (48 bytes, xsched_guardian_abi.h) materialized by the
trusted trampoline from the real device words; both the native-C decision
and the compiled eBPF program consume that same snapshot, and the BPF
program dereferences no device pointer. The reused SASS exporter is
adapted only through level2/bpf/bpf_to_ptx_ctx48.patch applied to a copy
under ../level2-build (.output/adapter), widening the strict PREVAIL
verify-time context from 8 to 48 bytes; granted size equals used size.
Still to build and exercise: the native sm_120 artifact path and
end-to-end replay. Both decision arms keep the same trusted actuator. No
host decision fallback or performance gain is claimed.

The compiled eBPF guardian is expressed branchless (straight-line u64
arithmetic, no conditional jumps): the strict PREVAIL GPU warp-uniform
branch verifier rejects branch predicates that vary per lane, so the same
policy as the native-C decision is computed with a nonzero mask via
`(x | -x) >> 63`, full-u64 equality via XOR, and decision selection via a
`(0 - c)` all-bits/zero mask. Each computed mask passes an empty volatile
asm compiler barrier so the BPF backend cannot re-fold the idiom back into
a lane-varying comparison; the emitted instructions stay plain
straight-line ALU ops. Decision semantics are unchanged, including the
`recorded==0` precedence over `recorded==kernel_idx` in the else-if chain;
no snapshot value is marked uniform and no verifier check is weakened or
disabled.

The stored HAL patch now includes `GuardianSM120`, its factory case and the
generated-array consumer, enabled with `XG_SM120_GENERATED_HEADER` (the exact
value is printed by `make hal-arrays` in `../level2-build`). The matched
NVBit actuator path returns before cuXtra-only Guardian/InstrMemAllocator
initialization. The real HAL/tool-actuator build now succeeds in an isolated
source/build/install directory after two small compile fixes; frozen Level-1
dependencies are unchanged. End-to-end execution remains unfinished. The native cuXtra
reference remains separate from the native-C/BPF pair, which uses the same
NVBit actuator. Existing Level-1 dependencies and measurements are unchanged.

The current HAL patch uses an 8192-slot mapped context pool, reserves slot
zero, and does not retire assigned slots. This is a bring-up limitation,
not support for unbounded long-running service. No binaries, cache payloads
or historical-result replacements are included in this checkpoint.
