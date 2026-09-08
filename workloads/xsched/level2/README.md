# XSched Level-2 sm_120 source checkpoint

Status: development snapshot, **not built or measured**. The existing
Level-1 measurements and dependency checkout are unchanged. This is not a
completed reproduction of the original Level-2 system.

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

The current HAL patch uses an 8192-slot mapped context pool, reserves slot
zero, and does not retire assigned slots. This is a bring-up limitation,
not support for unbounded long-running service. No binaries, cache payloads
or historical-result replacements are included in this checkpoint.
