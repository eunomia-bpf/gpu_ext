# XSched Level-2 sm_120 source checkpoint

Status: development snapshot, **not built or measured**. The existing
Level-1 measurements and dependency checkout are unchanged. This is not a
completed reproduction of the original Level-2 system.

The source includes a device BPF guardian, shared native-C/BPF trampoline,
NVBit launch-context adapter, native probe/guardian/restore stubs, LDC
patcher, and an opt-in HAL patch. Component build commands are in
`../level2-build/`. The HAL patch passed a non-mutating `git apply --check`
against the local XSched source; it has not been applied to that checkout.

Known integration work remains: the shared scalar device-state context and
its local exporter must agree (the reused SASS exporter declares only eight
context bytes); the native sm_120 artifact path and end-to-end replay must
still be built and exercised. Both matched decision arms must keep the same
trusted actuator. No host decision fallback or performance gain is claimed.

The current HAL patch uses an 8192-slot mapped context pool, reserves slot
zero, and does not retire assigned slots. This is a bring-up limitation,
not support for unbounded long-running service. No binaries, cache payloads
or historical-result replacements are included in this checkpoint.
