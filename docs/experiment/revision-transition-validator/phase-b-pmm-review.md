# Transition validator Phase B PMM review

Date: 2026-08-31

Verdict: `PASS` for the 575 and 610 production integration, kernel-native test,
extension ABI migration, and the PMM live preflight. This is not approval of
the five unexecuted scheduler fixture outcomes.

Independent review confirmed that the 575 production path uses one
lock-asserting helper for every root-capable list mutation, advances generation
only on membership changes, keeps same-list reorders generation-stable, and
leaves proven subchunks unchanged. The callback-local context, typed raw-width
setter, source/identity/generation/range/repeat/conflict validation, access
truth table, and post-native activate behavior match the frozen contract. The
old raw head/tail kfuncs have no definition, registration, or in-tree caller.

The 610 review separately traced all native aliases, including the lazy path in
`uvm_devmem.c`. It confirmed exact metadata mapping for `USED`, `UNUSED`, and
the driver-only `DISCARDED` state. Policy validation exposes only `USED` and
`UNUSED`; `DISCARDED` can be moved by native behavior but cannot be a policy
source or destination.

Review of both kernel-native tests confirmed unique ioctl registration, real
production objects and lock/list helpers, complete invalid-sequence and action
coverage, stale and foreign-identity controls, illegal source states, activate
semantics, generation assertions, and cleanup without publication into live
GPU or global state. The 610 adaptation initializes all three allocation-list
heads and explicitly tests `DISCARDED` rejection.

For the extension, review confirmed the 72-byte public decision layout, all 60
callback signature migrations, typed setter propagation through helper
functions, removal of old raw-move calls, and agreement between maintained and
generated public headers. The two PMM fixtures use the received callback-local
context: the negative emits the intended hidden-state write and the positive
uses the typed kfunc. The loader places four positives before three negatives,
counts only `-EACCES` as a verifier denial, never attaches, and enforces the
frozen 7/4/3/7 totals.

Fresh result review additionally checked the live PMM evidence: the loaded
custom module exposed the expected BTF, the registered kernel-native ioctl
returned success, the setter fixture loaded, and the direct hidden-state store
was rejected at offset 56 with `-EACCES`. The runner reported the frozen PMM
subset totals 2/1/1/2, and its two raw verifier logs agree with the console
result. Recovery restored the distribution UVM module with built-in tests off
and did not unload the display-owned core modules.

Phase B remains `PARTIAL` only for the default seven-fixture run and scheduler
runtime path: the five scheduler objects require a BTF-enabled core `nvidia`
module, which cannot replace the display-owned module without a maintenance
window.
