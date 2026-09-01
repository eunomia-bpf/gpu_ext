# Transition validator Phase B PMM review

Date: 2026-08-31

Verdict: `PASS` for the offline 575 and 610 production integration,
kernel-native test implementation, extension ABI migration, and focused BPF
fixture construction. This is not approval of a live verifier or ioctl result.

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

The remaining limitation is environmental, not converted into a pass: the
custom module was not loaded, the ioctl was not invoked, and the BPF fixtures
did not reach the running verifier. Phase B therefore remains `PARTIAL` until
a safe custom-module window supplies those live observations.
