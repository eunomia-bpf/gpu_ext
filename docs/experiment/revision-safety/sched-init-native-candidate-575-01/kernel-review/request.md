You are an independent read-only systems reviewer. Do not call tools, edit
files, run commands, use the network, or infer that any GPU/module experiment
has run. The complete relevant files are attached directly.

Audit the actual kernel-side scheduler-init diagnostic patch against the
attached frozen experiment plan. Return `READY` or `REQUIRED FIXES`, with
blockers first and exact file/function references. Focus on semantic evidence
integrity, not style:

1. Is the new context fixed-width and address-free, with enough immutable
   identity to join VALIDATED, NATIVE_RETURN, CONSTRUCTOR_RETURN, and the
   existing post-wait GSP events without providing policy input or control?
2. Is VALIDATED emitted only after the production validator, and is each
   NATIVE_RETURN emitted immediately after the corresponding existing setter
   and before the pre-existing failure branch, with status and actual post-call
   field captured?
3. Are setter order, the timeslice `NV_TRUE` argument, policy booleans, status,
   assertions, jumps, remote replay behavior, and cleanup semantics preserved?
4. Does every constructor that activated the task-init observation emit exactly
   one CONSTRUCTOR_RETURN on success or failure, including a
   `ctxBufPoolReserve` failure and lock-reacquire early return? Is the final
   snapshot taken while the group is valid and under the existing lock, never
   after cleanup?
5. Is the hook implementation barrier-only and void, without struct_ops
   dispatch, state mutation, return channel, `notrace` relaxation, or pointer
   exposure?
6. Does the CPU test genuinely pin the shared ABI, all phases/fields,
   constructor epoch, status preservation, and void hook signature? Flag any
   test that could pass while the actual constructor placement is wrong.

Treat module compilation and CPU test output as separately reported gates;
review the source itself here. Do not request a GPU run at this stage. Do not
report hashes, checksums, fingerprints, or digests. Keep the final review under
900 words.
