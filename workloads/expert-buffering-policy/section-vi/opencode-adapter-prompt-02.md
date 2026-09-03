Read-only OpenCode code-drafting subagent; use your default model. No shell,
editing, GPU, builds, networking, or file hashes/checksums/fingerprints/digests.
The attached context contains ALL needed exact source excerpts and the already
implemented CUDA-free State API. Do not search directories or request more
files. If a detail is absent, explicitly bound the missing detail and finish.
The prior run wasted time looking for a header; its complete definition is now
attached as workloads/finemoe/finemoe_runtime_safety.h. Do not repeat that search.

Task: give the COMPLETE FINAL concrete minimal C++/Python patch proposal for
Expert Buffering Section VI on an independent FineMoE source copy. Native,
actual host-BPF and FIFO share true gating and whole-expert serial execution.
Fixed per-(GPU0, model layer) K=16 is an explicit port choice, not a recovery of
the original cross-layer undocumented implementation. Positive CURRENT batch
token counts, inactive-first then LIFO, serial increasing-ID expert execution.

Use the attached implemented State API rather than invent a second state model:
Begin/End, Locate/Get/ActiveEpoch, Decide/Validate/Evicted, CanAdmit/Admitted.
State owns dlopen/uBPF lifetime and checks output status against snapshot state;
BPF really selects the victim. State does NOT lock and caller must call Admitted
only AFTER successful physical copy. State.Get exposes cohort node-ID vector
and per-expert resident/admission metadata, which the live adapter must compare
with actual locked Node residency. End/Begin require caller checks no live exec.

Return actual minimal insertion snippets for four real points:
1. Qwen forward actual top-k assignment counts, layer Begin before expert fetch,
   End after expert loop; disable trajectory/embed prediction equally in arms.
2. Python model_offload engine configure + model layer metadata, representative
   tensor IDs resolve to complete topology nodes; add concrete pybind methods.
3. task_scheduler real victim commit and bounded failure with current epoch,
   whole-node residency and execution eligibility revalidated under locks;
   keep BOTH cohort K and existing global strict byte budget. Do not merely
   set Python priorities or C-preselect a victim for BPF approval.
4. actual SetNodeDevice / Node::SetDevice completion: stamp admission only
   AFTER synchronous whole-expert H2D succeeds, preserve existing acquire/release
   compute synchronization, no silent fallback or unsafe overflow allocation.

Pay special attention to lock order. Existing main acquire owns Node::mutex
then StartExec takes exec_mutex. CompleteDemand acquires Node::mutex and invokes
SetNodeDevice, whose operation publishes ready afterward. A new metadata lock
must never block waiting for Node locks; take metadata then TRY_LOCK candidates,
fail boundedly if unavailable. One Python inference / GPU0 copy worker is the
supported mode; concurrent clients are rejected. Explain the direct C ABI call
does not reacquire Python GIL; lifetime survives worker teardown. Avoid claiming
live safety before build/GPU tests. Return a finished report now, with no tools
unless reading the single attached context is needed, and no generic checklist.
