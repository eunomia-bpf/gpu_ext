You are the requested OpenCode read-only code-drafting subagent. Use the default
configured model. Your permissions are read/glob/grep/list only, snapshot=false.
Do not execute shell, edit files, build, import torch/CUDA, access GPU/network,
or generate/record any file hashes/checksums/fingerprints/digests. Do not read
weights, raw experiments, unrelated code, or the full repository recursively.

Return a COMPLETE final concrete code proposal (not merely a checklist), with
minimal insertion/replacement snippets and lock-order reasoning. The parent
will independently review and apply the code under section-vi, never modifying
the frozen FineMoE source. No need to investigate other policies or papers.

Read section-vi/plan.md and policy.h/c first. The task is the real Expert
Buffering Section VI adapter, sharing actual gating, serial increasing-ID
expert execution and whole-expert transfers between FIFO/native/BPF arms.
Current-batch positive token counts, inactive-first then LIFO eviction;
insertion serial updates only after a successful whole-node CPU->GPU copy.
Our explicit port chooses a per-(device, MoE-layer) K=16 cohort (not claimed
to be the original undocumented cross-layer capacity implementation).

Then limit investigation to these exact files under
workloads/finemoe/deps/FineMoE-EuroSys26/:
- finemoe/models/modeling_qwen/modeling_qwen2_moe.py (offloaded forward near 842-926)
- finemoe/runtime/model_offload.py (engine construction, expert_tensor_map, hooks)
- core/prefetch/task_scheduler.cpp and .h
- core/prefetch/archer_prefetch_handle.cpp and .h
- core/model/model_topology.cpp and .h
- core/python/py_archer_prefetch.cpp
- op_builder/prefetch.py and setup.py only as needed for build integration
Do not expand this set except a specific directly included type/header whose
definition is indispensable; state any such addition in your final report.

We need four concrete integration points:
1. Actual positive routing counts/epoch after gating, BEFORE any expert fetch;
   disable all trajectory-prefetch prediction equally in the three EB arms.
2. Representative tensor-ID -> complete node mapping and batch notification
   through Python/C++, avoiding assumption model layer_id == topology stage.
3. RemoveCachedSparseNode real victim choice: snapshot complete eligible nodes,
   call C or actual host-uBPF selector (BPF chooses, no native preselection),
   revalidate epoch/residency/eligibility under correct locks, enforce both K
   and strict existing byte budget, bounded explicit failure, never fallback.
4. Successful admission updates serial; preserve existing compute-stream sync
   and node ownership; no evict while execution or transfer is in flight.

Prefer a small shared adapter state/interface source plus patches in an
independent staging copy. Expose native/BPF bridge via a single configured
function handle if that avoids relinking dependencies. The existing selector
ABI is 1008 bytes/60 experts, eb_select + eb_jit_open/eb_jit_select; status enum
HIT/ADMIT/EVICT/INVALID/BLOCKED. FIFO baseline must use identical cohort and
transfers, replacing only victim order. No new experiment framework.

Call out unresolved lifetime, GIL, blocking lock or build requirements honestly.
Do not claim GPU results or safety from CPU oracle tests. Focus final report on
compilable interfaces, actual callsites, a minimal patch plan, and pitfalls that
would make the first real GPU run unsafe. Keep it bounded and finish the report.
