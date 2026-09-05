# ASPLOS revision 1797 — literature and experiment scouting (2026-09-05)

Scouting for recent (2023–2026) primary papers in GPU memory/offloading,
scheduling/preemption, binary instrumentation/SASS, multi-tenant GPU sharing,
and agent-generated resource policies. Selection rule: not already present in
`docs/paper-material/policy-expressibility-papers/` (60 numbered PDFs after
this addendum, entry numbers 02–65) or `docs/paper-material/ref-paper/`
(17 PDFs), public primary source,
and either a public runnable artifact or a policy plausibly expressible with
the current gpubpf hooks (`gpu_page_prefetch[_iter]`, `bpf_gpu_set_prefetch_region`,
`gpu_block_activate/access`, `gpu_evict_prepare`, `bpf_gpu_request_reorder`,
scheduler `gp_task_init`/`gp_timeslice_control` and whole-TSG preemption,
device callbacks, SASS AOT at bpftime `fd976ea`).
Six PDFs were downloaded and first-page title/author-checked.

Searches used the arXiv API, GitHub API, and Semantic Scholar (rate-limited,
used only to confirm CuAsmRL's CGO'25 venue). Rejected candidates: Kamino
(OSDI'24, no public artifact located), the PPoPP'24 FlexSP scheduler paper (the
arXiv `FlexSP` hit is the unrelated sequence-parallelism paper), "FlexIS" GPU
interrupt work (no reachable primary source under that title), and
AgentRM/AgenticOS-style OS papers without runnable policy components (2603.13110,
2606.21129). "Towards Efficient and Practical GPU Multitasking in the Era of
LLM" (2508.08448) was skipped as a survey; msched (2512.24637) is already entry 11.

## Selected papers

### 60. Llumnix: Dynamic Scheduling for Large Language Model Serving
- Venue/year: USENIX OSDI 2024. Primary URL: <https://arxiv.org/abs/2406.03243>
- Local PDF: `docs/paper-material/policy-expressibility-papers/60-osdi24-llumnix.pdf`
- Artifact: <https://github.com/llumnix-project/llumnix> at `aa9097c899`
  (2026-05-26); companion <https://github.com/llumnix-project/llumnix-ray> at
  `3fb6c0376b` (2026-03-12). Live KV migration plus request rescaling.
- Blockers: targets vLLM-era serving stacks (version pinning), Ray control
  plane, and its headline benefit needs multiple GPUs; on one RTX 5090
  (driver 575.57.08, Linux 6.15) only same-GPU multi-instance coexistence is
  reachable. Its defining action — live migration of a running instance's KV —
  is exactly the cross-context data movement the driver ABI lacks (the matrix
  `NO` boundary), so a gpubpf arm can only make scheduling decisions.
- Smallest real reproduction cell: two ~3B-instance vLLM servers sharing the
  5090 under a trace-driven arrival load; measure rescheduling's effect on
  SLO attainment for one workload pair; skip cross-GPU migration.
- Tests: **policy benefit** of scheduling decisions (priority/timeslice
  analogue via host-uBPF), and a concrete **mechanism-gap marker** (migration
  actuator absent).

### 61. ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference
- Venue/year: ICML 2025 (Spotlight); arXiv v3. Primary URL: <https://arxiv.org/abs/2410.21465>
- Local PDF: `docs/paper-material/policy-expressibility-papers/61-icml25-shadowkv.pdf`
- Artifact: <https://github.com/ByteDance-Seed/ShadowKV> at `e51904cdea`
  (2025-05-01); README documents single-GPU (A100) reproduction.
- Blockers: kernels are built for A100/sm_80-class; sm_120 (Blackwell) rebuild
  of FlashTransformer/Triton paths must be verified on the 5090; accuracy
  suite expects 8×A100 (skip it). Its per-step top-k group loading uses
  tensor-level KV semantics gpubpf cannot name (sub-page/token identity is a
  `NO` boundary in the matrix), so the original selector stays outside the ABI.
- Smallest real reproduction cell: one long-context decode throughput point
  (offload on vs off) for Llama-3.1-8B-class model at ~128k context on the
  5090, then rerun the same tensor layout under forced GPU-memory
  oversubscription and let a gpubpf prefetch/reorder policy choose the loaded
  KV region per decode step.
- Tests: **policy benefit** baseline plus the closest current-ABI analogue of a
  semantic KV selector — expressibility of selection at contiguous-region
  granularity.

### 62. LLM in a flash: Efficient Large Language Model Inference with Limited Memory
- Venue/year: ACL 2024 (Apple). Primary URL: <https://arxiv.org/abs/2312.11514>
- Local PDF: `docs/paper-material/policy-expressibility-papers/62-acl24-llm-in-a-flash.pdf`
- Artifact: **no official public code located**; only unofficial mirrors
  (e.g. <https://github.com/peircerandy/llminflash>, OPT-6.7B flash engine).
- Blockers: the engine targets Apple silicon (ANE/Metal, flash-resident
  2-bit weights); there is no CUDA reference port, so an original-system
  reproduction on the 5090 is not possible. What ports cleanly is the paging
  *policy*: which weight block to bring in ahead of each token's activation.
- Smallest real reproduction cell: reproduce the paper's paging-throughput
  *shape* only — reuse the existing llama.cpp/UVM oversubscription harness:
  weight-heavy decode under 32 GB pressure, comparing built-in paging against
  a gpubpf contiguous-region prefetch choice (already the same decision class
  as `prefetch_adaptive_sequential`).
- Tests: **policy benefit** of weight-paging choices as a same-decision
  expressibility control; original-system claim impossible (state this).

### 63. CuAsmRL: Optimizing GPU SASS Schedules via Deep Reinforcement Learning
- Venue/year: IEEE/ACM CGO 2025. Primary URL: <https://arxiv.org/abs/2501.08071>
- Local PDF: `docs/paper-material/policy-expressibility-papers/63-cgo25-cuasmrl.pdf`
- Artifact: <https://github.com/hgl71964/cuasmrl> at `fed7fb14ad`
  (2024-11-09); README claims Compute Capability 7.0+.
- Blockers: binary disasm/reasm and instruction-encoding assumptions predate
  Blackwell; sm_120 (5090) support must be proven by one successful kernel
  round-trip before any training run; rewards come from real execution, so a
  mis-assembled schedule is a correctness risk on a production GPU.
- Smallest real reproduction cell: one shipped example kernel round-trips
  (disassemble → apply one RL-chosen schedule → reassemble → execute → compare
  outputs/timing) on the 5090; no RL training needed to test the mechanism.
- Tests: **mechanism expressibility** of below-source binary policy
  application — the external complement to our SASS AOT result at `fd976ea`,
  which currently proves only standalone generated cubin, not patching of
  foreign SASS.

### 64. Towards Agentic OS: An LLM Agent Framework for Linux Schedulers
- Venue/year: arXiv 2509.01245v4 (2025). Primary URL: <https://arxiv.org/abs/2509.01245>
- Local PDF: `docs/paper-material/policy-expressibility-papers/64-arxiv25-agentic-os-schedulers.pdf`
- Artifact: <https://github.com/eunomia-bpf/schedcp> at `730e258987`
  (2026-02-12) — same upstream organization (eunomia-bpf) as bpftime/gpubpf.
- Blockers: CPU-side sched_ext, not GPU; needs Linux ≥6.12 with sched_ext
  (6.15 present is fine) and a capable agent loop. No GPU blocker because the
  reproduction runs entirely on CPU — the safest new experiment here.
- Smallest real reproduction cell: one closed-loop session where the agent
  writes/edits a sched_ext policy for a fixed microbenchmark pair until the
  kernel verifier or a benchmark gate rejects it; log admissions and effects.
- Tests: **mechanism expressibility** of the agent→verifier→run loop for
  generated resource policies, on a kernel whose verifier-admission pattern
  (write policy, admit or reject, never admit an unsafe policy) matches the
  gpubpf STRICT route.

### 65. KunServe: Parameter-centric Memory Management for Efficient Memory Overloading Handling in LLM Serving
- Venue/year: arXiv 2412.18169 (2024, v5 2025). Primary URL: <https://arxiv.org/abs/2412.18169>
- Local PDF: `docs/paper-material/policy-expressibility-papers/65-arxiv24-kunserve.pdf`
- Artifact: <https://github.com/SJTU-IPADS/kunserve> at `35a7d7f43a`
  (2026-04-25); research execution stack with optional vLLM/SGLang backends.
- Blockers: serving-framework pinning and an original testbed larger than one
  5090; its parameter-granular offload/reload decisions sit on the matrix's
  `PARTIAL` boundary (tensor identity and destination selection are missing
  driver actions).
- Smallest real reproduction cell: single 5090 with reduced GPU-memory
  fraction driving an overload episode on a small model; compare built-in
  overload handling with the paper-style parameter-centric handling on one
  workload mix.
- Tests: **policy benefit** under serving overload; gpubpf side would express
  only the region/ranking component, which is itself the open question.

## Proposed gpubpf-enabled algorithms (hypotheses, not novelty claims)

1. **Decode-window contiguous-region KV prefetch** (from ShadowKV/KunServe
   pressure patterns). Use `gpu_block_access` counters per VA block, select
   the prefetch window inside the faulting block with
   `bpf_gpu_set_prefetch_region` at `gpu_page_prefetch`, and demote stale KV
   blocks via `bpf_gpu_request_reorder` at `gpu_evict_prepare`.
   Hypothesis: under forced oversubscription in the llama.cpp/UVM harness, the
   access-weighted window beats `prefetch_adaptive_sequential` on decode
   kernel time by a margin whose CI excludes zero, and beats no-prefetch with
   lower cost than the known 3.2% mechanism tax — quantifying how much of a
   tensor-semantic selector's benefit survives contiguous-region granularity.
2. **Migration-window timeslice protection** (from Llumnix). At
   `gp_task_init` classify instance roles; during a host-signaled KV-copy
   episode, `gp_timeslice_control` widens the receiver's timeslice and
   whole-TSG-preempts background TSGs, restoring baseline slices on episode
   end. Hypothesis: reusing the gpreempt LC/BE harness, protected windows cut
   LC p99 during transfer events by ≥15% versus fixed timeslice, with BE loss
   inside the already-measured ~9% band.
3. **Verifier-gated agent policy loop** (from Agentic OS/schedcp). An
   opencode/LLM loop drafts candidate gpubpf prefetch/eviction policies; every
   candidate must pass the existing STRICT device admission (PREVAIL+SIMT,
   target-PID bound) before any timed cell runs, mirroring the loader study's
   STRICT/WARNING contrast. Measurable: STRICT first-pass rate of generated
   programs, rejection reasons, and best policy effect per unit candidate
   budget versus the hand-written in-tree policies. CPU-side admission keeps
   GPU risk bounded. This tests the expressibility+safety envelope of
   agent-written policies, not their novelty.

## Files downloaded

- `60-osdi24-llumnix.pdf` (1,770,503 B), `61-icml25-shadowkv.pdf` (1,712,244 B),
  `62-acl24-llm-in-a-flash.pdf` (1,333,354 B), `63-cgo25-cuasmrl.pdf`
  (1,708,161 B), `64-arxiv25-agentic-os-schedulers.pdf` (430,348 B),
  `65-arxiv24-kunserve.pdf` (1,905,568 B), all under
  `docs/paper-material/policy-expressibility-papers/`.

## Top next experiment

Algorithm 1 (decode-window KV prefetch): zero new dependencies, reuses the
existing UVM harness and in-tree selectors, directly probes the open
"semantic-selector versus contiguous-region" expressibility question, and its
no-prefetch control already has a same-policy mechanism-cost baseline. Run the
ShadowKV single-cell reproduction first only if the sm_120 kernel build passes;
otherwise Algorithm 3's CPU-only loop is the zero-risk fallback.
