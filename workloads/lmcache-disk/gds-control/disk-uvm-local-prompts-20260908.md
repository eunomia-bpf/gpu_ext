# Disk-UVM local implementation task prompts — 2026-09-08

These are new follow-up task prompts, not recovered transcripts from the
original paper's agentic policy search. They preserve the actual assigned
scope. No local-model reasoning traces or internal snapshot metadata are
published here. Root owns experiments and publication; local models own
nontrivial implementation. The existing LMCache policy-analysis GLM session
continues separately, keeping the live OpenCode count at three.

## Driver implementation

Session: `ses_f8109d0d1ffets4QE1Y1sWOj3q`. Model: `spark-gateway/qwen3.8-27b-nvfp4-200k`.

```text
Implement the user's requested REAL disk-backed UVM mechanism, not another paper reproduction. You are the sole kernel-code owner in /home/yunwei37/workspace/gpu/gpu_ext-kernel-575-gds on current revision/gpu-storage-decision-575 branch, initially clean. Read /home/yunwei37/workspace/gpu/AGENTS.md and relevant local build notes. Root owns current running 20-cell LMCache total-cost campaign, so NO GPU runs, module install/reload, killing processes, changing main gpu_ext workload code, paper edits, Git operations, new worktrees, hashes or ownsubagents. Use shell apply_patch at /home/yunwei37/.codex/tmp/arg0/codex-arg029oUiU/apply_patch if available; no arbitrary short timeouts.

Scope: first actual registered managed-range disk backing with staged CPU transport, async offload and SAME-VA demand restoration. This is a feature implementation, not a report-only/task stub. Restrict initial registration to an explicitly sealed/read-only range if necessary; record how that contract is enforced and what remains for writable support. Do not pretend a fixed vLLM KV pool releasing slots frees physical GPU pages. Need actual disk-resident state with host/device data copies removable after successful write; demand restore must prevent the existing zero-fill path. I/O must not sleep under UVM locks. BPF eventual control of offload/prefetch is the overall target; implement reusable backing mechanism now, don't claim existing cmd82 decision-only IO is that mechanism.

Existing source route (verify before trusting): managed_range in uvm_va_range.h:270; GPU->CPU staging in uvm_va_block_evict_chunks around13090; PMM tracker/chunk eviction around13143; eviction_mappings_q_item retain/release pattern around13136; CPU chunks uvm_pmm_sysmem.h; block_populate_pages_cpu; sameVA GPU fault via service_fault_batch_block_locked; CPU fault path must also work or explicitly reject unsupported operations rather than silently zero data. Root found an IMPORTANT error in prior analysis: returning NV_WARN_MORE_PROCESSING_REQUIRED at replayablefaults.c2333 does NOT automatically park a fault until async I/O completes: it continue's SAME index after dropping VAspace locks, so could busy-retry/starve workers. Design real in-flight wait/retry outside UVM locks using actual callchain, not handwaving GPU replay. Another risk: holding chunk kref is not a stable data snapshot; protect copy validity from CPU/GPU writes, migration, range split and teardown. Existing CPU dirty bits are NOT a disk-version oracle. Initial immutable data contract can simplify, but cannot silently allow unsupported mutations. Store backing file reference safely; lifetime/file offset bounds/errors/unregister teardown must not leave dangling block or range pointers. Restrict exact range/splits as needed rather than broad driver rewrites.

Root allows ordinary build inspection, no gate/preflight/audit campaigns. Reuse actual current build invocation for kernel6.15.11 (not outdated GPUBPF-RUNTIME doc saying610). Build only, do not install. No make clean. Preserve existing generated module/assets used by root; separate output/module copy before overwrite if required. This task may take time; complete a useful coherent implementation, do not stop at only a new ioctl definition or write-only sidecar. If a concrete primitive prevents a full slice, return exact source/code/error and implementation landed, don't fake success.

Before extensive edits, send a concise chosen ABI/ownership plan in session text so root can assign a matching CUDA benchmark independently. Keep file scope minimal, a necessary small backing module + existing integration points only; no framework, testsuite boilerplate, newpapersearch or extra correctness campaigns. Final handoff: exact changed files, build command/result, what runtime action should demonstrate original address restored and actual memory freed, limitations. Root reviews/runs/commits/pushes.
```

## CPU fault integration

Session: `ses_f81094c96ffeOj627bps5io5Wy`. Model: `spark-gateway/qwen3.8-flash-next-nvfp4-220k`.

```text
Bounded read-only source task supporting an actively implemented disk-UVM feature, not a paper survey. Root/user require local OpenCode models only, no ownsubagents, no hashes, no shorttimeout. Workdir /home/yunwei37/workspace/gpu/gpu_ext-kernel-575-gds. Read parent AGENTS. Qwen27 owns all driver writes: DO NOT change files, run GPU, build/reload modules, Git, or read whole huge logs. Focus ONE missing integration: CPU access to an explicitly registered managed (cudaMallocManaged) range whose valid data copy is disk-only. Find exact CPU-fault path/lock ownership and smallest place to wait for a worker to restore bytes without holding UVM/VA locks or accidentally zero-filling. Include how CPU writes/read-only sealing and teardown intersect that path. Existing GPU service NV_WARN_MORE_PROCESSING_REQUIRED loops sameindex after dropping locks, not an async wake API, so do not recommend blind retry as sufficient. Existing filebackedHMM and userfaultfd shortcuts already ruled out; skip that survey. Return <=1000words actionable functions/lock constraints and one recommended integration point. Distinguish sourcefacts from designs; no toy implementation, extra audit, hashes, otherpaperdownloads, or claims disk-UVM already works. Root will relay to codeowner and runs real benchmark after current GPU batch finishes.
```

## Coordination after dispatch

The root subsequently asked the driver owner to defer CPU-heavy module
compilation until the active GPU campaign finishes, avoiding interference
with vLLM on CPUs 8–15. Source inspection and edits continue. No driver
installation, reload, or GPU access is delegated. The historical runtime
preparation note must not be interpreted as the current driver state.

