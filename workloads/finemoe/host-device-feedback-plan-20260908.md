# Existing FineMoE: device-feedback extension

Status: assignment corrected by the user; the subsequent Qwen Next session
ended with a terminal provider HTTP 524 error without a source patch. No new
performance measurements exist. The task was not stopped for silence. The
currently selected faithful host/device implementations are XSched's guardian
and Hummingbird's block mapping, whose original device roles are explicit in
their PDFs; this FineMoE proposal remains unimplemented, not completed.

## User correction: preserve the original algorithm first

The user clarified that several original systems already span host and device.
The immediate task is faithful host/device placement of existing logic, not a
new adaptive feedback rule. The local session received this correction before
any implementation was observed. The original assignment below is retained
as proposal history, not the current algorithm specification. Inspect the local
FineMoE PDF at docs/reference/2026-yu-finemoe.pdf and preserve Eq. 6–8,
thresholds, observations, and update semantics. Do not equate ordinary CUDA
model computation with a device-BPF policy. See
../../docs/experiment/original-policy-host-device-boundaries-20260908.md.

The shepherd asks whether existing policies fit the mechanism and what the
mechanism costs; the user now asks to extend existing cases across host and
device rather than add research systems. This extension is a new policy variant,
not a replacement for the completed original-algorithm reproduction.

## Hypothesis and comparison

Past actual expert usage, aggregated by device BPF and consumed in batches by
the host selector, can reduce useless speculative transfers enough to offset
instrumentation and adaptation overhead. The existing results show about 220 GB
of evicted-unused speculative payload per dynamic-set cell, with demand-only
faster than both original policy implementations. This identifies an opportunity,
not proof that feedback will improve throughput.

Reuse the official model, held-out inputs, executor, and throughput measurement
in results-performance.md. Preserve demand-only and original native/BPF modes.
New native/BPF feedback modes must use the same observations and algorithm;
feedback-disabled execution isolates observation cost where practical. Compare
throughput and TTFT, with unused speculative payload as an explanatory metric.
Instrumentation cost or stale observations may outweigh reduced speculation;
that negative result must also be retained. No new paper or unrelated workload
is required, and this session does not edit the paper.

GPU execution will follow the live LMCache campaign and its queued missing
control. No completed original cells are rerun now. Exact new-mode commands and
the repetition schedule will be recorded after the source interface exists,
before performance execution. Source preparation alone is not an experiment
result. All prior results remain unchanged.

## Local implementation assignment

One local OpenCode session owns only workloads/finemoe. The other two sessions
continue disk-UVM implementation and CPU-fault analysis. No additional agent,
short timeout, driver reload, or concurrent GPU measurement is authorized.

```text
Implement host+device feedback for the EXISTING FineMoE experiment; do not reproduce another paper. Work in /home/yunwei37/workspace/gpu/gpu_ext. You own only workloads/finemoe/ source changes, a narrowly necessary device telemetry helper under that directory, and a small implementation note there. Do not edit paper/docs/paper, LMCache, driver worktrees, bpftime main, or frozen Table1 builds/assets. Preserve all old modes and raw results unchanged. No commits/push; root does those.

Goal: actual device-side eBPF records/aggregates real expert usage, host policy consumes batched feedback to reduce admitted-but-unused speculative expert transfers. This must NOT be host-only counters renamed device feedback, a marker-only standalone kernel, or merely a plan. Reuse existing bpftime device runtime / SASS EXIT injection support if suitable; read-only assets: workloads/sass-kretprobe/ (real existing-application EXIT injection), /home/yunwei37/workspace/gpu/bpftime-sass-existing-application, /home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt. A SASS-independent existing PTX path is fine if it actually supports the FineMoE kernels. Do not build a general SASS backend as a separate project. No applications can be assumed PTX-free without inspecting their build.

Read workloads/finemoe/results-performance.md, policy_runtime.py, inference.py, compare.py, finemoe_policy.{h,c,bpf.c}, policy.mk, and the necessary official executor under deps. Original dynamic selector is host uBPF JIT, not device BPF. Current source copies router probabilities to CPU for every prediction; roughly 220 GB speculative payload is evicted unused per cell; demand-only beats native/BPF on throughput. Develop a bounded feedback algorithm with a clearly explained update rule and cold-start behavior using only past actual usage (not future trace). Keep original Eq6-8 selector untouched in old modes. Improved native and BPF host variants must share the same selection/update algorithm and the SAME device observations. Include a practical host-only feedback-off ablation if it can reuse modes without framework growth.

Device observations must be attributable to actual layer/expert generation or a non-reused launch identifier; do not map every kernel EXIT to a guessed expert. Aggregate duplicate observations on GPU where semantics permit; batch export at an existing completion/iteration boundary, no CPU polling thread or per-thread PCIe event stream. If there is no legal observation point or reliable mapping in the existing runtime, report the exact missing API and implement the narrowly necessary adapter instead of faking device engagement. A missing common helper may be added locally; no cross-repo changes without telling root first.

GPU is occupied by a formal LMCache campaign AND a queued missing-control job. Do not launch GPU workloads, reload driver, change loaded BPF, compile CPU-heavy dependencies, or run broad tests. Source implementation and lightweight source inspection now; root will coordinate ordinary build and a real performance run once GPU free. Do not add correctness campaigns, clock calibration, hashes/digests, preflight or approval gates. Use apply_patch for edits. Preserve uncommitted unrelated docs/paper and faiss changes. No new worktrees, subagents, papers or downloads. No artificial timeout; keep working to actual source deliverable or an actual error. Context limit 200k: prioritize a usable patch over repeated broad reading, write concise findings as needed before compaction.

Deliver: source patch, minimal existing runner invocation for the new native/BPF modes, expected observation-to-decision data path, ordinary build command and any unresolved dependency. Do not claim measured gains before real runs.
```
