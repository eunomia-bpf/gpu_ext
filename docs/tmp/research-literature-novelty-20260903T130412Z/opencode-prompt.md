You are the independent, read-only reviewer for one bounded literature node.
Use the configured default model. Do not change any file, invoke bash, network,
other agents, builds, GPU operations, services, Git, or any content integrity
calculation. Only read/glob/grep/list are permitted. Do not browse the repository.
At most EIGHT read-tool calls total, and at most the SIX files listed below;
the two papers are included in that total. If information is missing, say so
rather than extending the search. Do not read workload raw data or dependencies.

Repository root: /home/yunwei37/workspace/gpu/gpu_ext
All paths below are relative to this root. Read these six files only:
1. docs/tmp/research-literature-novelty-20260903T130412Z/specmd-v1.txt
   Limit attention to lines 250–399 (Sections 3.1.3, 3.1.4, 4.1–4.2) and,
   if necessary, lines 750–800 (implementation); not the complete paper.
2. docs/tmp/research-literature-novelty-20260903T130412Z/urgengo-v1.txt
   Limit attention to lines 348–427 and 484–606 (Sections 4.2, 4.4.3–5).
3. workloads/finemoe/results-performance.md
4. workloads/finemoe/finemoe_policy.h
5. workloads/finemoe/finemoe_copy_ledger.h
6. workloads/hummingbird/idle_policy.h

Verified source boundary: SpecMD is Duc Hoang et al., Apple,
arXiv:2602.03921v1, 2026-02-03. The Apple author page labels ICML May 2026;
the downloaded v1 still says under review. UrgenGo is Hanqi Zhu et al.,
arXiv:2509.12207v1, 2025-08-26, with placeholder ACM venue/DOI metadata.
No official artifact link was confirmed from the inspected primary pages or
these PDFs. An unrelated project named SpecMD is not an author artifact.
Do not independently assert absent code worldwide or verified accepted venue.

Current completed evidence: FineMoE's 20-cell C/BPF component port reduced
evicted-unused completed logical payload versus all-positive, but BPF was
12.62% slower than demand-only; its paired C/BPF throughput difference was
unresolved. Hummingbird's completed 50-cell host-C/host-BPF component study
lost about 19–20% BE throughput versus fixed GPreempt. Its common executor
allows one LP launch in flight and waits for completion; this differs from
the paper pipeline and is not evidence that original Hummingbird loses 20%.
No improvement experiments or new algorithm have been run in this node.

Review one proposed next algorithm, without inventing a name:
Keep the existing FineMoE ranking, mandatory demand work, model and pool fixed.
Use past completed-copy outcomes (first demand use versus eviction before use),
currently outstanding speculative bytes, and current mandatory-demand backlog
to control a bounded speculative-byte admission budget. Resident-unused copies
are censored, not failures. Shrink admission when demand is queued or resolved
unused payload is high; cautiously reopen/probe only from already available
feedback. The exact rule is NOT frozen and has not been shown novel or useful.
Native and BPF must receive the same immutable, timestamped event-derived
snapshot and return the same bounded admission decision. A governor around
FineMoE is not a faithful reproduction of its Eq. 6–8 minimum-K selector.

Questions:
- What exactly do Least-Stale and UrgenGo decide, and which required inputs or
  actuators exist versus require a new shared ABI? Distinguish host JIT from
  kernel/device BPF and a component port from the complete authors' systems.
- After stripping names, is the proposed rule merely renamed least-stale,
  confidence thresholding, or ordinary feedback control? Identify overlap and
  unresolved novelty rather than asserting novelty from these two papers.
- Identify critical correctness/information problems: delayed/censored feedback,
  all-zero budget lockout, causality, actual demand backlog versus cumulative
  counters, admission versus completed bytes, fairness/native parity.
- Give the smallest decisive future experiment against demand-only, original
  FineMoE and a fixed-byte-budget alternative with equal tuning budgets. State
  what each baseline winning would mean. No GPU commands or broad plan.

Final answer MUST be at most 600 Chinese characters, with concrete file/line
or paper-section evidence and one next action. After the bounded reads, return
the final answer immediately; do not iterate or expand the task.
