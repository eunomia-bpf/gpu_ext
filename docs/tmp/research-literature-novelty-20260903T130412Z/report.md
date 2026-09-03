# Bounded literature verification: SpecMD and UrgenGo

Started: 2026-09-03 13:04:12 UTC. Parent: the active revision follow-on
literature node. Owner: `/root/pod_resume`; root owns experiments and Git.
Status: PDF metadata, relevant algorithm sections, local ABI inspection and
one bounded independent review complete. Canonical update awaits root review.
LMCache remains paused.

## Scope and candidate claims

This node examines exactly two primary-paper candidates, not an exhaustive
novelty review or a commitment to reproduce either complete system.

1. An expert-prefetch decision rule can use completed useful/evicted-unused
   copies and current demand backlog to bound speculative bytes while leaving
   mandatory demand execution unchanged. Existing least-stale eviction and
   confidence thresholds must not be renamed as a novel algorithm.
2. A kernel-admission decision rule can trade dispatch delay and batching cost
   against foreground urgency while a common executor preserves every launch.
3. Native and BPF versions can receive identical snapshots and produce the same
   decisions. A performance or novelty benefit is not established by this map.

Coverage: same problem, mechanism and observable metric in the two papers;
official artifact links; feasibility against the current FineMoE/Hummingbird
component interfaces. Hardware and whole-system reproduction are separate.
The root is still working on Table 1; no build, GPU, service or raw modification
is authorized here.

## Search history and primary-source verification

Root completed the following four queries on 2026-09-03 UTC before delegation;
their exact individual timestamps were not supplied. This node does not rerun
or broaden those queries.

- `MoE expert prefetch unused transfers adaptive cache GPU inference 2025 2026 artifact`
- `GPU kernel scheduling launch queue preemption overhead adaptive admission hysteresis 2025 2026 paper`
- `"UrgenGo" github artifact`
- `"SpecMD" "github"`

At approximately 13:04 UTC this node opened the following primary pages:

- <https://machinelearning.apple.com/research/specmd-expert-prefetching>
- <https://arxiv.org/abs/2602.03921>
- <https://arxiv.org/abs/2509.12207v1>
- <https://arxiv.org/html/2509.12207v1>

Apple's author page describes SpecMD as ICML, May 2026. arXiv exposes v1,
submitted 2026-02-03 18:36:56 UTC. The downloaded first page explicitly identifies
`arXiv:2602.03921v1`, 3 February 2026, and says "Preprint. Under review".
These are different metadata observations; the local PDF is not silently
relabeled as a verified conference camera-ready version.

UrgenGo arXiv and its first page identify v1, submitted 2025-08-26
10:08:15 UTC. The PDF has placeholder ACM venue/DOI metadata; no accepted venue
is inferred from the template. The PDF-generation date is not its submission
date. An unrelated project with the same SpecMD name is not an author artifact.

## Download inventory

Downloaded with `curl --fail --location --retry 1 --max-time 45` from the explicit
version URLs below. `pdfinfo` and `pdftotext -f 1 -l 1 -layout` independently
confirmed readable PDFs and their first-page titles/authors/version labels.
Only ordinary sizes and metadata are recorded; no content identifier is used.

| Paper | Stable repository path | Direct PDF URL | Bytes | Pages |
| --- | --- | --- | ---: | ---: |
| Duc Hoang, Ajay Jaiswal, Mohammad Samragh, Minsik Cho: SpecMD: A Comprehensive Study On Speculative Expert Prefetching | `docs/reference/2026-hoang-specmd-v1.pdf` | <https://arxiv.org/pdf/2602.03921v1> | 3,587,758 | 17 |
| Hanqi Zhu, Wuyang Zhang, Xinran Zhang, Ziyang Tao, Xinrui Lin, Yu Zhang, Jianmin Ji, Yanyong Zhang: UrgenGo: Urgency-Aware Transparent GPU Kernel Launching for Autonomous Driving | `docs/reference/2025-zhu-urgengo-v1.pdf` | <https://arxiv.org/pdf/2509.12207v1> | 4,394,281 | 15 |

The PDFs are the authors' work. Downloading and inspecting them proves neither
artifact availability nor compatibility, performance, novelty or reproducibility.

## Algorithm/source verification and coverage review

Read SpecMD Sections 3.1.3–3.1.4, 4.1–4.2 and Appendix A; read UrgenGo Sections
4.2, 4.4.3–4.4.5 and 5. The concise algorithm descriptions, primary citations
and baseline consequences are in the canonical background document. PDF text
extracts in this temporary node are review inputs, not another publication
corpus or evidence of executing either artifact.

The inspected Apple/arXiv pages and SpecMD PDF did not expose an author-code
link. UrgenGo Section 5 says source will be released before the conference but
does not identify an available artifact. We did not infer a worldwide absence
of code, import unrelated same-name projects, or expand beyond the root's four
queries. No repository version or runnable status is invented for either work.
Remaining ambiguity includes exact eviction tie rules and author artifacts;
neither paper is admitted as a runnable main baseline here.

| Required information / actuator | Local evidence | Consequence |
| --- | --- | --- |
| FineMoE current decision snapshot | `workloads/finemoe/finemoe_policy.h:7–16`: count, top-K, threshold, probabilities; mask output | No resident/pass/position labels, expert sizes, demand backlog or outstanding-byte budget is currently supplied to native/BPF selection. |
| Completed-copy outcomes | `workloads/finemoe/finemoe_copy_ledger.h:51–103`: per-copy generation, byte size, start/completion/first-use/eviction events | Useful/evicted-unused outcomes have concrete event sources, not just predicted probabilities. Exporting a bounded decision snapshot is still new glue. |
| In-flight and queued demand/speculation | Same header, `CopySnapshot` at 24–27 and counters at 105–125 | Started-versus-completed copy records can describe observed in-flight copies if sampled coherently; cumulative misses are not current demand backlog and queue counts are not queued byte sizes. No zero-cost or currently wired feedback ABI is established. |
| Censoring / correctness | Same header, 82, 117–123; `workloads/finemoe/results-performance.md` | Warmup generations are left-censored; resident-unused generations are right-censored. Count them separately and never call them proven waste. Mandatory demand output remains unchanged. |
| Existing idle scheduler snapshot | `workloads/hummingbird/idle_policy.h:18–37` | Real pending/completion, bubble and duration fields exist, but chain deadlines, remaining chain work, stream assignment and batch-completion state do not. A full transparent UrgenGo integration is not already present. |

The new candidate controls admission bytes, not victim choice, and would be a
separate governor around FineMoE. It is not the original minimum-K algorithm.
Current execution substrate would be host C versus host uBPF JIT, not a Linux
kernel/device policy, and no verification claim follows from compilation.
Native/BPF must share snapshot fields and observation/update rules with an
explicit clock, generation, units, overflow bounds and stale-snapshot policy.
Shadow evaluation must agree for an identical immutable snapshot; independently
timed online runs need not have identical feedback values. Trusted runtime
code must still preserve demand progress and validate the returned budget/mask.

The mechanism needs both delayed-outcome handling and a bounded reopening/probe
rule: a zero budget cannot be expected to generate new success evidence by
itself. Only outcomes already observed before the decision may be used. A
controller must account for admitted queued plus in-flight speculative bytes,
not mistake all completed bytes for current pressure. Fast first-use evidence
and delayed unused evictions can bias a naive resolved-only utility ratio;
the future design must specify outcome cohorts/age handling, not simply ignore
unresolved lifetimes. With equal-sized experts a byte cap reduces to a count
cap, so units alone do not distinguish this from top-K admission. These are implementation
requirements, not yet a frozen mathematical rule or passing test.

## Novelty, alternatives and experiment-design handoff

After stripping labels, copying the two published rules has intentional high
same-claim overlap; it can test expressibility but cannot establish algorithm
novelty. Combining realized utility and current pressure is a plausible
experiment candidate, but generic feedback-control overlap is high and this
two-paper boundary cannot decide novelty. Do not narrow or rewrite the paper's
author-selected mechanism/agent claims in response to that uncertainty.

The next scientific question is whether feedback can retain useful speculative
overlap while avoiding costly unused admission under changing request groups.
The competing positions are demand-only, the original FineMoE selector,
fixed-byte admission, and the candidate adaptive native/BPF pair. Equal tuning
budgets, common information, a shared executor, separate development inputs and
held-out blocks matter more than adding weak variants. Demand-only winning
contradicts a net speculation benefit; fixed-byte matching/winning defeats the
need for adaptation; original FineMoE winning shows lost useful overlap.

The accepted FineMoE paper and official demo/MT-Bench assets are reusable
precedents, subject to the already documented model, offline-history, short
input and repaired-executor deviations. Neither new paper is silently promoted
from a citation precedent to an official runnable artifact. A future
experiment-design node should freeze an output-preserving workload, shared
snapshot/actuator, development-only parameter search, held-out request shifts,
paired repetitions, and demand-progress gates before any GPU launch. Report
drain-inclusive wall throughput and actual demand wait/backlog alongside
logical completed useful/evicted-unused/censored bytes. Correlation alone does
not isolate transfer-time or controller-overhead causality.

Canonical updates correct the obsolete FineMoE/Hummingbird/POD preparation
statuses using their completed reports, while retaining their negative
results, component scope and host/device verification boundaries. They add
the two PDF inventory rows and the explicitly unexecuted candidate. No paper
section, workload protocol, runtime, raw data or dependency was modified.

## Independent review invocation and outcome

One OpenCode process started at approximately 2026-09-03 13:07:57 UTC with the
configured default model (no `-m` override), `--agent plan --format json`, and:

```json
{"snapshot":false,"permission":{"*":"deny","read":"allow","glob":"allow","grep":"allow","list":"allow"}}
```

The process inherits CPU affinity 17. No `--pure`, model override or approval
bypass was used. The full input is `opencode-prompt.md`. It limits the reviewer
to two paper extracts and four explicitly named local files, at most eight
read calls and a final reply of at most 600 Chinese characters. The process
completed with exit code 0; its final stop event is 2026-09-03 13:16:47.885 UTC.
Exactly seven `read` calls accessed the two paper extracts and four allowed
local files; there were no other tool calls. The final contains 412 Han
characters, 1,073 Unicode characters including English/punctuation. No second
model invocation or revision loop was used. The closed `opencode-events.jsonl`
contains 130,603 bytes; `opencode-stderr.log` is empty. The transcript below is
the complete final reply, preserved rather than treated as authoritative fact.
Root already persisted the two PDFs and initial version/source report in
commit `511e204`; that bookkeeping is not a research gate.

### Complete OpenCode final reply

> **判定与ABI**：Least-Stale（specmd §4.2，L380–398）只决定驱逐受害者的 stale/current 双堆排序；UrgenGo（§4.2 Eq.2 L354–359；§4.4.3–4.4.5 L519–599）决定流优先级绑定、延迟发射、批量同步，执行器是 cuLaunchKernel 拦截、睡眠循环与批同步。二者皆主机侧（specmd L326 hook；urgengo L571 ROS2），无设备 BPF；本仓亦仅选择器端口、主机 JIT（results L14、L125），非作者完整系统。fm_input（finemoe_policy.h L7–10）仅概率/top_k/阈值，fm_output 仅 mask，无在途字节、需求积压字段与预算执行器，需新共享 ABI；copy_ledger.h 有完成副本结局（first_use L86、evicted L97、截尾 L117–124）但只有累计计数，无瞬时gauge，锁下可变快照不满足双方不可变带时间戳快照（idle_policy.h L13–17 才是正确范式）。
>
> **去名后**：该规则近似分位数过滤预取（specmd §3.1.4 L319–322）、T_Hurgent 阈值（L511–513）与常规反馈控制的组合，非 Least-Stale 改名；新颖性未决，不能由这两篇推出。
>
> **关键问题**：反馈延迟且 resident 右截尾（results L57–58）不得计为失败；ledger 缺"在途字节/需求队列深度"gauge，累计计数≠当前积压；预算归零即无新反馈致锁死，须最小探测下限（参 UrgenGo L546–555 分位阈值与豁免）；决策只许用快照时刻前事件，BPF 的 fm_add 整数浮点（L38–63）示 native 奇偶性脆弱，快照应携预计算 binary64；准入须计"started 未完成"防重复计量。
>
> **最小实验**：同一冻结负载、5配对块、四臂：demand-only、原 FineMoE（native+BPF）、固定字节预算（同调参次数）、自适应规则；主指标=吞吐比CI。demand-only 赢→投机不划算、控制器退化为纯需求；FineMoE 赢→规则无增量；固定预算持平/赢→自适应多余、新颖性坍缩。
>
> **下一步行动**：仅 CPU 回放，扩展 ledger 的在途字节与需求积压瞬时 gauge 加最小探测下限，验证 native/BPF 同快照决策奇偶性，不触及 GPU。

### Source-checked disposition of the review

Accepted: missing live demand/byte fields; a common native/BPF decision ABI;
delayed and censored feedback; bounded reopening; the fixed-budget competing
position; no established novelty or full-system reproduction. These are
reflected in the canonical candidate and experiment-design handoff.

Corrections/limits, not silently incorporated as facts:

- `CopySnapshot` contains both counters and per-copy records (header 24–27),
  not only cumulative counters. `Snapshot()` copies them under a mutex
  (105–107). It lacks the new policy ABI/timestamp/backlog fields, but locking
  does not prove incoherence or prevent an immutable exported copy. Started
  but incomplete records can describe existing in-flight copies; queued but
  unstarted admitted bytes require accounting as well.
- `fm_output` also contains cumulative bits, selected count and status
  (`finemoe_policy.h:11–14`), not only a mask. The existing `fm_add` mechanism
  is not evidence of fragile parity: the completed FineMoE report retains
  actual C/BPF input/mask/enqueue parity and numerical checks. No replacement
  arithmetic or dependency edit is justified by this review.
- Similarity to a percentile or urgency threshold is a high-level analogy,
  not proof that either paper implements the proposed completed-outcome/demand
  feedback rule. Conversely, two papers cannot establish its novelty. A fixed
  baseline winning defeats a measured adaptation benefit, not logically every
  possible novelty claim. For equal-sized experts a byte cap alone is a count
  cap; the canonical text states this explicitly.
- A permanently positive probe floor could harm demand progress. Any bounded
  exploration/reopening must respect the same demand-progress rules; no
  unconditional minimum admission or tuning value is adopted.
- The review says four arms but nests native/BPF within an arm. The current
  handoff is five executable arms: demand-only, original FineMoE, fixed-budget,
  adaptive C, adaptive BPF. It is not a frozen matrix or approved measurement.
  Same-input parity is checked separately from independently evolving online
  feedback; preserve both positive and negative effects.
- The review's host-JIT statement concerns its scoped FineMoE/Hummingbird
  inputs, not the repository's completed POD device-BPF study. Its informal
  line references are reading aids, not substitutes for the source checks.

## Final handoff

Ready for root review: `docs/background-related-work.md`, this detailed report,
the exact prompt and the closed independent-review event/stderr files. The
paper-text extracts are temporary review inputs and need not duplicate the
already retained PDFs in Git. No additional search, runtime edit, CPU replay,
build, GPU launch, service change or experiment result was made. The next
possible task is a separately selected experiment-design/ABI review, not
automatic implementation. LMCache remains paused.
