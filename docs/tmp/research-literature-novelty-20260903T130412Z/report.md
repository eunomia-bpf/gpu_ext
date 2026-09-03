# Bounded literature verification: SpecMD and UrgenGo

Started: 2026-09-03 13:04:12 UTC. Parent: the active revision follow-on
literature node. Owner: `/root/pod_resume`; root owns experiments and Git.
Status: PDF existence, downloaded version and first-page metadata verified;
algorithm/implementation review is in progress. LMCache remains paused.

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

## Remaining work

Read the algorithm sections, inspect official artifact evidence, and run one
bounded default-model OpenCode read-only review of these two papers and no more
than four local related files. Preserve its complete final reply here. Then
update `docs/background-related-work.md` with the concise evidence and corrected
completed-experiment statuses; root reviews and persists each coherent step.
