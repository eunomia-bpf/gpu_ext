# Original agent-study artifact inventory — 2026-09-03

Status: **original prompt/transcript release remains incomplete**. This scoped
follow-up to [the existing investigation](../revision-remaining-artifacts.md#3-agent-prompts-original-logs-and-harness-release)
found no recovered original study transcript. Existing repository reports and
harnesses can accompany the release, but cannot replace the missing originals.
The author must supply the original backup/export or its exact location to
continue recovery. This is not a claim that no archive exists elsewhere.

## Scope and evidence

The historical sources are the [public artifact index](../eval/agent/README.md),
[Q6 inventory](../eval/agent/q6_precise_metrics.md),
[Q5 source list](../eval/agent/q5_safety_events_from_sessions.md), and
the extractor's `FOCUS_PREFIXES`. Q6 covers February–March 2026 and records
25 primary plus 259 nested JSONL files; its ten focus sessions are a subset,
not a sufficient completeness check for all 284 historical files.

Checks used file-name inventories and byte sizes on CPU 17. The search covered
the GPU workspace's project files and research/archived-plan directories, plus
the three exact Claude project directories already named by Q5/Q6:

- `/home/yunwei37/.claude/projects/-home-yunwei37-workspace-gpu-gpu-ext`
- `/home/yunwei37/.claude/projects/-home-yunwei37-workspace-gpu`
- `/home/yunwei37/.claude/projects/-home-yunwei37-workspace-gpu-bpftime-gpu-verifier`

Workspace filename searches included hidden/ignored project artifacts but
excluded Git internals, caches, build outputs, dependency environments,
third-party directories, models/datasets and the unrelated JAX source tree.
No filename matched any of the 25 Q6 primary prefixes or the two additional
Q5 source prefixes. No JSONL or archive container was found in the repository's
`docs`, `scripts`, `.agents`, `.claude`, `.codex`, `cd`, or sibling GPU `results`
directories. The named archived-plan directories contain Markdown plans,
not a primary/subagent transcript export. Focus-ID text matches in research
documents lead only to the existing derived reports and extractor.

No credential/configuration contents, unrelated private projects, or raw chat
text were searched or copied. For the two present project-session candidates,
only record type, session ID and timestamp fields were inspected. The extractor
was not run and historical aggregate metrics were not recomputed. No GPU,
build, Git operation, archive extraction or publication was performed.

## Missing historical corpus

All prefixes below come from Q6, not from reconstructed conversations. Every
listed primary is missing in the checked locations. Nested counts are Q6's
historical expectations; no associated nested files were recovered.

| Primary prefix | Required focus | Expected nested files | Current status |
| --- | --- | ---: | --- |
| `9d654c47` | yes | 13 | missing |
| `1499f02c` | no | 0 | missing |
| `32c1ebfa` | no | 0 | missing |
| `cec13454` | no | 0 | missing |
| `00b4c113` | no | 0 | missing |
| `16156e61` | yes | 12 | missing |
| `84403ce2` | no | 0 | missing |
| `ee60df57` | no | 0 | missing |
| `3b3897bc` | no | 0 | missing |
| `9dbdebcd` | no | 0 | missing |
| `d662e303` | yes | 3 | missing |
| `de6eabd4` | yes | 17 | missing |
| `ab2202db` | no | 0 | missing |
| `6b21980a` | yes | 36 | missing |
| `ce9a1079` | no | 1 | missing |
| `6c0aa1fb` | yes | 4 | missing |
| `e19dd100` | yes | 36 | missing |
| `f9e903b2` | no | 0 | missing |
| `b1e7bc20` | yes | 22 | missing |
| `648091fb` | yes | 29 | missing |
| `7cf7718d` | no | 0 | missing |
| `e557f3b2` | no | 0 | missing |
| `1ffa360b` | yes | 84 | missing |
| `7d3f43c8` | no | 0 | missing |
| `05039828` | no | 2 | missing |
| **Total** | **10 focus / 25 primary** | **259** | **0 recovered** |

The additional Q5 sources
`-home-yunwei37-workspace-gpu/0f335699-ec1b-4352-af7f-7f3772bc4d6e.jsonl` and
`-home-yunwei37-workspace-gpu-bpftime-gpu-verifier/8dd19606-7c00-4582-a474-bfe378736d3c.jsonl`
are also absent. They are not added to the Q6 25/259 denominator.

## Present private candidates, not recovered originals

Paths in this table are relative to `/home/yunwei37/.claude/projects/`.
These are existence/metadata observations, not publication or redaction
clearance. No files were copied into the repository.

| Path | Bytes | Source / missing status |
| --- | ---: | --- |
| `-home-yunwei37-workspace-gpu-gpu-ext/1fec0eda-3b22-404f-8b46-a7539c93402e.jsonl` | 1,460 | Six records; timestamped records are 2026-08-04. Not one of Q6's primary sessions. |
| `-home-yunwei37-workspace-gpu-gpu-ext/memory/MEMORY.md` | 17,161 | Project memory notes; not a transcript; contents not inspected for release. |
| `-home-yunwei37-workspace-gpu-gpu-ext/memory/feedback_codex_runs_experiments.md` | 596 | Project memory note, not an original session. |
| `-home-yunwei37-workspace-gpu-gpu-ext/memory/feedback_codex_writes_code.md` | 536 | Project memory note, not an original session. |
| `-home-yunwei37-workspace-gpu/623d1f7e-8e6a-4b74-b521-d2f940bbc750.jsonl` | 1,690,111 | First timestamped snapshot is 2026-08-03; different session from Q5/Q6. |
| `-home-yunwei37-workspace-gpu/623d1f7e-8e6a-4b74-b521-d2f940bbc750/subagents/*.jsonl` | 13,483,398 across 21 files | Nested under that different August session; not the 259 Q6 nested files. |
| `-home-yunwei37-workspace-gpu/sessions-index.json` | 1,734 | Three January session-index entries, not raw conversations or the missing Q5/Q6 sessions. |

The verifier project directory has memory notes but no JSONL transcript.
Later session IDs, memory notes and index entries must not be substituted for
the missing February–March interactions, even if they discuss the same paper.

## Present repository release materials

Paths below are repository-relative; byte sizes were observed on this date.
These are the existing public-index sources, not newly recovered prompts.
File presence is not a fresh executable or publication-permission audit.

| Path | Bytes | Source / release role |
| --- | ---: | --- |
| `docs/eval/agent/README.md` | 3,914 | Public index, explicitly identifies the release gap; updated with current prompt templates. |
| `docs/eval/agent/reproduction-prompts.md` | 4,644 | Newly authored revision templates, not original study prompts or a completed trajectory. |
| `docs/eval/agent/q1_git_archaeology.md` | 59,531 | Derived policy/run inventory. |
| `docs/eval/agent/q2_safety_taxonomy.md` | 20,300 | Derived safety taxonomy. |
| `docs/eval/agent/q3_case_studies.md` | 21,770 | Derived case narratives. |
| `docs/eval/agent/q4_session_exploration_log.md` | 7,674 | Derived session table, not raw sessions. |
| `docs/eval/agent/q5_safety_events_from_sessions.md` | 39,424 | Derived event report with selected excerpts; not a complete corpus. |
| `docs/eval/agent/q6_precise_metrics.md` | 18,973 | Historical aggregate and per-session inventory. |
| `docs/eval/agent_session_analysis.md` | 18,849 | Derived analysis, not original interactions. |
| `scripts/analysis/extract_claude_q6_metrics.py` | 43,481 | CPU extractor; needs the recovered corpus. |
| `workloads/faiss/bench_gpu_1bn.py` | 24,191 | FAISS benchmark source. |
| `workloads/faiss/README.md` | 5,737 | FAISS setup/reproduction instructions. |
| `scripts/extension/preempt/bench_preempt_kfunc.py` | 6,264 | Preemption benchmark source. |
| `scripts/extension/preempt/run_preempt_kfunc_test.sh` | 2,385 | Historical launcher; no fresh runtime qualification. |
| `workloads/llama.cpp/analyze_crossblock_v3.py` | 17,856 | Cross-block analysis source. |
| `docs/experiment/plans/cross_block_prefetch.md` | 50,793 | Retained experiment plan. |
| `workloads/pytorch/benchmark_gnn_uvm.py` | 30,040 | GNN benchmark source. |

The explicitly named research archives are also present as plans, not chats:

| Path | Bytes | Source / release role |
| --- | ---: | --- |
| `docs/experiment/plans/archived/xcoord_v1.md` | 138,725 | Archived experiment plan. |
| `docs/experiment/plans/archived/gpu_preempt_kfunc_v1.md` | 60,510 | Archived experiment plan. |
| `docs/experiment/plans/archived/cross_block_prefetch_v1.md` | 96,049 | Archived experiment plan. |
| `docs/experiment/xcoord/xcoord_plan_old_20260228.md` | 91,518 | Historical experiment-plan snapshot. |
| `docs/experiment/xcoord/bpf_core_access_findings.md` | 10,677 | Mechanism findings, not a transcript. |

## Current instructions and workload inputs are not original agent prompts

| Path | Bytes | Source / missing status |
| --- | ---: | --- |
| `CLAUDE.md` | 6,313 | Current repository guidance; does not establish which instructions the original sessions received. |
| `.claude/skills/paper-writer.md` | 6,310 | Current paper-writing guidance, not a policy-exploration conversation or a newly validated reproduction prompt. |
| `workloads/moe-infinity/prompts.json` | 140,263 | Inference benchmark inputs, not prompts to the policy-generating agent. |
| `workloads/lmcache-disk/prompts.json` | 564,713 | Inference benchmark inputs, not prompts to the policy-generating agent. |

The inventory audit did not reconstruct historical prompts. Subsequently,
the main revision added the separately labelled current reproduction templates
listed above. These may be released under that label, but must not be
presented as the missing historical prompts. Recovery still requires the real
primary/subagent archive, followed by explicit privacy redaction, an omissions
inventory and metric recomputation from the release copy.
