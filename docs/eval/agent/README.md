# Policy-exploration study artifacts

This index covers revision R7. The analysis and benchmark sources below are
public; the original prompt/interaction-log release is **not complete**.
The six Q1–Q6 reports are derived notes, not raw conversations or substitute
prompts. Benchmark sources alone do not reproduce the agent's decisions.

The 2026-09-03 [inventory](../../experiment/revision-artifact-inventory.md)
confirms that the 25 primary and 259 nested study transcripts remain missing
in the checked locations. [Current reproduction prompts](reproduction-prompts.md)
are newly authored, separately labelled templates for the revision's policy
ports and raw audits; they are not recovered original prompts or a completed
new agent-study trajectory.

## Analysis and benchmark entry points

| Material | Public entry point | Scope |
| --- | --- | --- |
| Policy and run inventory | [Q1](q1_git_archaeology.md) | Historical commits, policy paths, and result paths |
| Safety analysis | [Q2](q2_safety_taxonomy.md), [Q5](q5_safety_events_from_sessions.md) | Derived classifications; require source logs for independent audit |
| Exploration cases | [Q3](q3_case_studies.md), [Q4](q4_session_exploration_log.md) | Case narratives and session inventory |
| Session metrics | [Q6](q6_precise_metrics.md), [extractor](../../../scripts/analysis/extract_claude_q6_metrics.py) | Historical window/token/tool-call analysis |
| FAISS phase exploration | [benchmark](../../../workloads/faiss/bench_gpu_1bn.py), [setup](../../../workloads/faiss/README.md) | SIFT IVF build/search; policy history is mapped in Q1/Q3 |
| GPU preemption | [benchmark](../../../scripts/extension/preempt/bench_preempt_kfunc.py), [test launcher](../../../scripts/extension/preempt/run_preempt_kfunc_test.sh) | Preemption-kfunc case from Q3/Q6 |
| Cross-block analysis | [analysis](../../../workloads/llama.cpp/analyze_crossblock_v3.py), [plan](../../experiment/plans/cross_block_prefetch.md) | Historical cross-block exploration, not a fresh 610 result |
| GNN workload | [benchmark](../../../workloads/pytorch/benchmark_gnn_uvm.py) | Training/oversubscription workload referenced in policy exploration |

The old GPU launchers may unload modules or clean up policies; inspect their
commands and use an isolated, authorized test machine. Do not run them on a
shared GPU to check this index. The current 610 port does not retroactively
qualify these historical experiment paths.

Once the archived corpus is restored, run from the repository root:

```bash
python3 scripts/analysis/extract_claude_q6_metrics.py \
  --corpus /path/to/archived/project-transcripts --output /tmp/q6-recomputed.md
```

Preserve each top-level session JSONL and its corresponding nested transcript
directory. This command preserves the historical analysis algorithm and cost
assumptions; it is not a current pricing estimate. It refuses a corpus missing
the required study sessions before writing a report. CLI/error-path tests do
not establish that the historical totals have been reproduced.

## Missing original materials

The March Q6 report identifies 25 primary and 259 nested transcripts. On
2026-08-31 the original project directory contains only about 40 KiB, and the
referenced sessions such as `6b21980a`, `1ffa360b`, and `b1e7bc20` are absent.
Filename searches of local Claude storage and readable workspace paths found
no matching backup; some unrelated service directories were not readable.
The backup location has been requested from the author. No historical prompt
has been reconstructed from prose, and no new run can replace a missing
original interaction log.

Before publication, recover the original session set, remove credentials and
unrelated private material while retaining study prompts/tool interactions,
document omissions, and recompute the reported metrics from the release copy.
Until then, R7's prompt/log component remains open.
