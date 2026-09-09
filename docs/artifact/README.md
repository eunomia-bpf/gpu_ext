# Artifact documentation

Start at [ARTIFACT.md](../../ARTIFACT.md) for the current paper-to-evidence
map, completed measurements, supplementary results and remaining work.
This folder documents reproduction; it does not contain the manuscript.

## Choose a task

| Task | Entry | Expected outcome |
| --- | --- | --- |
| Recalculate existing measurements | [Reanalysis reference](reanalysis.md) | CPU-only statistics from 70 published cell summaries across four campaigns. |
| Regenerate the two supported paper figures | [Figure instructions](../../ARTIFACT.md#cpu-only-reproduction) | New output files outside the manuscript; no GPU run. |
| Prepare Table 1 tools and runtime | [Runtime dependency map](table1-runtime.md), [fresh runtime build record](../../workloads/llama.cpp/observability_overhead/revision-rq4/raw/runtime-published-build-20260909.IktLa7/README.md) | Tool and runtime-library builds, remaining execution dependencies and measured cohort definitions. |
| Build/run XSched Level-2 | [Build instructions](../../workloads/xsched/level2-build/README.md), [runner instructions](../../workloads/xsched/level2/README.md) | Separate shared-actuator policy ports from original-actuator bring-up. |
| Build components or find LMCache disk experiments | [LMCache runtime guide](lmcache-runtime.md) | Published-source build, storage-request latency versus serving throughput, and remaining runtime dependencies. |

## Where material belongs

- `scripts/artifact/`: portable CPU entrypoints, not historical launch scripts.
- `docs/artifact/`: navigation, dependencies and reusable command instructions.
- `workloads/<system>/`: implementation, workload README and dated reports.
- The workload's existing `raw/` or `results*/`: original small records and
  accompanying analysis. Keep old paths and numbers; link new cohorts separately.
- `docs/experiment/`: implementation/design notes and coordination history.
- `docs/paper/`: separately versioned manuscript, read-only for this work.
- Downloaded models, compiled binaries and regenerable caches: outside Git.

Each dated report should name its source/build/run entrypoints, raw cohort,
arm definitions, units and estimator, then distinguish completed work from
remaining work. A recorded absolute-path command documents that execution;
it is not automatically a portable installer. Do not move historical data
just to make the directory tree look cleaner.
