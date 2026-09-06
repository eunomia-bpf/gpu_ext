> **2026-09-06: 时钟公平性门槛已由用户指示废除；本文档中的 clock-state / P0 / 精确时钟对匹配要求不再适用。仅以性能数据为准。**

# Launch-latency attempt 09: pre-live path amendment

Status: recorded on 2026-09-04 after the CPU-only lifecycle dry-run refusal
and before any attempt-09 live execution.

The first attempt-09 dry run used the staging path frozen in
`launchlate-attempt09-frozen-path.md`:
`/tmp/gpubpf-endpoint-modules-575-09`. The lifecycle wrapper rejected it
because staging must be a direct child of its fixed `STAGE_ROOT`,
`/opt/gpubpf/modules/575.57.08`. Validation stopped before directory creation,
module operations, service changes, label changes, clock controls, or workload
execution. Neither the stage nor the attempt-09 output path was created.

This pre-live amendment replaces only the staging path with:

`/opt/gpubpf/modules/575.57.08/launchlate-endpoint-stage-575-09`

The lifecycle output remains
`raw/rm-correlation-575-09-endpoint-lifecycle`, and its child campaign remains
`raw/rm-correlation-575-09-endpoint-lifecycle/launchlate-preflight`. All paths
must be fresh at execution. Every experiment, lifecycle, calibration,
correctness, engagement, clock, raw-closure, safety, cleanup, and rollback gate
remains unchanged. No threshold is relaxed, and no attempt-08 data is reused or
reclassified.
