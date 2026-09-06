> **2026-09-06: 时钟公平性门槛已由用户指示废除；本文档中的 clock-state / P0 / 精确时钟对匹配要求不再适用。仅以性能数据为准。**

# Launch-latency attempt 09: frozen retry path

Status: frozen on 2026-09-04 after retaining attempt 08 and before any attempt
09 execution.

Attempt 09 must use these fresh paths:

- lifecycle output:
  `raw/rm-correlation-575-09-endpoint-lifecycle`
- child preflight output:
  `raw/rm-correlation-575-09-endpoint-lifecycle/launchlate-preflight`
- lifecycle staging directory:
  `/tmp/gpubpf-endpoint-modules-575-09`

None of these paths may exist when the lifecycle begins. Attempt-08 artifacts
must not be moved, copied, resumed, or reclassified.

The experiment definition and admission gates remain those in
`launchlate-frozen-plan.md`, `launchlate-frozen-plan-v2.md`, and
`endpoint-module-lifecycle.md`. In particular, both 200-sample clock controls,
all three correctness arms, exact launch engagement, three-anchor affine
held-out validation, raw evidence closure, safety, and rollback must pass.
This document changes only the retry path and incorporates the runner's
config-specific gpubpf timing-cell directory fix; it relaxes no threshold.
