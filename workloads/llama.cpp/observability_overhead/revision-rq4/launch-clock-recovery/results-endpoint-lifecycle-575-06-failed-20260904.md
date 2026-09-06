> **2026-09-06: 时钟公平性门槛已由用户指示废除；本文档中的 clock-state / P0 / 精确时钟对匹配要求不再适用。仅以性能数据为准。**

# Endpoint lifecycle attempt 06: retained child-environment failure

Date: 2026-09-04
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08
Result directory: `raw/rm-correlation-575-06-endpoint-lifecycle`

## Outcome

The endpoint and recovery portions passed, but this attempt is **incomplete and
contributes no launch-latency performance result**. With the endpoint-capable
candidate modules loaded, the fixed direct probe accepted all 200/200 samples,
rejected none, observed zero CPU-midpoint or PTIMER regressions, and reported a
758 ns median bracket width. The child then stopped before any workload cell
because its restricted `PATH` omitted `/usr/local/cuda-12.9/bin`, so the frozen
runner could not execute `cuobjdump` while collecting its initial PTX inventory.

The full campaign was correctly not started. The lifecycle then removed the
candidate and restored the exact admitted `gpreempt-849ea75d-6.15.11` four-module
set. Its final gates show the same boot, device nodes and module parameters;
400 W power limit; active GDM, NVIDIA persistence, and k3s-agent services;
restored scheduling labels; no compute application, UVM reference, `struct_ops`
residue, Xid, or abnormal kernel message. `recovery_errors` is empty.

## Retry boundary

The raw failure is retained and will not be overwritten. The wrapper must add
the pinned CUDA binary directory to its allowlisted child environment and add a
regression test for command discovery. A retry must use fresh stage and output
paths and repeat the whole module lifecycle; it may not resume or relabel this
attempt.
