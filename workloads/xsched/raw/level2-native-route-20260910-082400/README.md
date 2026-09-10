# XSched Level-2 matched native route comparison: five blocks, four arms

2026-09-10: the matched multi-arm campaign with the original cuXtra sm_120
actuator completes. Five randomized paired blocks run four arms each, 20
cells total: `baseline` (no XSched), `l1_native` (same HAL, upstream
Level-1 queue actuation), `l2_cuxtra` (original cuXtra/SASS Level-2
actuator, native xserver HPF), and `l2_bpfhost` (same cuXtra actuator,
host HPF decisions by the BPF xserver). All 20 cells complete: 400 LC and
800 BE service records each, zero sample diagnostics, and every engagement
gate passes (level-2 queues audited on the two cuXtra arms, level-1
otherwise, no audit or probe lines on the baseline). This campaign was the
open item left by the single-cell
[bring-up](../level2-native-retabs-20260909.oJDCbU/README.md).

## Configuration and components

The runner
([run_native_blob.py](../../level2/native/run_native_blob.py)) ran with
its default component paths from
[protocol.json](protocol.json): isolated native HAL install
`workloads/xsched/level2-build/.output/native-repro-supervisor-20260910.wEUkbn/install-gcc`
(the reproducible nine-patch source build committed at `013427f5`, built
with `/usr/bin/gcc`, `libcuda.so.1` shim-linked to
`libshimcuda.so`), its `bin/xserver` for the native HPF arm,
`workloads/xsched/build/xserver-bpftime` with
`workloads/xsched/build/bpftime_hpf.bin` for the BPF arm, and the service-only
workload
`workloads/xsched/level2-build/.output/service-mismatch-20260908.sHSYtE/priority_workload`.

Shape per cell: 2 LC + 4 BE processes, 4 streams each (24 XQueues), 50
kernels per stream, 9,511,106 recurrence iterations, 340 blocks, 256
threads, service-only output branch (`XG_SERVICE_ONLY=1`). Within each
block the arms run in round-robin order (the start arm rotates per block);
block 1 started at `l2_cuxtra`. Isolation is `LD_LIBRARY_PATH`-based, so
the baseline arm carries no shim and no xserver. The cuXtra arms select
the original binary-surgery route (`XSCHED_LEVEL2_TOOL_ACTUATOR=0`,
`XSCHED_CUDA_LV2_PORT_120=1`, presence-based
`XG_NATIVE_META_EXTEND=1` / `XG_NATIVE_META_KPARAM=1`). No NVBit tool, no
`LD_PRELOAD`, no tool-entry accounting; the runner takes no GPU lock (root
wraps the GPU lease).

Run start 2026-09-10T08:24:00Z; last cell finished 01:52:54 PDT and
`summary.json` was written 01:52:59 PDT.

## Results

Per-arm marginal medians over the five gate-passing blocks
(`result.json` values; service p99 in ms, BE throughput in kernels/s):

| Metric | baseline | l1_native | l2_cuxtra | l2_bpfhost |
| --- | ---: | ---: | ---: | ---: |
| LC service p99 (ms) | 2079.286752 | 1195.381632 | 960.347488 | 960.418112 |
| LC service mean (ms) | 1421.074600 | 497.848534 | 486.476463 | 484.511805 |
| BE service p99 (ms) | 2104.474976 | 1432.591424 | 1432.884704 | 1449.131904 |
| BE throughput (kernels/s) | 10.179290 | 10.128087 | 10.141144 | 10.134117 |

Within-block paired statistics, median of per-block
`pct(candidate - reference)/reference`
(`summary.json`; for p99, negative means the candidate's tail is lower;
for BE throughput, positive means the candidate is faster):

| Pair | LC p99 | LC mean | BE throughput |
| --- | ---: | ---: | ---: |
| l2_cuxtra vs l1_native | -19.6619% | -2.2842% | +0.1048% |
| l1_native vs baseline | -42.5100% | -64.9620% | -0.3233% |
| l2_cuxtra vs baseline | -53.9353% | -65.8720% | -0.3350% |
| l2_bpfhost vs l2_cuxtra | -0.0047% | -0.0970% | -0.0572% |

The original-actuator Level-2 route (l2_cuxtra) cuts LC p99 by about 19.7%
relative to the same-source Level-1 arm and about 53.9% relative to the
no-policy baseline, with the per-block sign consistent in all five blocks.
Moving host HPF decisions from the native xserver to BPF (l2_bpfhost)
leaves the cuXtra service numbers essentially unchanged (within about
0.4% per block). BE throughput stays within about 0.4% of the baseline
across all arms.

Each cuXtra cell carries one to three type-2 resume relaunches in each of
the four BE process logs (the bring-up gate), confirming the captured SASS
guardian and resume blobs operate across the whole campaign, not only in
the single bring-up cell.

## Scope and limits

These are GPU-service and host-elapsed measurements on one RTX 5090, not
arrival-to-completion or queueing latency. The comparison is
original-actuator native cuXtra versus Level-1 and baseline on the same
workload; it is not the separately measured
[shared-actuator policy-port campaign](../level2-device-policy-pair-20260909.buBjns/README.md)
and must not be relabeled as it. The `summary.json` medians are
recomputed CPU-only from these `result.json` files by
`scripts/artifact/reanalyze.py --campaign xsched-route`, which cross-checks
them against the runner's `summary.json` (all match).

Absolute paths in `protocol.json` document this execution; they are not a
fresh-machine installer. Raw cells, worker logs, `protocol.json`, and
`summary.json` are retained in this directory.
