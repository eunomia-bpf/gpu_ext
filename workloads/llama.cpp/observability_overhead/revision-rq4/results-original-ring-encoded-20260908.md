# Original-object ring transport: ten completed performance pairs

RTX 5090, NVIDIA 575.57.08, CUDA 12.9; llama.cpp TinyLlama Q4_K_M,
prefill 512 tokens, generation zero, one timed repetition per fresh process.
Runtime `241872b`; diagnostic callback counting disabled. Both attached arms
use the same original kernelretsnoop BPF object: per-thread 80-byte events,
256 entries per thread, 524,288 allocated thread slots. No manual warp-leader
guard or event-payload rewrite separates these two arms.

| Configuration | Mean token/s | Median token/s | Mean paired throughput loss vs earlier baseline |
|---|---:|---:|---:|
| No probe, ten retained baseline measurements | 38,325.847 | 38,401.219 | — |
| Original object, legacy transport | 34.514 | 35.352 | 99.9099% |
| Same object, aligned copy + encoded publication | 194.059 | 203.238 | 99.4936% |

All twenty attached benchmarks return zero and record numeric throughput.
The median within-block enabled/disabled ratio is **5.7702x**, with a
paired-bootstrap 95% interval **[5.6511x, 5.8121x]** (10,000 resamples,
seed 1797). Every pair improves; ratios range from 2.8094x to 8.1244x.
The median paired prefill-time reduction is 82.6694%. The low fourth enabled
sample (99.991325 token/s) and second disabled sample (25.918006 token/s)
remain in every summary; no outlier is removed.

This is a substantial improvement over this very slow unoptimized transport,
not a low-overhead result. The enabled configuration still loses about 99.5%
of baseline throughput. It does not recover the historical P40 percentage,
and no new NVBit comparison was run.

## Comparison and completion boundaries

The first attempt completed all ten baselines but failed all twenty loader
starts before timing: the requested 12,301 MiB segment exceeded the old
10 GiB runtime ceiling. The explicit ceiling fix in `241872b` preserves the
default allocation and original object. The resume reused the existing
compiled probe and all ten measured baselines; it ran only the failed
attached cells, in the original rotating within-block off/on order.
Consequently, baseline-normalized losses use **earlier** controls, not a
fresh fully interleaved thirty-cell campaign. The paired off/on comparison
is the primary optimization result.

Every loader was subsequently killed by the existing shutdown helper and
records return code -9. The helper allows eight seconds after SIGINT and
five after SIGTERM before SIGKILL. The collector performs a final drain and
full coordinate analysis after receiving a stop signal. Its final report is
missing; source inspection alone does not establish whether sorting, draining
or another shutdown stage consumed that allowance. These cleanup failures
are retained separately from successful timed benchmarks. No assertion of
complete final event delivery or clean collector shutdown follows from this
performance table. No completed cell is discarded or repeated for this issue.

Both loader and agent environments record the requested mode: automatic
execution 0 for legacy, and automatic execution 1 / ring transport 2 for the
optimized arm. The mode combines aligned-word payload copies with encoded
tail publication; this experiment does not isolate those two changes.
The setup-marker parser reports `absent` because both processes run with
`SPDLOG_LEVEL=warn`, while the runtime emits the transport marker at INFO.
The report therefore distinguishes recorded configuration from an observed
setup message. Ring transport selection is separate from whole-program
warp-leader admission. No clock or logging gate was added.

## Why this does not replace the old Table 1 row

The retained `results-table1-warp-plt-575-06` campaign reports gpubpf/NVBit
kernelretsnoop losses of 90.7051%/99.6210%. Its gpubpf record reports 720,896
events, 16,384 coordinates and 44 entries per thread. Its source patch
explicitly selects a warp leader and emits a compact coordinate/timestamp
record. This new experiment uses the original per-thread object and larger
payload/capacity. Their measurement units are the same, but instrumentation
work differs; the new 5.77x ratio cannot be applied to the old 90.7051% row.
All old RTX 5090 and P40 numbers remain unchanged.

## Artifacts

- Initial failed starts and completed baselines:
  `raw/table1-original-ring-encoded-20260908.iVdbES/`.
- Twenty completed attached cells, reused baseline references, build log,
  commands, loader/agent logs and exit records:
  `raw/table1-original-ring-encoded-resume-20260908.ceOL6M/`.
- `cells.json` contains all thirty numeric records;
  `paired-analysis.json` retains every pair and descriptive statistics.

Recompute without running the GPU:

```sh
python3 raw/table1-original-ring-encoded-resume-20260908.ceOL6M/analyze_pairs.py
```

Large binaries and shared-memory payloads are not published. The existing
private-segment cleanup records removal for every cell. Root ran this batch
serially under both shared GPU/struct-ops locks; the subsequent XSched
component build started only after those locks were released. No manuscript
file was changed.
