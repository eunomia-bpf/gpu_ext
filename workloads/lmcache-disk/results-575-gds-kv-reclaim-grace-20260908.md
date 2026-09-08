# LMCache disk-backed reclaim after the common async scheduler repair

Completed September 8, 2026 UTC: five paired blocks, fifteen serving cells.
All 120 cold and 120 warm requests completed; each cell generated 8192 warm
output tokens with no warm HTTP failures, and all server processes exited
zero. No completed serving cell is excluded from this comparison.

The BPF/native paired throughput difference is close to zero at the mean,
but both implementations of the current disk-cost policy are slower than
stock on average. This is evidence for improving the shared policy, not
for attributing the whole gap to BPF execution overhead.

## Configuration and scope

RTX 5090, NVIDIA 575.57.08, Qwen3-30B-A3B-FP8, vLLM 0.27.1 and LMCache
0.5.4; real LMCache GDS/cuFile compatibility-mode disk storage. This does
not establish hardware NVMe-to-GPU P2P. The KV pool is 384 MiB / 4096
tokens, the GDS buffer is 256 MiB, maximum running sequences is two, and
the warm client uses four workers with 250 ms scheduled arrival spacing.
Even-index warm prompts contain 1536 tokens and odd-index prompts contain
1024 tokens; every request has a 1024 output-token bound. The same arrival
rotation is used across arms within each block.

All arms use the same default-async scheduler plus the deferred-free grace
patch (`52af9e82`), which follows the common preemption-seam patch. Stock
means the stock victim-selection algorithm on this repaired runtime,
not an entirely unmodified upstream vLLM artifact. Native and BPF use the
same cost-per-actually-freeable-byte algorithm in `kv_reclaim_abi.h`, the
same backing metadata, and the same fixed 62502 ns/token recompute price.
The native/BPF adapters also select disk-prefix restore versus recompute.
Per-I/O storage admission is FIFO in all three arms.

The benchmark uses the existing `run_gds_kv_reclaim.run_cell` and
`warm_arrival_order` entrypoints. Blocks 1--4 imported runner `eeb61bbb`
before taking the GPU locks; the later partial-stream recording change
`a1011e64` was not loaded into the active process. No workload reduction,
clock test, extra preflight, new timeout, or policy change occurred during
the comparison. Warm elapsed time excludes startup, cold population and
shutdown; throughput is completed warm output tokens / warm elapsed time.
TTFT below is measured from actual request send, not scheduled arrival.

## Throughput

| Block | Stock token/s | Native token/s | BPF token/s | BPF/native change % |
|---|---:|---:|---:|---:|
| 0 | 71.121418 | 67.048495 | 66.099993 | -1.414652 |
| 1 | 68.594004 | 67.518700 | 68.949088 | +2.118507 |
| 2 | 67.430024 | 64.073592 | 65.094250 | +1.592946 |
| 3 | 69.998877 | 70.652086 | 69.958760 | -0.981324 |
| 4 | 72.657860 | 65.590553 | 63.636841 | -2.978650 |
| Median | 69.998877 | 67.048495 | 66.099993 | -0.981324 |

Median across the five per-cell TTFT medians is 26195.968 ms for stock,
30055.502 ms for native, and 30818.840 ms for BPF. These are not pooled
request percentiles. The eight-request per-cell p95 fields in raw records
are retained but are not used to claim a tail-latency improvement here.

Each paired percentage is `100 * (tested / reference - 1)` using throughput
from the same block. Aggregating those differences gives:

| Comparison | Paired mean % | Paired median % | Observed range % | 95% bootstrap interval for paired mean % |
|---|---:|---:|---:|---:|
| BPF / native | -0.332635 | -0.981324 | -2.978650 to +2.118507 | -1.953586 to +1.306763 |
| Native / stock | -4.213133 | -4.977652 | -9.726830 to +0.933172 | -7.294968 to -1.067474 |
| BPF / stock | -4.495951 | -3.463997 | -12.415751 to +0.517661 | -8.872984 to -0.508659 |

The interval exhaustively enumerates all 5^5 = 3125 ordered bootstrap
resamples of the five paired differences, takes the arithmetic mean in
each resample, and uses linearly interpolated 2.5/97.5 percentiles. The
independent unit is a block, not each of the eight requests. Five blocks
remain a small sample; these intervals are not equivalence margins or
general hardware/workload guarantees. BPF exceeds native in two of five
blocks; each policy exceeds stock in only one block.

## Recovery traffic and the next policy question

Counting existing `Time taken for batched_get_blocking` and
`rolled back from` lines yields the following. Native and BPF have identical
counts within each block. These figures cover the saved server logs and
are backend request payloads, not physical SSD traffic measurements.

| Blocks | Stock: batches / MiB / ops / rollbacks | Each policy: batches / MiB / ops / rollbacks |
|---|---|---|
| 0 | 15 / 1632 / 68 / 7 | 42 / 5856 / 244 / 34 |
| 1, 3 | 15 / 1968 / 82 / 7 | 15 / 1968 / 82 / 7 |
| 2, 4 | 15 / 1632 / 68 / 7 | 43 / 6000 / 250 / 35 |

Extra restores are therefore arrival-dependent and shared by native and
BPF. The existing score prices one recovery divided by freeable bytes; it
does not account for repeated restore/preemption cycles or the collateral
cost of re-admitting a request alongside another growing request. That is
the next optimization hypothesis, not a proven complete causal trace.
Importantly, a 1024-token restore for an odd-index request is expected from
the workload; it must not be misdiagnosed as missing disk coverage solely
because even-index requests use 1536-token prefixes.

## Evidence and limitations

The raw root is
[`raw/gds-kv-reclaim-grace-575-20260908-01`](raw/gds-kv-reclaim-grace-575-20260908-01).
Its [paired summary](raw/gds-kv-reclaim-grace-575-20260908-01/paired-summary.json)
lists every source file, extracted per-cell metric and unrounded paired
result. In each source result, the relevant fields are `arm`, `block`,
`warm_phase.output_tokens_per_s`, `warm_phase.warm_ttft_median_ms`,
`warm_phase.requests`, `warm_phase.failures`, and `server_returncode`.

Block 0 uses `native-async`, `bpf-async-absolute`, and
`stock-async-absolute`, executed in that order. Subsequent directories are:

| Block | Position 0 | Position 1 | Position 2 |
|---|---|---|---|
| `block-01` | `position-0-bpf` | `position-1-stock` | `position-2-native` |
| `block-02` | `position-0-stock` | `position-1-native` | `position-2-bpf` |
| `block-03` | `position-0-native` | `position-1-bpf` | `position-2-stock` |
| `block-04` | `position-0-bpf` | `position-1-stock` | `position-2-native` |

Each directory retains `result.json` and `server.log`. The intervening
relative-path startup failure in `bpf-async` is preserved separately; it
never served requests and is not a replacement for the successful absolute
path cell. Earlier bootstrap-failed, non-overlapped, and failed async
campaigns are preserved in the
[chronological integration record](gds-control/automatic-offload-integration-20260907.md).

Exit-time adapter diagnostics are absent in these runs; no new per-cell
callback-count claim is inferred from them. Earlier repaired BPF-sync map
observations do not establish callback counts for these async cells.
The result compares the configured same-policy implementations, not a
standalone microbenchmark of BPF instructions. Likewise, freeing vLLM KV
blocks returns capacity to its fixed pool; it is not proof of returning
physical HBM to another process or transparent same-address disk-UVM paging.

Conclusion: the common scheduler repair enables all planned serving cells
to finish. The current BPF/native performance difference is small relative
to paired variability, while the shared disk-cost policy needs improvement
against stock. Completion of this batch does not close that optimization
or the broader revision work.
