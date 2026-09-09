# LMCache disk/UVM serving after physical GPU-chunk reclaim

## Complete result, 2026-09-09 PDT

All five rotated blocks / 15 cells finish: each generates 8192 observed output
tokens with zero HTTP failures, no recorded runner error, and server exit zero.
Total warm output is 122880 tokens across 120 requests. The campaign runs
06:47:26–07:27:25 PDT; lifecycle exit is zero and restoration reports success.
The previous incomplete/OOM campaigns remain unchanged.

| Warm generation throughput, token/s | stock | native policy | BPF policy |
| --- | ---: | ---: | ---: |
| Block 0 | 71.698241 | 69.291949 | 66.487342 |
| Block 1 | 70.572913 | 68.387745 | 68.733332 |
| Block 2 | 73.961755 | 67.656516 | 65.422796 |
| Block 3 | 73.575320 | 69.575409 | 65.166163 |
| Block 4 | 73.574518 | 68.608032 | 65.303293 |
| Median | 73.574518 | 68.608032 | 65.422796 |

Paired percentages are calculated within each block, then summarized; they
are not ratios of the marginal medians:

- BPF/native: median −4.047522%, range −6.337362% to +0.505334%; adverse in 4/5 pairs.
- BPF/stock: median −11.241969%, range −11.545101% to −2.606639%; adverse in all pairs.
- Native/stock: median −5.436485%, range −8.524999% to −3.096326%; adverse in all pairs.

The policy does not improve throughput in this workload. The full BPF/native
difference is an end-to-end comparison, not an isolated BPF execution-overhead
measurement or a confidence bound. No equivalence or stable-superiority claim
is supported. Successful completion here does not prove OOM is impossible in
other workloads.

## What is compared

All three arms use the same real LMCache disk backend, disk/UVM promotion
adapter, FIFO per-I/O admission and repaired driver. Stock disables the
additional KV reclaim policy; native and BPF enable the matched reclaim
decision implementations. Stock is therefore not an unmodified upstream
LMCache or a no-disk baseline. Earlier recompute/CPU/disk comparisons remain
separate; this campaign isolates the added residency/recompute policy on the
disk/UVM serving path.

The model is Qwen3-30B-A3B-FP8, eager vLLM, max model length 4096, two engine
sequences, 384 MiB engine KV pool and 256 MiB GDS buffer. Eight warm requests
use alternating 1536/1024-token prefixes, 1024 output tokens each, concurrency
four and 250 ms nominal stagger. Warm order is 4,5,6,7,0,1,2,3. Limited client
concurrency can delay actual sends; raw arrival records retain those delays.
The unchanged 62502 ns/token recompute price is an earlier end-to-end TTFT
proxy, not pure GPU compute or a fresh calibration.

The metric is warm completed output tokens divided by warm elapsed time. It
excludes startup, cold population, cold-store barriers and shutdown. It is
neither Table 1 prefill throughput nor the earlier storage-request p99 metric.
Disk restoration is CPU-staged; these data do not establish NVMe-to-GPU P2P.

Driver commit 95097e20 on revision/gpu-storage-decision-575 adds physical
release of allocated, unmapped, nonresident GPU chunks after offload group
processing. It uses existing MMU/PMM tracker handling; it does not change
the policy or disk transport. The complete kernel build exits zero; the run
restores the saved original module and policy loaders afterward.

## Available counters and limitations

Twelve cells have disk/UVM diagnostics. Each reports 48 prepared ranges,
zero restore errors and zero stock fallbacks. Native/BPF each have 40 restores
where reported; stock has 72,72,82,72 in blocks 0,2,3,4. Fewer restores do not
imply higher throughput: recomputation or other serving work can offset
saved I/O. Exact decision counts are not recoverable from missing policy
diagnostics, so this report does not infer them by subtraction.

Disk/UVM counters are unavailable for block-0 BPF (empty diagnostic),
block-1 stock and block-2 BPF (missing diagnostic). These are unavailable,
not zero; all 15 performance measurements are included without rerunning.

Two small CPU-side compiler actions by local XSched sessions overlapped
serving despite the source-only request. A 16424-byte metadata cubin
completed at 06:51:18 during block-0 native; a 26808-byte host tool object
completed at 07:24:43 during block-4 BPF. Neither command launches a GPU
kernel. Their duration and performance effect were not measured; zero effect
is not assumed and these pairs are not silently excluded. Root's full
callback-tool build starts only after both campaign leases are released.

## Reproduction and retained artifacts

The executed driver build is in build-driver.sh/build-driver.log. The exact
serving command and module lifecycle are in run-serving.sh, lifecycle.log
and runner.log. The lifecycle script records historical PIDs: do not replay
its kill commands without resolving current owners.

cells/campaign.json records the configuration and rotated arm order;
cells/block-*/position-*/result.json and server.log retain every measurement.
analysis.json contains the recomputed medians, per-block percentages and
missing-counter list. To recompute, read warm_phase.output_tokens_per_s from
each result, group by block/arm, and use 100*(candidate/reference-1).
Large generated binaries and disposable KV caches are not Git artifacts.
No manuscript files are changed.
