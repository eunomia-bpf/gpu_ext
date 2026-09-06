# LMCache/UVM-KV + gpubpf KV-range canary — tracked result (2026-09-06)

Source: raw run
[raw summary](raw/lmcache-uvm-kv-gpubpf-range-canary-575-01/summary.md),
[raw result.json](raw/lmcache-uvm-kv-gpubpf-range-canary-575-01/block-00/position-0-lmcache_disk_uvm_kv_gpubpf_debt/result.json),
[raw loader.log](raw/lmcache-uvm-kv-gpubpf-range-canary-575-01/block-00/position-0-lmcache_disk_uvm_kv_gpubpf_debt/loader.log),
and [raw server.log](raw/lmcache-uvm-kv-gpubpf-range-canary-575-01/block-00/position-0-lmcache_disk_uvm_kv_gpubpf_debt/server.log)
(kind `lmcache_uvm_kv_perf`, timestamp `20260906T004614`).

Environment: NVIDIA GeForce RTX 5090 (31.36 GiB), driver parameter
`575.57.08`, CUDA 12.9, vLLM 0.27.1, LMCache v0.5.4-g3e11b8ed,
Qwen/Qwen3-30B-A3B-FP8 (FP8, enforce-eager, `--max-model-len 4096`,
`--gpu-memory-utilization 0.98`, `--max-num-seqs 1`,
`--no-enable-prefix-caching`, port 18080, server pinned to CPUs 8-15 via
`taskset`, worker affinity 0-23), UVM-KV plugin v3 (`uvm_allocator.so`),
LMCache local disk + ODirect (chunk size 256, 16 GiB disk cap, local CPU
disabled), kernel version not recorded in this raw run (the prior coarse
canary recorded `6.15.11-061511-generic`).

## Selected arm, one cell

One selected arm only: `lmcache_disk_uvm_kv_gpubpf_debt`, 1 block, 1 cell
(position 0), 8 cold population requests + 8 warm requests (8 prefixes x
1,536 cached tokens, 16 output tokens). No correctness, engagement,
admission, retry, or filtering gates ran (`retry: false`,
`result_filtering: false`, `attempts_per_cell: 1`); server return code and
loader state are preserved. All 8 cold-store barriers were satisfied
(stored 1536/1536 tokens per prefix).

| arm | ready | warm TTFT median (ms) | warm req/s | warm out tok/s | warm ok/failed | server rc | loader rc |
|---|---|---:|---:|---:|---:|---:|---:|
| lmcache_disk_uvm_kv_gpubpf_debt | true | 93.841292 | 1.87116 | 29.93856 | 8/0 | 0 | 0 |

Exact warm-phase values from result.json: median 93.841292 ms,
p95 106.4789971 ms, max 102.494026 ms,
TTFT values [102.494026, 94.097787, 93.50345, 95.248624, 95.141433,
93.584797, 91.907491, 92.752296] ms; 8 requests / 128 output tokens in
4.275422 s (1.871160 req/s, 29.938564 out tok/s); 8/8 warm requests
returned HTTP 200 and each warm request retrieved 1,536/1,536 cached
tokens from LMCache disk.

## KV-range / tracked / durable evidence (loader.log)

- All four struct_ops programs (`gpu_block_activate`, `gpu_block_access`,
  `gpu_evict_prepare`, `gpu_page_prefetch`) loaded and attached against
  `nvidia_uvm` (struct `gpu_mem_ops`, instance `uvm_ops_debt`); the kernel
  hooks `bpf_gpu_request_reorder` and `bpf_gpu_set_prefetch_region`
  resolved inside `nvidia_uvm`.
- `uprobe`/`uretprobe` attached to `uvm_kv_malloc` in `uvm_allocator.so`.
- Loader configuration: debt cap 4, prefetch-suppression pressure
  threshold 32, warm-phase disk-durable flag off until the warm signal,
  sampling only chunks inside the recorded KV pool range (single-largest-
  allocation tracking).
- Before the warm signal, four statistics dumps showed `Tracked chunks: 0
  (KV-range: 0, KV disk-durable: 0)` and `KV pool range: none captured
  yet`.
- The pool range was then captured: `KV pool range:
  [0x72263a000000, 0x72263ba00000) tgid 2168759 (26 MiB; largest
  uvm_kv_malloc)`; tgid 2168759 is the vLLM EngineCore PID in server.log.
- Chunk accounting at capture: `Tracked chunks: 624 (KV-range: 13, KV
  disk-durable: 0)` — 624 activated chunks tracked for the single engine
  PID, of which 13 fell inside the recorded KV pool range.
- After the warm signal (`"w\n"` written, loader key `w`): `warm-phase
  disk-durable flag: ON; retroactively marked 13 of 624 tracked chunks
  disk-durable (KV-range entries only)`, giving final
  `Tracked chunks: 624 (KV-range: 13, KV disk-durable: 13)`.
- Final totals: `activated=624 used=891 saved=0 evicted=0`; aggregate
  debt pressure 0; per-PID 2168759: `active=624 used=891 saved=0
  evicted=0`.
- Coverage caveat (loader's own note): because tracking is
  single-largest-allocation, a KV pool split across smaller allocations is
  only partially covered; 13 of 624 is the in-range sample, not full-pool
  coverage.

## Cautious comparison to the prior coarse canary

The prior coarse run
([results-575-gpubpf-recoverability-canary-20260906.md](results-575-gpubpf-recoverability-canary-20260906.md),
raw debt canary 575-01, timestamp `20260906T002502`) used the same arm
name and shape: 1 block, 1 cell, 8 warm requests (8 prefixes x 1,536
cached tokens, 16 output tokens).

| quantity | prior coarse (00:25:02) | this KV-range (00:46:14) |
|---|---:|---:|
| warm TTFT median (ms) | 95.6518 | 93.841292 |
| warm req/s | 1.8509 | 1.87116 |
| warm out tok/s | 29.6146 | 29.93856 |
| warm ok/failed | 8/0 | 8/0 |
| tracked chunks | 624 | 624 |
| disk-durable chunks | 624 (coarse warm-phase signal, no range capture) | 13 (KV-range entries only) |
| used | 930 | 891 |
| saved / evicted | 0 / 0 | 0 / 0 |
| aggregate debt pressure | 0 | 0 |
| server rc | 0 | 0 |

The two campaigns are non-contemporaneous (~21 minutes apart) and the
policy builds behave differently on the warm path (the prior run
retroactively marked all 624 tracked chunks disk-durable from a coarse
warm-phase signal; this run marks only the 13 chunks inside the recorded
KV pool range), and their `used` counts differ (930 vs 891). The
performance deltas above are therefore not a controlled measurement: they
are not evidence of a performance win or loss, only that the range-scoped
policy did not disturb the no-pressure path.

## Interpretation

This no-pressure workload establishes per-range cross-layer mechanism
engagement at low apparent overhead: the user-space `uvm_kv_malloc` probe
and the kernel `nvidia_uvm` struct_ops hooks jointly capture and act on a
recorded KV pool range, tracking 624 activated chunks for the engine PID
with 13 sampled inside the range, and the warm signal marked exactly those
13 range entries disk-durable — while keeping 8/8 success, a 93.841292 ms
warm TTFT median and 29.93856 out tok/s, with zero debt pressure and zero
loader/server errors. It cannot show policy benefit: `saved=0` and
`evicted=0` (nothing was ever put under eviction pressure, so the
save/prefetch machinery never acted), and this is only one non-
contemporaneous block (one cell), so there is no controlled contrast
against the prior coarse policy.

Next needed: a pressure comparison. A pool-scoped oversubscription run —
KV pool sized beyond GPU HBM with real eviction pressure, run
contemporaneously as a head-to-head (policy engaged vs. not engaged) — is
the workload needed to exercise activation, eviction preparation, and
save/prefetch so that `saved`/`evicted` become nonzero and any policy
benefit or overhead can be attributed.
