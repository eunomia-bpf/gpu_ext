# LMCache GDS five-arm performance — five blocks on RTX 5090

Date: 2026-09-06
GPU: NVIDIA GeForce RTX 5090 (31.36 GiB)
Driver: 575.57.08 (explicit `--expected-driver 575.57.08`)
GDS transport: CUDA 12.9 cuFile backend in compatibility mode; `nvidia-fs` is
not loaded and NVMe-to-GPU P2P is not proven on this GPU

## Shared setup (main campaign and pilot)

The 25-cell main campaign
(`raw/gds-five-arm-575-20260906-five-block-formal`) and the completed
single-block pilot (`raw/gds-five-arm-575-20260906-block1-formal`) share:

- driver 575.57.08;
- a 768 MiB KV cache (`--kv-cache-memory-bytes 805306368`) applied to every
  arm, with `--gpu-memory-utilization 0.98`, `--max-num-seqs 1`,
  `--enforce-eager`, `--no-enable-prefix-caching`, model Qwen3-30B-A3B-FP8;
- for the three GDS arms, a 256 MiB GDS staging buffer (`LMCACHE_GDS_BUFFER_SIZE=256`,
  `LMCACHE_USE_GDS=True`, `LMCACHE_GDS_BACKEND=cufile`, `use_direct_io=true`);
- per cell: eight cold requests followed by eight warm requests, each with a
  1,536-token cached prefix and 16 output tokens; warm-phase timing excludes
  server startup, cold population, and the cold-store barriers.

The five arms are `recompute` (no LMCache), `lmcache_cpu` (LMCache CPU local
cache), and three GDS policy modes on LMCache 0.5.4's `GdsBackend`
(`gds_fifo`, `gds_native`, `gds_bpf`). The main campaign ran five rotated
blocks (each arm occupies each block position exactly once), 25/25 cells
measured with 8/8 warm successes and zero warm failures in every cell.

## Main campaign: five-arm medians (5 cells each)

| Arm | warm TTFT median (ms) | output token/s median | requests/s median |
|---|---:|---:|---:|
| recompute | 66.8884 | 31.0098 | 1.9381 |
| LMCache CPU | 74.4118 | 29.7637 | 1.8602 |
| GDS fifo | 78.0768 | 37.4099 | 2.3381 |
| GDS native | 78.4502 | 37.4703 | 2.3419 |
| GDS bpf | 77.4424 | 38.3743 | 2.3984 |

## Same-block BPF-versus-native relative differences

Per-block differences `(bpf - native) / native`:

| Metric | block 0 | block 1 | block 2 | block 3 | block 4 | median | range |
|---|---:|---:|---:|---:|---:|---:|---|
| TTFT (ms) | -1.58% | -1.28% | +0.55% | +2.51% | -6.32% | -1.28% | [-6.32%, +2.51%] |
| output token/s | -0.24% | +2.41% | +0.28% | -3.42% | +5.17% | +0.28% | [-3.42%, +5.17%] |
| requests/s | -0.24% | +2.41% | +0.28% | -3.42% | +5.17% | +0.28% | [-3.42%, +5.17%] |

(requests/s and output token/s differ only by the fixed 16-token output.)
Same-block BPF versus recompute, for context: TTFT +18.22% median
(range +9.10% to +19.95%), output token/s +20.41% median
(range +18.33% to +26.23%); native versus recompute: TTFT +16.46% median
(range +14.04% to +21.51%), output token/s +20.03% median
(range +17.58% to +23.95%).

## What the GDS arms measure here: all-submit mechanism floor

All three GDS arms ran with the default process-level telemetry: the runner
sets only `LMCACHE_GDS_POLICY_MODE` and no `LMCACHE_GDS_POLICY_*` signal
variables, so every admission request is built from the all-zero `Telemetry`
(HBM pressure 0 permille, zero deadline and slack, zero transfer-cost and
recompute-cost estimates, no speculative-recomputable flag). With that input:

- `fifo` unconditionally submits in arrival order;
- in `native` and `bpf`, the decision precedence (demand read must submit;
  recompute only when recompute cost is below the transfer cost and within
  slack; defer only at HBM pressure >= 800 permille for speculative reads or
  >= 600 permille for safe-to-defer writes) also degenerates to submit-now,
  with `bpf` making the identical decision through the live gpubpf UVM hook.

So all three GDS arms executed the same all-submit policy. The measured
numbers are the mechanism floor of the cuFile GDS storage path on this host
(including the per-object admission decision), and no dynamic
pressure-gated deferral, deadline-driven scheduling, or recompute-substitution
benefit is exercised or established by this campaign. The BPF arm additionally
exercises the UVM ioctl decision path per object, which is why the
BPF-versus-native spread in this table describes the observed end-to-end
differences under all-submit: it shows no consistent sign on either metric.
The five observed pairs do not establish equivalence or an overhead bound.

The GDS arms' higher warm throughput (about +20% output token/s versus
recompute in same-block pairs) with worse warm TTFT (about +18%) is an
observed whole-request-mix result on this short-prefix, single-stream
workload. TTFT measures the first response token, whereas output throughput
includes the whole eight-request warm interval, including later decoding and
client overhead. This campaign does not isolate the cause of the shorter
whole-request interval in the GDS arms. It is not evidence that SSD retrieval is faster than
recompute in this configuration, and it is not a BPF policy gain.

## Pilot (single block, completed, same shared setup)

Per arm, one cell each:

| Arm | warm TTFT median (ms) | output token/s | requests/s |
|---|---:|---:|---:|
| recompute | 63.4205 | 32.4355 | 2.0272 |
| LMCache CPU | 72.2240 | 30.8580 | 1.9286 |
| GDS fifo | 76.5925 | 38.4786 | 2.4049 |
| GDS native | 76.6477 | 38.2832 | 2.3927 |
| GDS bpf | 74.4450 | 39.5191 | 2.4699 |

Pilot same-block pairs: BPF versus native TTFT -2.87%, output token/s +3.23%;
BPF versus recompute TTFT +17.38%, output token/s +21.84%. The pilot shows the
same qualitative shape as the main campaign: GDS warm throughput above
recompute with warm TTFT above recompute. Its single pair cannot establish a
BPF/native performance difference.

## Prior failed attempts (retained, not pooled)

- `raw/gds-five-arm-575-20260906-block1`: first block-1 attempt, which used
  the runner's default expected driver 610.43.02 (with 512 MiB staging) on
  this 575.57.08 host. The engine core segfaulted in `cuModuleLoadData` during
  startup, right after model load and JIT MoE-kernel setup, so all five cells
  ended `ready: false` with `server_returncode: 1` ("server never became
  ready") and produced no warm metrics.
- `raw/gds-five-arm-575-20260906-block1-rerun`: re-run under 575.57.08 but
  still with 512 MiB GDS staging. `recompute` and `lmcache_cpu` completed;
  the first GDS cell (`gds_fifo`) then failed LMCache initialization with
  `torch.OutOfMemoryError` (512.00 MiB staging allocation against a GPU whose
  ~30.4 GiB was already held by PyTorch under the 0.98 budget), the engine
  was marked unhealthy and skipped all lookups/stores, and the cell was torn
  down before any store barrier could be satisfied. No GDS cell completed.

Neither failed run is pooled into the pilot or the main campaign; they are
kept only as records of the failed configurations (wrong expected driver;
oversized staging buffer).

## Raw records

The published append-only `raw.jsonl` contains each completed cell record;
`campaign.json` and `summary.json` contain the runner's summaries under
`raw/gds-five-arm-575-20260906-five-block-formal` (main) and
`raw/gds-five-arm-575-20260906-block1-formal` (pilot). Per-cell `result.json`
and `server.log` additionally remain in the local run directories. No correctness,
admission, retry, or result-filtering gates were applied by the performance
runner.
