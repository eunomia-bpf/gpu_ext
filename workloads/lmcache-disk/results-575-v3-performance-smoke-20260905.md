# 575-v3 performance smoke — preliminary evidence report (2026-09-05)

Status: **preliminary only**. Two complete, validated three-cell blocks under
`raw/storage-575-v3-performance-smoke-01` and
`raw/storage-575-v3-performance-smoke-02`. This is a performance smoke on the
575 stack: no correctness claim beyond the six recorded `validate-cell`
passes, no p99 or tail-latency metric, and no gpubpf arm.

## Environment

- RTX 5090, NVIDIA driver `575.57.08`, kernel `6.15.11-061511-generic`.
- `Qwen3-30B-A3B-FP8`, official vLLM `0.27.1+cu129`, LMCache `0.5.4`.
- 8 frozen prefix pairs, 1,536 cached tokens per prefix, 16 output tokens,
  sequential warm requests (same workload as the 575-v3 storage protocol).

## Blocks

Arm order within each block follows the frozen randomized schedule.
Throughput is warm-phase output tokens/s; TTFT and E2E are medians (ms)
across the eight warm requests.

### `raw/storage-575-v3-performance-smoke-01` (order: cpu, recompute, disk)

| arm | throughput (out tok/s) | warm TTFT median (ms) | warm E2E median (ms) |
|---|---:|---:|---:|
| lmcache_cpu | 26.5797613185 | 69.012931 | 502.37376 |
| recompute | 32.4564843623 | 63.248073 | 493.0837405 |
| lmcache_disk | 30.2470254721 | 93.224653 | 522.334264 |

### `raw/storage-575-v3-performance-smoke-02` (order: recompute, disk, cpu)

| arm | throughput (out tok/s) | warm TTFT median (ms) | warm E2E median (ms) |
|---|---:|---:|---:|
| recompute | 32.1658963826 | 63.768952 | 492.100143 |
| lmcache_disk | 28.6376004328 | 96.8897195 | 558.120222 |
| lmcache_cpu | 29.5246952337 | 75.8736415 | 542.8025165 |

All six cells passed `validate-cell`. In each block the disk arm reported
8/8 warm hits at exactly 1,536 tokens and a stable disk footprint of
48 files / 1,207,959,552 bytes.

## Two-block summary

Median over the two complete blocks.

| arm | median throughput (out tok/s) | vs recompute |
|---|---:|---:|
| recompute | 32.3111903725 | — |
| lmcache_disk | 29.4423129524 | -8.879% |
| lmcache_cpu | 28.0522282761 | -13.181% |

## Scope

Preliminary engagement and order-stability evidence only. Two blocks are
insufficient for the frozen formal-protocol analysis (paired intervals,
classification rules). No correctness claim, no p99, no gpubpf arm.
