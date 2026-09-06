# LMCache/UVM-KV + gpubpf migration-debt canary — tracked result (2026-09-06)

Source: [raw summary](raw/lmcache-uvm-kv-gpubpf-debt-canary-575-01/summary.md),
[raw result.json](raw/lmcache-uvm-kv-gpubpf-debt-canary-575-01/block-00/position-0-lmcache_disk_uvm_kv_gpubpf_debt/result.json),
and [raw loader.log](raw/lmcache-uvm-kv-gpubpf-debt-canary-575-01/block-00/position-0-lmcache_disk_uvm_kv_gpubpf_debt/loader.log)
(kind `lmcache_uvm_kv_perf`, timestamp `20260906T002502`).

Environment: RTX 5090, driver parameter `575.57.08`, kernel
`6.15.11-061511-generic`, Qwen/Qwen3-30B-A3B-FP8 (FP8, enforce-eager,
`--max-num-seqs 1`, UVM-KV plugin v3, LMCache local disk + ODirect).

## Selected arm, one cell

One selected arm only: `lmcache_disk_uvm_kv_gpubpf_debt`, 1 block, 1 cell
(position 0), 8 warm requests (8 prefixes x 1,536 cached tokens, 16 output
tokens). No correctness, engagement, admission, retry, or filtering gates
ran; the server return code is preserved.

| arm | ready | warm TTFT median (ms) | warm req/s | warm out tok/s | warm ok/failed | server rc |
|---|---|---:|---:|---:|---:|---:|
| lmcache_disk_uvm_kv_gpubpf_debt | true | 95.6518 | 1.8509 | 29.6146 | 8/0 | 0 |

## gpubpf loader evidence (loader.log)

- Loader `eviction_debt -w 0`: ready `true`, returncode `0`; warm key `w`
  sent (`"w\n"` written).
- After the warm signal: `warm-phase disk-durable flag: ON`, retroactively
  marked 624 tracked chunks disk-durable (coarse warm-phase signal; exact
  LMCache chunk -> UVM page identity unavailable).
- Final callback totals: activated=624, used=930, saved=0, evicted=0;
  aggregate debt pressure 0; tracked chunks 624 (single PID, the vLLM
  engine worker).
- All four struct_ops programs (`gpu_block_activate`, `gpu_block_access`,
  `gpu_evict_prepare`, `gpu_page_prefetch`) loaded and attached against
  `nvidia_uvm` on this driver/kernel.

## Cautious comparison to the prior UVM-KV arm

The prior five-block ordinary UVM+disk head-to-head
([results-575-uvm-kv-h2h-20260905.md](results-575-uvm-kv-h2h-20260905.md))
reports a `lmcache_disk_uvm_kv` median warm TTFT of 95.2314 ms and
29.0509 out tok/s (5 cells per arm, 2026-09-05). This canary's single
sample (95.6518 ms, 29.6146 tok/s) sits within that arm's spread, but the
two campaigns are non-contemporaneous and this is one cell in one block:
this is not evidence of a performance win or loss, only that the attached
policy did not blow up the no-pressure path.

## Interpretation

This no-pressure workload measures mechanism overhead and expressibility
only: the gpubpf policy attaches, tracks 624 activated chunks with 930
uses, retroactively marks warm chunks disk-durable, and keeps debt
pressure, saves, and evictions at zero while preserving 8/8 success and
parity-level TTFT/throughput. It cannot show policy benefit because
nothing pressures the pool; pool-scoped oversubscription (a KV pool sized
beyond GPU HBM with real eviction pressure) is the workload needed to
exercise the debt machinery for benefit.
