# STRICT warp-map scaling on RTX 5090

This experiment is a strict-admitted follow-up to the sibling per-lane strict
map-placement campaign. It measures the practical cost of map-key granularity when
the callback is scalar-per-thread `call.uni` and each launch uses one block.

The workload varies block shape (1, 4, 8, 16, 32 active warps) and compares:

- `native`: no injection baseline,
- `noop`: strict admitted empty callback,
- `shared_update`: strict admitted `warp_id`-independent map write (`key=0`, `value=magic`),
- `warp_update`: strict admitted `key=warpid` map write (`value=magic ^ warpid`).

The strict admission flow is:

1. map readback shows expected final effect,
2. verifier prints a single target-PID `mode=STRICT` accepted line and one timing line,
3. map descriptor is reported as type `1503`, key size `4`, value size `8`, entries `64`,
4. exactly one attached callback program is selected and attached,
5. CUDA output is exact.

Build and run from this directory:

```sh
make
python3 -m unittest -v test_strict_warp_map_scaling.py
python3 run_strict_warp_map_scaling.py --phase preflight \
  --output raw/strict-warp-scaling-preflight-575-01
python3 analyze_strict_warp_map_scaling.py \
  raw/strict-warp-scaling-preflight-575-01
python3 run_strict_warp_map_scaling.py --phase full \
  --output raw/strict-warp-scaling-full-575-01
python3 analyze_strict_warp_map_scaling.py \
  raw/strict-warp-scaling-full-575-01
```

The live runner requires RTX 5090 (compute 12.0), NVIDIA driver `575.57.08`,
CUDA 12.9, and a strict verifier-capable `build-table1-575-strict`.
It refuses ambient eBPF/CUDA-hook injection and reads both lease files in
read-only exclusive mode; this run layout must be used to avoid contention.

Use:

- preflight result: `results-preflight-575-01-20260904.md`
- full result: `results-full-575-01-20260904.md`

Both are produced by `analyze_strict_warp_map_scaling.py` and reviewed in
`independent-review.md`.
