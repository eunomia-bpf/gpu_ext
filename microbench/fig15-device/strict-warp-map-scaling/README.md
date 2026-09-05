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

The syscall-server creates the shared map/program/link inventory and is checked
for presence plus live readiness.  STRICT device-program verification executes
in the target through `libbpftime-agent.so`, so the verifier log-string
inventory is required from that DSO and the target-PID admission records—not
from the syscall-server binary, whose static link may omit unused attach code.

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

The commands above show the layout; every real attempt writes a fresh, never
reused `raw/` directory, so retained attempts are never overwritten.

The live runner requires RTX 5090 (compute 12.0), NVIDIA driver `575.57.08`,
CUDA 12.9, and a strict verifier-capable `build-table1-575-strict`.
It refuses ambient eBPF/CUDA-hook injection and reads both lease files in
read-only exclusive mode; this run layout must be used to avoid contention.

Use:

- failed preflight attempt 05 (syscall-server CUDA init error 3, no sample):
  `results-preflight-575-05-failed-20260904.md`
- accepted execution-only preflight: `results-preflight-575-06-20260904.md`
- full result: `results-full-575-01-20260904.md`
- independent raw audit and claim boundary: `independent-review.md`

Attempts 01-04 are retained under `raw/` and contribute no sample. The
warp-uniform-key hypothesis is **contradicted**: cross-shape factors from 1 to
32 warps per block are 1.0102 [0.9768, 1.0390], 1.0060 [0.9915, 1.0604], and
1.0062 [0.9878, 1.0257] for shared/noop, warp/noop, and warp/shared. This is
a negative result, kept as repository evidence only, and it is not a
paper-positive claim.

Reported effects come from `analyze_strict_warp_map_scaling.py`, whose
bootstrap seeds are now a deterministic function of frozen seed 1797 and the
fixed shape/comparison indices rather than of `PYTHONHASHSEED`. The 18 tests in
`test_strict_warp_map_scaling.py` pass, and both accepted campaigns replay
byte-identically.
