# RTX 5090 STRICT warp-map scaling preflight 575-06

Date: 2026-09-04
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08 (CUDA 12.9, bpftime `9a8af41`,
`build-table1-575-strict`)
Result directory: `raw/strict-warp-scaling-preflight-575-06`

## Outcome

- Run status: **valid (execution-only preflight)**
- Tested hypothesis: **not tested**
- Raw arm processes: **20/20** fresh processes returned code zero
- Attached cells with one target-PID `mode=STRICT` acceptance: **15/15**
- Effects computed: **none** — a preflight supplies no paired or cross-shape
  estimate and is not a paper result.

## Command

```sh
python3 run_strict_warp_map_scaling.py --phase preflight \
  --output raw/strict-warp-scaling-preflight-575-06
python3 analyze_strict_warp_map_scaling.py \
  raw/strict-warp-scaling-preflight-575-06
```

The design is one complete four-arm block per shape (32, 128, 256, 512, and
1024 threads) at one warmup and four measured launches per process, in the
frozen seed-1797 arm order. Each process is a fresh `LD_PRELOAD` launch of the
same single-block target binary.

## Admission, engagement, and correctness gates

All 20 cells recorded the RTX 5090 at compute 12.0, the frozen shape/warp
marker, and an exact CUDA output record with zero mismatch. The 15 attached
cells (noop, `shared_update`, and `warp_update` at every shape) each show:

- return code zero and `STRICT` in the execution record, with exactly one
  target-PID `mode=STRICT` acceptance and one verifier timing record; the
  instruction counts are 2 for no-op, 14 for the shared-key update, and 23 for
  the warp-key update;
- one verified map descriptor of type 1503, key size 4, value size 8, and 64
  entries, with no reject, skip, or verifier-unavailable record;
- exactly one `matched=1` target transformation, one patched PTX module load,
  one successful attach, one loader ready marker, and one detach marker;
- the planned final map effect: no-op leaves zero nonzero entries,
  `shared_update` leaves exactly key 0, and `warp_update` leaves 4, 4, 8, 16,
  and 32 distinct nonzero keys at shapes 32, 128, 256, 512, and 1024, every
  value equal to `magic XOR key`.

The 15 target PIDs and 15 private shared-memory names are unique, and no
target process or `fig15_warp_*` segment survived the campaign.

## Descriptive medians (not a result)

| Shape | native | noop | shared_update | warp_update |
|---|---:|---:|---:|---:|
| 32 | 2.216000 | 4.944000 | 6.640000 | 6.944000 |
| 128 | 1.888000 | 4.848000 | 6.384000 | 6.584000 |
| 256 | 1.944000 | 4.976000 | 6.560000 | 6.776000 |
| 512 | 2.040000 | 4.896000 | 6.352000 | 6.648000 |
| 1024 | 2.128000 | 4.848000 | 6.488000 | 6.648000 |

Four measured launches per process make these medians a plumbing check only;
the campaign-level absolute levels differ from the full run and must not be
compared with it. Attempts 01-05 are retained as failures and contribute
nothing here; see
`results-preflight-575-05-failed-20260904.md`.
