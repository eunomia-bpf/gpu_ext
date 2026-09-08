# Rebuilt full-record buffer: 32 banks, record-major stores

`build.sh` completes with exit zero under both shared experiment leases.
BPF compilation, skeleton generation and host collector linking succeed
without warnings. `build.log` records the actual BTF value size 335675408,
matching the C layout after splitting the unchanged total event capacity
into 32 banks of 16384 slots. Each thread still has 256 full 80-byte records.

Local Qwen supplied record-major writer/collector indexing and format casts;
root applied the bounded bank constants/layout-size repair. The initial
eight-bank failure remains in `../full-record-device-buffer-first-20260908.BytFAr/`.
Generated object, skeleton and executable remain local, excluded from Git.
## First prefill completed

`execute.py` ran the existing pp512/tg0 workload with the built `4dd36b63`
writer/collector and bpftime `886b4ca`. Client and collector both exit zero.
Prefill throughput is **23229.857626 token/s**. The collector observes all
23068672 full 80-byte records, 524288 active slots, zero overflow and zero
out-of-range events; all recorded timestamps are nonzero. It copies
10741613056 bytes in 1457517662 ns after client completion, reported
separately from prefill. The private segment is removed and GPU is idle.

This is one initial measurement, not a five-block paired result or a fresh
baseline-relative overhead number. The BPF storage implementation changes
from ring output to a GPU-local array: it is not the same-object compiler
automatic-warp experiment. Logical record fields/capacity remain unchanged.
Keep this successful cell as the first GPU-local observation when filling
the paired comparison; do not rerun it. Old Table 1 and failed eight-bank
records remain unchanged.
