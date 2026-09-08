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
The next action is a real pp512 prefill retry, not a separate test campaign.
