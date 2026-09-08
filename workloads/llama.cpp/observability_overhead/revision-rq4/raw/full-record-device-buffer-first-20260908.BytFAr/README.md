# First real full-record GPU-buffer invocation: failed before throughput

The initial thread-major implementation at `743ef571` was run once through
the existing Table 1 pp512/tg0 helper on RTX 5090 / driver 575.57.08, using
bpftime `886b4ca`. `execute.py` records the invocation. No baseline or NVBit
cell was repeated, and no clock calibration or additional test batch ran.
Automatic warp execution and diagnostic call counting are disabled. The
collector is the binary from `../full-record-device-buffer-build-20260908.sLXIqf/`.

The llama.cpp process exits -6 with a CUDA error before reporting token/s.
The runner finishes normally and retains `throughput_tok_s: null`; this is
not a successful performance measurement. The collector exits zero after
SIGINT, reports zero committed records and removes its private segment.
The saved execution records confirm normal cleanup; GPU returns to idle
with 1 MiB used, and both experiment leases are released. No driver reload.

## Concrete size defect

The C layout declares 1342701584 bytes per bank, but the actual object BTF
reports `STRUCT frdb_value size=268959760`, as retained in
`btf-value-size.txt`. The runtime allocates that smaller value and the
collector reports only 2151678080 total drain bytes across eight banks.
Thus the size mismatch already exists in compiler output; this is not
evidence that the runtime correctly allocated the requested ten-GiB arena.
The difference is consistent with truncating a size-in-bits field before
conversion to bytes. Its relationship to the CUDA abort is not isolated.
Existing verifier warnings are retained; they did not reject this run.

Local Qwen has the repair: use 32 banks of 16384 slots, keeping all 524288
thread slots, 256 records per slot and 80 bytes per record. Each bank would
then be 335675408 bytes, below 512 MiB. This changes banking, not logical
capacity or event sampling. Rebuild that source and retry the real prefill
after the repair; do not report a throughput number from this failed run.

The same local-model edit is arranging record-major storage and matching
collector indexing. Neither that layout nor the proposed banking repair
has been measured in this directory. All initial source/build records
remain, including the adverse run.
