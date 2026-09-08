# Record-preserving transport 2 versus 3

Five fresh rotating pairs on bpftime `886b4ca`, using the existing Table 1
`run_arm_cell` path and original per-thread kernelretsnoop tool. No baseline,
NVBit or completed historical batch is replayed. Both arms retain 80-byte
payloads, 256 entries per thread and 524288 allocated thread slots. The
runtime changes only the header/payload memory layout and host reassembly.

`execute.py` records the experiment invocation; the internal helper label
`auto_warp_on` applies in BOTH arms and is not a leader-only execution claim.
The outer `transport2` / `transport3` labels identify the compared layouts.
Each row's `cell_root` locates its helper-relative logs. Timing is the
existing llama.cpp pp512/tg0 token/s metric. Diagnostic call counting is off.

Private loaders receive SIGINT after their CUDA client exits and finish
normally, without a short kill deadline; process ownership and private SHM
cleanup remain in the existing helper. Numeric throughput and collector
status are recorded separately. No clock, oracle or logging gate is added.

Status: complete, five pairs / ten measurements. All benchmark and loader
processes exit zero; each private shared-memory segment is removed normally.
Transport-2/3 throughput medians are 208.756714/346.808416 token/s. The
median paired ratio is 1.650416 (bootstrap 95% interval 1.632708–1.680261).
Every collector reports 23068672 events, zero drop counters and zero pending
events. The disabled multiplicity oracle is not claimed as a passed check.
`analyze_pairs.py` adapts the existing paired-median bootstrap calculation
to the two transport labels (10000 resamples, seed1797); it never runs cells.
