# GPU-local full records: five completed three-arm blocks

All 15 pp512/tg0 measurements complete on RTX 5090 / NVIDIA 575.57.08,
with the existing TinyLlama 1.1B Q4_K_M workload. GPU-local buffering
substantially improves throughput over the host-mapped full-record ring,
but still has substantial overhead relative to the uninstrumented baseline.

| Metric | Baseline | Original full-record ring, transport 3 | GPU-local full records |
| --- | ---: | ---: | ---: |
| Median prefill token/s | 38240.043809 | 345.221100 | 23267.228804 |
| Median paired throughput loss vs baseline | — | 99.098649% | 39.099166% |
| Completed measurements | 5 | 5 | 5 |

The median paired GPU-local/ring throughput ratio is **67.472424x**,
range 66.940838–68.628615x. GPU-local baseline-relative loss ranges from
38.992478% to 39.394399%. Paired-block bootstrap median 95% intervals
(10000 resamples, seed 1797) are 66.940838–68.628615x and
38.992478–39.394399%, respectively. Five blocks establish the observed
repeatability here, not generality across applications or hardware.

## Same logical records, different storage implementation

Both instrumented arms retain 23068672 records per run, each containing
the original ten u64 fields: block xyz, thread xyz, block dimensions xyz,
and per-thread timestamp. Logical capacity remains 524288 thread slots
with 256 records per slot. No leader-only sampling or field removal is used.

The ring arm uses the existing per-thread BPF object with transport 3 in
bpftime `886b4ca`. The GPU-local arm uses the new BPF array writer/collector
at main `4dd36b63`: 32 banks of 16384 slots, record-major physical indexing,
and the existing GPU_ARRAY map/CUDA IPC path. Automatic warp execution is
off for this arm; the ring arm retains its prior on setting and per-thread
execution semantics. This is a **BPF storage implementation change**, not
a same-object compiler automatic-warp result, and it does not isolate each
individual contribution of placement, layout, or publication protocol.

The GPU-local path is bounded capture followed by collection, not an
unbounded concurrently drained stream. Each collector copies 10741613056
bytes after the client finishes. Median copy time is **1490.922220 ms**
(range 1457.517662–1515.503690 ms), outside the prefill token/s measurement.
Allocation, instrumentation/bootstrap and collection are not free; the raw
outer elapsed times retain those lifecycle costs. The prefill ratio is not
an end-to-end application speedup.

All 15 clients and 10 collectors exit zero. Both probes observe all records;
the GPU-local runs have zero overflow/out-of-range counts, and the ring runs
have zero drops/pending records. All private shared-memory segments are
removed. GPU is idle at 1 MiB and both experiment leases are released.
Existing warning-mode verifier output is retained; no strict-admission or
new safety guarantee follows from these performance measurements.

## Ordering, evidence and retained history

The successful first GPU-local run was retained as block 1 rather than
repeated. Its baseline/ring controls ran afterward; its first position and
the gap before the controls arose from bring-up, not randomization. The
remaining blocks rotate the starting arm. `paired.py` therefore executes
14 new cells, filling five three-arm blocks with that retained observation.
No completed transport2/3 campaign or old Table 1 batch is replayed.

Raw commands, per-process logs, all rows and per-block analysis are in
[full-record-device-buffer-32bank-20260908.gaSN8I](raw/full-record-device-buffer-32bank-20260908.gaSN8I/README.md).
`paired_cells.json` is the collected measurement table; `analysis.json`
retains per-block ratios, summary statistics and resampling details.

The initial eight-bank BTF-size truncation and failed CUDA invocation remain
in their [failure record](raw/full-record-device-buffer-first-20260908.BytFAr/README.md).
The completed [transport2/3 comparison](results-transport23-20260908.md)
and all older compact-record RTX 5090/P40 Table 1 numbers remain unchanged.
Those older compact records differ from this full per-thread stream; do not
replace their numbers or multiply this speedup into their reported overhead.
No manuscript file was edited. The remaining roughly 39% overhead is still
an optimization target, not evidence that the P40 overhead has been matched.
