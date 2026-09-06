# Next storage-policy experiment: overlapping reads and writes

The completed end-to-end five-arm campaign measures immediate-submit policy
overhead. The policy-input ablation adds fixed pressure/slack/cost inputs, but
its cold-store-then-warm-read sequence does not establish benefit under
read/write contention. Demand reads in that path always submit immediately.

## Selected implementation

`../run_gds_mixed_backend.py` is being developed through local OpenCode/Qwen
27B. It drives the installed LMCache 0.5.4 GdsBackend and the committed
admission adapter, with real cuFile/O_DIRECT operations on 24 MiB objects.
Already-stored demand-read objects and fresh background-write objects are
distinct. Requests overlap and preserve their offered arrival times.

| Arm | Decision implementation | Inputs |
| --- | --- | --- |
| FIFO | Immediate submission | Same offered workload |
| Native | Existing native policy | Controlled pressure 801 permille; slack 10 ms |
| BPF | Same policy through UVM command 82 | Identical inputs to native |

Use bounded GPU buffers, fresh output directories, one first block followed
by five rotated blocks. Build/setup and population are outside the measured
steady interval; all steady-interval writes must complete before its end.

## Measurements and interpretation

Record offer, admission/submission and completion times, bytes, failures and
available decision counts. Report demand-read offer-to-completion p50/p99,
write completion throughput, total bandwidth and makespan. Compare native
against FIFO for policy behavior, and BPF against native for mechanism cost.
Deferral may improve urgent reads, worsen them, or reduce aggregate bandwidth;
the outcome is empirical.

The pressure value is a controlled policy input, not measured live HBM
pressure. Current adapter code delays individual writes; it does **not**
enforce the returned batch target or implement write coalescing/priority
ordering. The previous draft's coalescing claims were unsupported and are
not part of this experiment. There is no modeled busy-loop substitute for
real KV recomputation; recomputation remains outside this selected workload.
The transport remains cuFile compatibility until direct NVMe/GPU DMA is
actually demonstrated.

Prior executor decision medians (native 0.063 us, BPF 1.005 us) motivate
the comparison, but do not predetermine its application-level result.
Collect every attempted cell without correctness, clock, preflight or retry
gates, and retain the preceding campaigns separately.
