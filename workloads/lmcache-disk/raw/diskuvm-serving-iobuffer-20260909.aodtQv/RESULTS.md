# Disk/UVM serving: completed observations and OOM failure

Run: 2026-09-09 03:36:41–04:05:47 PDT. Implementation: 8a352587 plus a26c57a5.

The runner saved 13 of 15 planned cells. After the actual EngineCore OOM
was identified, root requested the existing between-cell stop; the in-flight
cell finished and the runner exited 3. The saved original UVM module was
restored (RESTORATION_OK=1; restored loaders 514572/514573). Neither local
OpenCode nor an in-flight request was killed. The remaining two cells were
not run under this defective treatment.

| Block | Arm | Recorded token/s | Completed output tokens | Observed streamed tokens | HTTP failures |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | stock | 69.150960 | 8192 | 8192 | 0 |
| 0 | native | 62.769598 | 2048 | 2167 | 2 |
| 0 | bpf | 61.733256 | 2048 | 2167 | 2 |
| 1 | native | 67.300869 | 8192 | 8192 | 0 |
| 1 | bpf | 69.638864 | 8192 | 8192 | 0 |
| 1 | stock | 68.142541 | 8192 | 8192 | 0 |
| 2 | bpf | 64.312786 | 2048 | 2166 | 2 |
| 2 | stock | 73.152749 | 8192 | 8192 | 0 |
| 2 | native | 55.973392 | 5120 | 6040 | 0 |
| 3 | stock | 72.230213 | 8192 | 8192 | 0 |
| 3 | native | 64.737887 | 8192 | 8192 | 0 |
| 3 | bpf | 66.486788 | 8192 | 8192 | 0 |
| 4 | native | 57.459864 | 5120 | 6040 | 0 |

Every cell attempted eight requests of 1024 output tokens. Eight cells
completed all 8192 tokens. Five cells encountered EngineCore OOM and emitted
partial output; HTTP failures are zero in two of those cells because the
HTTP 200 streams had already started. Those partial rates remain above for
traceability, not as comparable throughput or a policy win. Blocks 1 and 3
are the only complete three-arm blocks; this is not the planned five-block
comparison and no aggregate superiority claim is made.

The first-block README originally asserted all requests completed. Commit
eb19df09 corrects that error. The actual exception is torch.OutOfMemoryError
in model embedding: allocating an additional 20 MiB fails with 8.38 MiB free
(e.g. block-00/position-1-native/server.log:853 and
block-04/position-0-native/server.log:1541). Server process exit zero is not
proof of completed serving.

The adapter successfully prepared retained disk-backed managed ranges and
served some restorations, but also fell back to the original GDS reader.
Restored ranges are retained alongside the destination KV storage; their
residency/lifetime is a concrete next repair target, not yet a proven sole
cause. The restore wrapper currently swallows the specific exception, so
first-error reporting is included in the pending implementation repair.
No new timing, clock, or correctness campaign is introduced.

Raw result.json, server.log, warm-progress.json and existing per-process
counter files accompany each cell; raw.jsonl and summary.json retain the
runner output. Generated KV cache files are not publication artifacts.
The old successful GDS campaigns and all failed bring-up records remain.
The disk/UVM transport is common to stock/native/BPF arms; stock disables
KV reclaim, native/BPF use the same reclaim algorithm. This remains a
CPU-staged disk restore, not NVMe-to-GPU P2P, and does not isolate transport
against untouched GDS. No manuscript was edited.

