# Physical GPU-chunk reclaim: disk/UVM serving comparison

The local Qwen driver implementation is committed and pushed as 95097e20
on gpu_ext-kernel-modules branch revision/gpu-storage-decision-575, based
on dea1fefc. The complete driver build exits zero at 06:46:56 PDT on
6.15.11-061511-generic. Root corrected only a misleading tracker-wait comment;
the implementation uses the existing MMU unmap and PMM free tracker path.

After disk-offload group processing, the worker takes the block lock and
checks that the block is alive. It releases allocated physical GPU chunks
only when their whole span has no owning-GPU residency or CPU/GPU PTEs.
Any UVM-Lite participation disables this release; temporarily pinned chunks
are retained. The helper does not change policy decisions, residency masks,
or disk transport. This repairs a source-level retention path; elimination
of serving OOM remains a hypothesis until the run completes.

run-serving.sh reuses the preceding EFVUGf experiment without changing the
workload, output length, concurrency, arm rotation, five-block count,
62502 ns/token calibration, adapter, or CUDA restore helper. The three arms
remain stock UVM, native reclaim policy and BPF reclaim policy. All previous
results, including partial outputs and OOM failures, remain preserved.

Both shared GPU/struct-ops leases cover execution. The original loader PIDs
583417/583418 were re-read immediately before this run; they are historical
targets, not reusable values for a later invocation. The shell restores the
saved original UVM module and original-policy loaders on exit. No model or
large generated binary is committed, and no manuscript is edited.

build-initial.log records the earlier helper-only build, before the worker
call and final mapping checks. build-driver.log records the complete build;
build-driver.sh contains the command. The three source files are available
in driver commit 95097e20; local source snapshots and module binaries are
not a separate publication requirement. lifecycle.log and runner.log retain
the current run. Runtime completion and results are pending.

## First completed block (partial campaign)

All three block-0 cells finish with 8192 completed/observed output tokens,
zero HTTP failures and server exit zero. The remaining four blocks continue;
these rows are not a completed five-block comparison.

| Arm | Warm token/s | UVM restores | Restore fallback/error |
| --- | ---: | ---: | --- |
| stock | 71.698241 | 72 | 0 / 0 |
| native | 69.291949 | 40 | 0 / 0 |
| BPF | 66.487342 | unavailable | unavailable |

BPF is slower in this first block; no improvement or tight overhead bound
is claimed. Its disk-UVM diagnostic file is zero bytes, and the final policy
diagnostic file was not produced before shutdown. The raw performance result
is retained without turning this missing diagnostic into another run gate;
unavailable counters must not be presented as zero. Stock/native have complete
disk-UVM counters, including 48 prepared ranges each. Earlier OOM results are
not overwritten, and the later blocks will determine whether completion is
stable across the campaign.

Block 1 also completes all three cells with 8192 tokens and zero HTTP
failures each: native 68.387745, BPF 68.733332, stock 70.572913 token/s.
Native/BPF each record 40 UVM restores, zero fallback and zero restore
errors. Stock's diagnostic counters are unavailable in this block. Thus
six of fifteen planned cells are complete; the campaign continues and
missing counters are retained as missing rather than forcing a repeat.

Block 2 also completes all three cells with 8192 completed/observed tokens
and zero HTTP failures each: BPF 65.422796, stock 73.961755, native
67.656516 token/s. Stock/native record 72/40 UVM restores and zero
fallback/errors; BPF's diagnostic counters are unavailable. Nine of
fifteen planned cells are now preserved; the remaining blocks continue.
