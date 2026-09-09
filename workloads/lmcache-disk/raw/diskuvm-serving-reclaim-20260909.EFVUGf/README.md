# Disk/UVM serving with per-range reclaim: start record

Final status: five observations saved, including two EngineCore OOM failures.
The deferred stop and original-module restoration completed at 05:11:29 PDT.
See [RESULTS.md](RESULTS.md). The start record below is retained as history;
this run does not establish a successful five-block performance comparison.

This is a fresh five-block comparison using the same workload, output length,
concurrency, arm order and existing 62502 ns/token calibration as aodtQv.
The former 13 observations, including five EngineCore OOM failures, remain
published; no incomplete output is presented as a paired performance win.

The local Qwen change serializes restore plus reclaim for each backing range.
After GPU fault restoration and the existing D2D copy, it calls the current
driver's OFFLOAD and waits for that range's existing completion information.
Already durable spans skip redundant file writes in the driver. This targets
accumulated GPU residency without adding a whole-device synchronization.
The managed allocation, disk registration and disk data remain reusable.

The first restore exception and the first per-backing cleanup exception are
logged and retained in the existing diagnostics. Cleanup failure is visible,
not grounds for suppressing its timing; successful D2D restoration alone does
not prove successful reclamation. Actual EngineCore/request failures will be
reported as such. Fixing the observed OOM is a hypothesis until this run.
The temporary CPU-prefetch/global-synchronize implementation was not run and
was removed before this source snapshot.

source/ contains the adapter and unchanged CUDA helper sources. The helper
library remains /tmp/lmcache-diskuvm-serving-build-20260909.7dZdC8/libdiskuvm_fault.so.
The same existing disk/UVM driver and native/BPF policies are used. No new
CUDA/driver build, workload, clock test, or extra correctness campaign is added.

run-serving.sh preserves the exact invocation and restores the saved original
UVM module and GDS/KV loaders on exit. Both shared leases cover the entire run:

flock /tmp/gpubpf-revision-gpu0.lock flock /tmp/gpubpf-revision-struct-ops.lock sudo -n bash /home/yunwei37/workspace/gpu/gpu_ext/workloads/lmcache-disk/raw/diskuvm-serving-reclaim-20260909.EFVUGf/run-serving.sh

The original loader PIDs were re-read as 514572/514573 immediately before this
invocation; those values are historical targets and must be re-resolved for
any later run. This record does not claim that all cells have finished.
No manuscript is edited and no old number or failure record is deleted.
