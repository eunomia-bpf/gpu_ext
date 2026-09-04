# OpenCode/Qwen read-only review

Date: 2026-09-04  
Model: `spark-gateway/qwen3.8-27b-nvfp4-200k`  
Configuration: snapshots and sharing disabled; every OpenCode tool denied  
Sessions: `ses_f9321f641ffecN3VxztYfXohE3`, `ses_f93154cf0ffeiPvVGvZGyUTiT5`,
`ses_f9307369cffeCHFlYwoXAqyWqe`

The reviews were CPU-only and read-only. They did not execute a GPU workload or
modify the repository.

## Findings applied before attempt 02

The first S0 attempt exposed a real parser defect rather than a workload
failure. `llama-bench` prints `avg_ts` with six decimal places but emits each
`samples_ts` element through a default C++ stream with six significant digits.
Consequently, `38135.254576` and its printed sample `38135.3` are consistent;
the old relative tolerance rejected them. Qwen confirmed that accepting at
most half of the six-significant-digit print unit, while retaining the separate
`pp * 1e9 / avg_ns` check, does not weaken the arithmetic evidence. The failed
attempt remains invalid and was not resumed or relabelled. Attempt 02 started
from a new directory after the repair and its regression tests passed.

## Corrected review findings

The initial review treated warmup and resume handling as blockers. The second
review inspected the underlying sources and corrected both conclusions:

- `llama-bench` defaults `no_warmup` to false, changes it only for an explicit
  `--no-warmup`, executes warmup before measured repetitions, and the frozen S0
  command contains no such opt-out flag.
- A resumed campaign lacks only a redundant run-level idle snapshot. Every new
  cell still executes the same driver, idle, telemetry, kernel-log, UVM,
  `struct_ops`, and cleanup checks as a fresh campaign.
- The resume `is_file` gate correctly names each probe executable rather than
  its containing directory.

Therefore none of these observations requires attempt 02 to stop or rerun.

## Completed final replay hardening

The independent analyzer was strengthened without changing the raw data,
throughput metric, or statistical plan. It now:

1. parse the final numeric `# exit:` footer and require it, the execution
   record, and the expected return code all to be zero;
2. independently rederive the fixed-driver, 400 W, idle GPU, clean kernel-log,
   UVM, `struct_ops`, boot-continuity, and cleanup gates from each raw safety
   record;
3. reparse every telemetry CSV and reproduce its sample count, peaks, mean
   power, clock range, and throttle decision; and
4. report one consistent `llama-bench` build identity across all cells.

The final deny-all Qwen review returned `PASS` with no blocker. All 18 CPU-only
test methods passed, including 37 new negative mutation subcases. The hardened
analyzer also accepted every completed attempt-02 cell available during the
review while correctly retaining the incomplete campaign status. These are
fail-closed offline checks and do not require another GPU execution.
The final claim remains limited to the ten-block paired pp512 throughput
effects for `kernelretsnoop` and `threadhist` on this RTX 5090/driver/workload
configuration. It is not an equivalence result, device-versus-host cost
decomposition, or a general claim over other tools or GPUs.
