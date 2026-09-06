# Stale-state formal performance campaign: interface and run readiness

Date: 2026-09-05. Scope: CPU/source implementation and CPU validation only.
No GPU experiment was run or started here; no module, service, or lease state
was changed. This note does not claim any performance result.

## Gap status after inspection

The three blockers named for this study (driver-owned timestamped snapshot,
matched native consumer, common decision diagnostics) are implemented in
[`driver-bridge-v1.patch`](driver-bridge-v1.patch), and the completed
owner-12 excluded preflight exercised all of them live:
`snapshot-publications.jsonl` records `publisher=shared_driver_snapshot`
with driver-captured `published_mono_ns`; the native cells consume the same
driver-captured snapshot through the in-driver native model; BPF and native
rows carry matched diagnostics (`policy-decisions.jsonl`,
`policy-final.json`). Revalidation confirms all
seven owner-12 cells pass the frozen validator today. What remained absent
was the end-to-end formal 21-cell execution path itself, which is now
implemented.

## Implemented interface (this change)

- `live_runner.py`: `execute-full --output … --preflight <absolute excluded
  preflight>` runs `validate_preflight` on the named preflight **before**
  leases, bridge checks, output creation, or any cell; writes the formal
  `campaign.json` (`stage=full`, 3 blocks, absolute `preflight` reference);
  executes the 21 frozen cells with the unchanged per-cell path; then
  returns `validate_campaign` (paired mechanism native-vs-BPF at each delay
  and information fresh-vs-delayed comparisons). `dry-run` plans either
  stage; `execute-preflight` rejects `--preflight`.
- `run_module_lifecycle.py`: `--preflight` argument; child command
  `execute-full … --preflight …` for `stale-state-575-full-*` outputs;
  final stage-appropriate validation (`validate_campaign` for full,
  `validate_preflight` for preflight).
- `run_study.py`: new `validate-preflight --input` revalidates one excluded
  preflight root without any event or record execution.

## Excluded preflight state

`raw/stale-state-575-preflight-20260905-owner-12` completed all seven cells
and passes `protocol.validate_preflight` (7/7 valid). Its outer lifecycle
record (`raw/stale-state-575-lifecycle-20260905-owner-12`) ends at
`candidate_loaded` without recovery events, and the host later rebooted
(cause unrecorded); a fresh-boot observation found stock 575 UVM, an idle
GPU, active services, and no stale-state ABI. The preflight records, not
that lifecycle record, are what the formal gate revalidates.

## Executed CPU commands

```text
sudo -S python3 -B run_study.py validate-preflight \
  --input "$PWD/raw/stale-state-575-preflight-20260905-owner-12"
# -> run_status "valid", 7/7 cells valid (real owner-12 records, root-readable)

taskset -c 18 make test-offline          # 66/66 tests OK (was 60; +6 new)
python3 -B live_runner.py dry-run \
  --output "$PWD/raw/stale-state-575-full-01" \
  --preflight "$PWD/raw/stale-state-575-preflight-20260905-owner-12" \
  --inherited-lease-fds 11 12
# -> stage "full", 21 planned cells, no GPU/module action declared
python3 -B run_study.py dry-run full \
  --output "$PWD/raw/stale-state-575-full-01" \
  --preflight "$PWD/raw/stale-state-575-preflight-20260905-owner-12"
```

CPU test identities (build identities, not integrity evidence): clang/llvm
18.1.3 BPF path, g++ 13.3.0 host wrapper, bpftime uBPF JIT at the in-repo
default root; Python tests ran under the system CPython on CPU 18.

## Formal execution command (authorized window only; not run here)

Run from `workloads/stale-state-575` as a single authorized lifecycle
invocation (it loads the matched candidate UVM, passes both lease FDs to the
child, restores the admitted stock module in `finally`, and validates the
21-cell campaign before returning):

```text
sudo -S python3 -B run_module_lifecycle.py execute \
  --candidate /opt/gpubpf/modules/575.57.08/stale-state-v1-stage-20260905-owner-preflight-12/nvidia-uvm.ko \
  --restore  /opt/gpubpf/modules/575.57.08/stock-dkms-6.15.11-owner12/nvidia-uvm.ko \
  --stage    /opt/gpubpf/modules/575.57.08/stale-state-v1-stage-full-01 \
  --output   "$PWD/raw/stale-state-575-full-01" \
  --record   "$PWD/raw/stale-state-575-lifecycle-full-01" \
  --preflight "$PWD/raw/stale-state-575-preflight-20260905-owner-12"
```

If a review rejects owner-12 as the admitted excluded preflight, run a fresh
seven-cell preflight the same way and substitute its absolute path. The
accepted analysis command is:

```text
sudo -S python3 -B run_study.py analyze --input "$PWD/raw/stale-state-575-full-01"
```

## Measurement fields (recorded by every cell; aggregates by analyze)

Per cell (`workload-result.json`, `uvm-events.jsonl`, policy rows):
`end_to_end_ms`, `total_kernel_ms`, `checked_values`, `mismatches`,
`verified_words_per_second`, phase wall/kernel times;
`gpu_faults`, `migrations`, `migrated_bytes`, `prefetch_migrations`,
`prefetch_bytes`, `thrashing_events`, `eviction_events`, `dropped_*`
counts, and derived `gpu_faults_per_second` / `migrated_bytes_per_second`;
policy `decision_age_ns` distribution, `wrong_phase_decisions` /
`wrong_phase_fraction`, action and effect counters, and observer/driver
counter closure. Per campaign (`run_study.py analyze`): paired block rows
under `mechanism_cost` (native vs BPF at each delay) and `information_cost`
(fresh vs delayed), each with `end_to_end_ratio`,
`verified_throughput_ratio`, `gpu_fault_rate_ratio`,
`migration_rate_ratio`, `wrong_phase_fraction_delta`,
`observed_degradation`, and `negative_result`; per-campaign
`negative_results_retained`. Decision fields are recomputed by the
analyzer from raw records and are never taken as self-declared values.

No formal cell has run. Preflight medians (one block, descriptive only,
excluded from any paper claim) were: fresh ~55–59k faults/s with ~270k
verified words/s; 100 ms ~72k faults/s, ~267–270k words/s; 1 s ~162–165k
faults/s, ~210k words/s — the 1 s delay inflates faults and reduces
verified throughput, and native vs BPF stayed within ~2% at matched delay.
These are not formal results; the 21-cell matrix above is the runnable
performance experiment.
