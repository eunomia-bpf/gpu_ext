# Full XSched four-arm execution, after persistent-timeslice repair

This is the ready execution protocol, **not a full-performance result**. The
completed five-block/five-kernel pilots remain separate. Driver `849ea75d`
has passed original and BPF two-context real-GSP canaries: final LC/BE values
are preserved through CUDA's default control; unmarked native contexts remain
unchanged. That canary does not measure XSched contention performance.

## Frozen workload and scope

All four arms use one RTX 5090, NVIDIA 575.57.08 on Linux 6.15.11, 400 W,
2 LC + 4 BE processes, four streams each, **50 kernels per stream**, 340 blocks
and 256 threads per kernel. A fresh isolated calibration freezes repetitions
for an 80 ± 4 ms kernel. Seed 1797 fixes all ten four-arm block orders.
The campaign also requires three isolated LC and three isolated BE controls.

Each mixed cell has 400 LC and 800 BE samples and validates 104,448,000 output
values. All 40 mixed cells plus six isolated controls total 49,200 kernels and
4,282,368,000 values. LC nearest-rank P99 is no longer the pilot's sample max.

| Arm | Policy and execution boundary |
| --- | --- |
| `native` | Native CUDA streams on the same custom driver, with no attached scheduling policy. |
| `xsched` | Original upstream HPF, original CUDA frontend and Level-1 actuator, 24 XQueues, 50 ms server period. |
| `bpftime_hpf` | Actual ubpf-JIT HPF decisions, same original XSched frontend/Level-1 actuator; bounded 64-queue port, not driver-only BPF or pure VM-overhead isolation. |
| `gpubpf` | Process timeslices LC 1,000,000 / BE 200 us through init plus validated persistent control; LC-launch-triggered GR-target preemption with the original 100 us per-CPU cooldown. |

The driver arm retains its all-engine process timeslice semantics; preemption
targets are GR RM `(hClient,hTsg)` handles, not numeric grpIDs. Captures now
require valid envelope sizes, successful reads, successful syscall and RM
status, and actual GR engines 1..8. The new runtime gate requires nonzero
persistent control engagement and no setter errors. Old init-only negative
results are preserved, not merged with this new-driver campaign.

This is an XSched **Level-1** workload reproduction, not the paper's Level-3
mechanism on sm_120. Neither extra no-cooldown/interleave arms nor a stock-driver
baseline are added to these four frozen arms. The primary results remain LC
submission-to-device-entry P99 and BE kernels per second from common release
to last BE completion, with paired block intervals and the existing 5% BE
noninferiority margin. Every LC release follows confirmed BE activity plus
at least 5 ms. Failed/partial cells are retained and invalidate their campaign.

## Commands after the coordinator grants the GPU slot

No rebuild is required. Replace `FROZEN_REPS` below with the newly generated
`calibration.json` value, not the old pilot's repetitions.

```sh
python3 -B workloads/xsched/run_xsched_rq3.py calibrate --configs native,xsched,bpftime_hpf,gpubpf --output workloads/xsched/raw/calibration-persistent-575-20260903 --timeout 120
python3 -B workloads/xsched/run_xsched_rq3.py preflight --configs native,xsched,bpftime_hpf,gpubpf --reps FROZEN_REPS --output workloads/xsched/raw/preflight-persistent-575-20260903 --timeout 180
python3 -B workloads/xsched/run_xsched_rq3.py full --configs native,xsched,bpftime_hpf,gpubpf --reps FROZEN_REPS --output workloads/xsched/raw/full-persistent-575-20260903 --timeout 900 --stop-after-blocks 1
```

The last command runs all six controls, then one complete four-arm block and
releases both leases. The campaign still requires all ten blocks. Resume on
the next assigned slot with the same frozen arguments:

```sh
python3 -B workloads/xsched/run_xsched_rq3.py full --resume --configs native,xsched,bpftime_hpf,gpubpf --reps FROZEN_REPS --output workloads/xsched/raw/full-persistent-575-20260903 --timeout 900 --stop-after-blocks 1
```

Omit `--stop-after-blocks` to finish all remaining blocks without a planned
pause. A pause is permitted only after all four cells of a block complete
successfully and pass raw audit. Resume rejects changed structured protocol,
schedule, kernel/driver identity and ordinary source/binary/BTF metadata. It
rechecks every saved worker executable, taskset affinity, frozen runtime
environment, full workload argv, raw sample ordering/clock conversion,
numerical validation count, recomputed metrics, policy/XQueue engagement and
pre/post safety for all six controls and all previous complete blocks. It
does not trust a saved `passed` flag or skip a failed/incomplete block.
Isolated LC/BE identities are bound to their expected control directory, not
inferred from the saved result's label. Environment records include only
runtime settings (CUDA/CUPTI/XSched, library paths and thread counts), not the
entire inherited environment or unrelated credentials.

Expected runtime, extrapolated from previous short cells rather than measured
on this new protocol: calibration 5–15 seconds, four-arm preflight roughly
30–60 seconds, first control-plus-block slot roughly 7–9 minutes, later paired
blocks roughly 5–7 minutes each, and the entire campaign roughly 55–70 minutes.
Natural pauses and their fresh admissions are recorded; they do not change
the predetermined order, kernel count, sample count or selected results.
