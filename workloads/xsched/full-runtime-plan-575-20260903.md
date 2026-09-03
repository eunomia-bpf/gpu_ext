# Full XSched four-arm execution, after persistent-timeslice repair

This is the frozen protocol for the **completed full campaign**. All ten
four-arm blocks and six isolated controls passed the independent raw audit;
see [full results](performance-full-575-20260903.md). The completed
five-block/five-kernel pilots remain separate. Driver `849ea75d`
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

These are the retained execution instructions. The completed run used
`FROZEN_REPS=9511106` from its new calibration and ran uninterrupted, without
`--stop-after-blocks`. Future repeats must use new output paths and their own
fresh calibration, not overwrite the retained records.

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

## Post-reboot admission and preflight completed

The restored `849ea75d` driver passed both fresh strict real-GSP canaries under
`raw/gpreempt-context-{original,bpf}-849ea75d-postboot-20260903-0147/`. Each
checks 2,048 integer outputs and 17 negative cases, with the last completed
LC/BE GSP values still 1,000,000/1 us before execution. Cleanup passed.

The new isolated calibration is retained in
[`raw/calibration-persistent-575-20260903/calibration.json`](raw/calibration-persistent-575-20260903/calibration.json):
**`frozen_reps = 9511106`**, producing a 79.968544 ms kernel. Use that value in
the full commands above; do not reuse the old pilot calibration.

All four arms passed the new
[`raw/preflight-persistent-575-20260903/`](raw/preflight-persistent-575-20260903/)
preflight with those repetitions. Each cell has 48 kernels and 4,177,920
validated output values. Both XSched frontends observed suspend/resume on all
16 BE queues; the BPF frontend executed 105 JIT calls and 1,489 queue decisions.
The driver BPF arm observed 20 initialization modifications, six persistent
control overrides, four BE targets and eight successful preemptions, with
zero setter/preemption errors. The independent raw audit also passed all four
cells, including worker commands, sample clock conversion, numerical counts,
engagement and pre/post safety.

These records remain a **two-kernel-per-stream preflight**, not the full
comparison. Their 16 LC samples make p99 the sample maximum; the subsequently
completed four-arm ten-block campaign and all six full isolated controls were
not replaced by these short cells. In particular, the short driver-BPF cell
reduced LC delay but also reduced BE throughput, so no full-performance win is
inferred. The GPU was then released for the separately recorded GPreempt
five-block campaign and the full MoE campaign. XSched full then completed all
forty mixed cells and six isolated controls with no restart or omitted block.
Its [independent raw audit](raw/full-persistent-575-20260903/independent-raw-audit.json)
and [full result](performance-full-575-20260903.md) supersede the former pending
status; historical checkpoint files still describe only their earlier stage.
