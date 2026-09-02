# Driver-side priority candidates: measured outcome

Both requested driver-policy candidates were implemented and measured in a
new campaign without altering or dropping the [original negative result](performance-575-20260902.md).
Five randomized complete blocks of **five configurations / 25 valid cells**
finished with zero failed or excluded cells. Neither candidate reached
XSched's LC latency on this workload.

| Configuration | Median LC P99/max (ms) | Median BE kernels/s |
|---|---:|---:|
| Native CUDA | 7266.733 | 10.2071 |
| Original XSched Level-1 HPF | 4127.607 | 9.5735 |
| Original gpubpf timeslice/preempt policy | 7220.817 | 10.2161 |
| gpubpf, preemption cooldown disabled | 7230.447 | 10.2548 |
| gpubpf, LC high / BE low interleave | 7252.304 | 10.2365 |

The no-cooldown candidate's paired LC difference versus XSched has 95% CI
**+2889.377 to +3719.336 ms**; its relative BE throughput interval is
**+6.636% to +7.572%**. The interleave candidate's corresponding intervals
are **+2911.329 to +3752.342 ms** and **+6.266% to +7.968%**. These are
exploratory paired bootstrap intervals, not a multiple-comparison-adjusted
claim of discovering an optimal policy.

## Exactly what changed

The workload remained 2 LC + 4 BE processes, four streams each, five kernels
per stream, 340 blocks × 256 threads, and the already frozen 9,496,464 FMA
repetitions (approximately 80 ms isolated). All cells used the same custom
575.57.08 / Linux 6.15.11 / RTX 5090 stack, release protocol, CPU masks, and
400 W power limit. Seed 1797 shuffled the five configurations in each block.
The new XSched runner revision was `3acd688`.

- `gpubpf_nocooldown`: timeslices remained LC 1,000,000 us / BE 200 us. Only
  `--cooldown-us` changed from 100 to 0. Every measured cell recorded exactly
  **40 LC launches × 4 BE targets = 160 successful preemptions**, zero
  cooldown skips, and zero preemption errors. The preceding two-kernel
  preflight recorded exactly 64 successful preemptions.
- `gpubpf_interleave`: timeslices and the 100 us cooldown remained unchanged.
  Only runlist interleave requests were added: LC level 2, BE level 0.
  Every measured cell recorded **20 requests, 18 observed bind values,
  zero mismatches, and zero setter errors**. This proves requests and bind
  observations, not that a CPU-side shadow field alone proves a particular
  hardware scheduling effect.
- Native and original XSched behavior were not modified. The original BPF
  configuration remained a separate contemporaneous control.

All 25 cells checked every output: **261,120,000 values and 3,000 completed
kernels**. Each XSched cell engaged all 16 BE queues with suspend/resume.
All post-cell checks returned GPU utilization and UVM references to zero,
with empty struct-ops state and no Xid or kernel abnormality.

This remains the explicitly shortened five-kernel/five-block protocol:
40 LC samples per cell make nearest-rank P99 the maximum. It is not the
original 50-kernel/10-block full campaign or XSched Level-3 reproduction.

## Why the original policy only made eight preemption calls

The original saved loader logs show 120 total launch events: 80 BE launches
are filtered out, and **38 of 40 LC launches hit the 100 us cooldown**.
The remaining two events each preempt four BE TSGs, giving exactly eight
successful calls. In original block 0, each LC process submitted its 20
kernels in only 3.918 / 4.677 ms; the last device entries occurred about
7.210 / 7.242 seconds after the first submissions. No later launch events
arrived to trigger further preemption during that long queued execution.

The source-level distinction is larger than cooldown. The current kfunc
issues `NVA06C_CTRL_CMD_PREEMPT` with `bWait=true`; it does not maintain a
paused BE queue. The BPF `on_bind` callback unconditionally permits binding.
By contrast, upstream HPF checks the highest ready priority per device and
calls `AsyncXQueue::Suspend`, whose Level-1 action is
`launch_worker_->Pause()`. That prevents future BE batches from being
submitted until the higher-priority work is no longer ready. Increasing
instantaneous preemptions is therefore not equivalent to implementing HPF's
persistent queue-admission semantics. The completed candidates provide
experimental evidence that these two small parameter changes do not close
the LC gap; they do not show that HPF is inexpressible in BPF.

## Evidence

- [Frozen five-configuration protocol](raw/candidate-pilot-575-20260902/protocol.json)
- [All aggregate and paired results](raw/candidate-pilot-575-20260902/summary.json)
- [No-cooldown preflight](raw/candidate-preflight-575-20260902/gpubpf_nocooldown/result.json)
- [Interleave preflight](raw/candidate-preflight-575-20260902/gpubpf_interleave/result.json)
- All worker, policy, and safety JSON is retained beside the summaries.

Analysis-only replay:

```sh
python3 -B workloads/xsched/run_xsched_rq3.py analyze --output workloads/xsched/raw/candidate-pilot-575-20260902
```
