# MoE-Infinity four-cell continuation on 575

Protocol: `proposal-3-revision-8-575`, recorded before any 575 performance result.

The user requested automatic completion of the actual MoE-Infinity, XSched,
and LMCache comparisons, explicitly including gpubpf. This continuation uses
Linux 6.15.11 and NVIDIA 575.57.08 for every cell; no 610 correctness or timing
sample is pooled. It reuses the existing request, exact-output, CPU-affinity,
policy-ownership, telemetry, and paired-analysis implementation rather than
creating an unrelated benchmark.

Unchanged: all four configurations, exact 120B MXFP4 source models, CPU 0–7,
512 input and 64 output tokens, eight measured prompts, one excluded warm-up,
new server/policy process per cell, frozen randomized configuration/prompt
orders, 60-second idle cooldown, five valid complete blocks with at most eight
attempts, and the paired geometric-mean/10,000-resample analysis. The MoE
artifact includes the already disclosed row-chunking and deterministic
accumulation correctness repairs. gpubpf is the existing safe per-CPU,
1/256-sampled host-stride/LFU ablation, not the full device-observed policy.

Execution corrections reflect facts established before this continuation:

- Each of the four cells is revalidated on 575 with two complete eight-prompt
  correctness passes. Exact equality is required within each configuration;
  cross-configuration equality remains diagnostic because the previous
  single-slot controls established differences between valid greedy outputs.
- MoE's seven-partition store is buffered-hydrated once into CPU memory by
  each new server. Expert dispatch/cache counters must engage; a measured
  positive disk-read delta is not required and no steady-state direct-I/O
  claim is made. A new 61 GiB derived disk store is created during preflight
  and reused by later fresh processes, so duplicate immutable model copies
  do not exhaust the disk. Model loading and hydration remain excluded from
  measured request throughput. No old evidence or user file is deleted.
- gpubpf requires positive page-fault, stride, prefetch, activation, access,
  sampled update/reorder, and eviction-prepare callback deltas. The earlier
  UVM Tools descriptor did not deliver type-14 events even during real
  oversubscription. This continuation therefore does not assert completed
  evictions from that monitor or rename prepare callbacks as completions.
- All cells keep the same enforced 400 W safety limit. Software power-cap
  activity is recorded rather than discarding the intended capped condition;
  thermal slowdown and hardware slowdown/brake conditions still invalidate
  the cell. Kernel/Xid, idle-GPU, empty struct_ops, and ownership-safe cleanup
  checks run before and after every cell. A failure stops the run instead of
  blindly spending replacement attempts on the same error.

Commands from this directory:

```bash
.venv/bin/python -m unittest -q test_offline.py test_575_head_to_head.py
.venv/bin/python run_575_head_to_head.py admit
.venv/bin/python run_575_head_to_head.py preflight \
  --output raw/head-to-head-575/preflight
.venv/bin/python run_575_head_to_head.py run \
  --preflight raw/head-to-head-575/preflight \
  --output raw/head-to-head-575/timing --max-blocks 2
# Continue the same frozen sequence, without rerunning accepted blocks:
.venv/bin/python run_575_head_to_head.py run \
  --preflight raw/head-to-head-575/preflight \
  --output raw/head-to-head-575/timing --max-blocks 5
```

The two-block stage is a time-budgeted preliminary checkpoint, not full
reproduction and not a replacement for the five-block objective. Historical
request durations for planning are 6.5–13.6 s for UVM, 6.4–7.3 s for N-CMoE32,
about 5 s for repaired MoE, and one 45.026 s gpubpf warm-up. These are not
current performance measurements. Startup, 575 performance, and any failures
may prevent the preliminary checkpoint from finishing within one hour.
