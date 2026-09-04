# Table 1 non-cross-clock subset

This is a predeclared, narrower campaign containing `kernelretsnoop` and
`threadhist`. It answers only the two Table 1 rows that do not depend on a
host-clock/`%globaltimer` comparison. It does not repair, relabel, or supersede
any retained `launchlate` failure, and it cannot satisfy the original
three-tool/seven-arm completion rule. Omitting `--tools` still selects all
three tools and preserves the original seven-arm campaign.

The selected matrix is derived once from
`--tools kernelretsnoop threadhist`: baseline plus the gpubpf and NVBit arm for
each selected tool. Correctness, timing, state/resume checks, summaries, and
independent analysis must contain exactly those five configurations. An extra
or missing configuration, schedule entry, correctness arm, timing arm, or
artifact is an error rather than an exclusion.

## Fixed dry-run matrices

These commands only print the planned matrix and exact gates; they do not read
GPU state, acquire GPU leases, build tools, or create a campaign directory.

```bash
python3 -B run_revision_rq4.py \
  --phase preflight --dry-run \
  --tools kernelretsnoop threadhist \
  --bpftime-root /home/yunwei37/workspace/gpu/bpftime-table1-575 \
  --bpftime-build-dir /home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575 \
  --gpu-thread-count 22528

python3 -B run_revision_rq4.py \
  --phase full --dry-run \
  --tools kernelretsnoop threadhist \
  --preflight-dir "$PWD/raw/passed-two-tool-preflight" \
  --output-dir "$PWD/raw/planned-two-tool-full" \
  --bpftime-root /home/yunwei37/workspace/gpu/bpftime-table1-575 \
  --bpftime-build-dir /home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575 \
  --gpu-thread-count 22528
```

Preflight is fixed at one pp=32 randomized block: five correctness cells and
five timing cells. Full is fixed at ten pp=512 randomized blocks: five
correctness cells and fifty timing cells. Both use seed 1797 and require every
selected configuration in every block. A subset full run additionally requires
`--preflight-dir` naming a distinct campaign for which the independent analyzer
reports `complete=true`, `phase=preflight`, and the exact same two-tool/five-arm
selection. The preflight and full paths must both be explicit, absolute after
normalization, and mutually non-nested in either direction. This admission is
checked before leases, builds, or GPU state are touched, recorded in the full
plan/state, and rechecked by every later independent full analysis. The default
three-tool full workflow retains its existing admission semantics and does not
require this new subset-only argument.

## Exact engagement gates

Every correctness cell must exit zero and produce the exact 47-byte normalized
stdout `Deterministic tests are essential\n> EOF by user`; each instrumented
cell must match baseline and pass its probe gate. Every timing cell must exit
zero and report the fixed prompt count plus finite positive throughput.

For `kernelretsnoop` correctness, gpubpf must report exactly 720,896 committed,
runtime-collected, host-collected, and nonzero-timestamp events; 220 launches;
22,528 unique Cartesian coordinates; 22,528 requested and allocated slots;
exactly 256 entries per slot; an 80-byte record; exact coordinate multiplicities
1,024 at 220, 1,024 at 44, 20,480 at 22, and zero other; zero segment mismatch;
and enabled/passed exact oracle fields. All drop, dirty, pending, second-drain,
and invalid-coordinate counters must be zero, and Cartesian/collector gates
must pass. NVBit must report exactly the same 720,896 nonzero-timestamp events
and 220 selected launches. Timed cells keep the same generic lossless and
Cartesian-complete gpubpf gate with the exact correctness oracle explicitly
disabled; every block additionally requires exact gpubpf/NVBit equality for
event and selected-launch counts. For timing, the frozen rope geometry is
`4 KV heads × pp tokens × 256 threads`: pp32 therefore requires exactly 32,768
coordinates, 44 launches, and 1,441,792 events, while pp512 requires exactly
524,288 coordinates, 44 launches, and 23,068,672 events. Every timing
coordinate must have multiplicity 44, with all other multiplicity bins zero.
Both gpubpf and NVBit
must independently report those exact event/launch totals before their pair is
compared. A count mismatch invalidates both cells.

The failed campaign `raw/preflight-575-noncross-clock-01` remains failed and is
never resumed or reclassified: its pp32 gpubpf timing cell allocated 22,528
slots but observed 32,768 coordinates on each of 44 launches, producing 450,560
OOB drops. The replacement uses a declared source patch to set the ring map
capacity before load. Correctness remains the original 22,528×256 exact-oracle
layout. Timing uses `pp×1,024` slots and exactly 16 entries per slot. At pp512,
the runtime's 24-byte slot header plus `16×88` aligned record bytes consumes
750,780,448 bytes including error counters (about 716 MiB), below the frozen
1,000 MiB segment budget. Dense 256-entry timing would exceed 10 GiB; even a
dense 44-record payload alone would exceed 1.7 GiB. The 16-entry setting does
not relax losslessness: any full/OOB/drop/pending/dirty/error counter or missed
event still rejects the cell.

For `threadhist`, gpubpf must report a positive total and nonzero-thread count,
exactly 1,048,576 configured and read-back entries, exactly 8,388,608 read-back
bytes, and `readback_complete=1`, including the zero-valued tail. NVBit must
report positive selected-launch, total-exit-probe, and nonzero-thread counts.
These gates apply unchanged to correctness and timing.

## Execution and independent analysis

After reviewing the dry-run output, remove `--dry-run`, add a fresh explicit
`--output-dir`, and keep the same `--tools` value for execution or resume. A
resume with a different selection, matrix, or fixed schedule is rejected.

The offline analyzer reruns the exact-output, engagement, timing, schedule, and
kernelret pair gates from recorded structured data instead of trusting the
runner summary. For every correctness and timing cell it also independently
requires both safety snapshots to retain driver 575.57.08, no compute apps,
zero UVM references, empty struct_ops maps/links and empty XID/dmesg/journal
abnormality lists, with the power-limit service active at 400 W. Telemetry must
contain at least one sample and report no throttling:

The analyzer also rechecks the frozen experiment parameters rather than only
the phase dimensions: tg=0, 99 GPU layers, 22,528 exit slots, 1,048,576
histogram entries, 1,000 MiB exit SHM, phase-specific exit slots and ring
entries, exact timing coordinates/44 launches/event count/shared allocation,
correctness/timing exact-oracle flags,
seed 1797, 10,000 bootstrap draws, driver 575.57.08, CUDA graphs disabled, UVM
and no-warmup disabled, worker CPUs 8–15, and telemetry CPU 16. Recorded input,
binary, source/build, NVBit, and uprobe paths must be absolute, and the target
symbol must equal the uprobe symbol hint. Dynamic launch-environment values and
the coordinator's affinity array are recorded but deliberately not frozen by
the offline analyzer.

```bash
python3 -B analyze_revision_rq4.py raw/<campaign-directory>
```

It returns success only when all five correctness cells and every required
five-cell block pass. Its output is scoped to the predeclared two-tool campaign;
it never converts a historical `launchlate` failure into an exclusion or a
successful result.
