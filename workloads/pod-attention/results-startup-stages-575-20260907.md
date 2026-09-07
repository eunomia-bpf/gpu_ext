# POD startup stages on the optimized host runtime

One instrumented fresh process completed with exit zero. Of its 229.353175 s
before Python main, PTX rewriting accounts for 88.959518 s and compilation
for 139.520027 s. Module loading takes 0.091513 s. This localizes the aggregate
startup cost; it does not measure a new optimization or establish a general
runtime overhead bound.

| Region, six modules | Elapsed time, s | Fraction of pre-Python startup |
| --- | ---: | ---: |
| `hack_fatbin` | 88.959518 | 38.787% |
| `compile_ptxs` | 139.520027 | 60.832% |
| Module load loop | 0.091513 | 0.040% |
| Outside these three regions | 0.782117 | 0.341% |

`compile_ptxs` already uses a thread pool. Its measurement covers the entire
function, not just time inside the compiler library; `hack_fatbin` likewise
does not isolate individual rewrite passes. The load loop includes its normal
module bookkeeping and trampoline initialization. These measurements do not
justify adding parallelism blindly or blaming module loading for the cold path.
The next startup optimization must reduce or reuse rewriting/compilation work,
then measure its own performance. None is claimed implemented by this patch.

## Raw observations and retained failed diagnostic setup

| Attempt | Timing switch in target | Pre-Python, s | Client wall, s | Operator mean, ms |
| --- | --- | ---: | ---: | ---: |
| `01` | absent | 225.821674 | 230.155641 | 3.322909 |
| `02` | `1` | 229.353175 | 233.763109 | 3.360422 |

Both benchmark processes finish with code zero, using the existing
`bench.py --phase-study`, Llama-3-8B / decode batch 32, ten warmups and 100
CUDA-event operator samples. Attempt `01` is not stage-timing evidence: root
set the opt-in variable outside the launcher, but the launcher's isolated
environment discarded it. The process was allowed to finish and its raw
outputs remain. Main commit `d3f5e93d` forwards that one optional variable
to BPF clients; `02` records it in `execution.json` and emits all three stage
lines. No other environment inheritance was enabled.

Do not pool these sequential observations with the earlier three-cell Release
campaign or call their difference instrumentation overhead. There is no paired
native control or repeated instrumented series here. The larger operator time
in `02` is retained, not omitted. The existing benchmark's own numerical output
is preserved; no extra correctness campaign, preflight, clock calibration or
artificial execution timeout was added.

## Implementation and build

Local OpenCode GLM generated the 41-line opt-in C++ change. Root applied and
built it in bpftime commit `eef8a51` on `revision/table1-host-plt-fix`, then
published the [patch](setup-steady-followup/startup-stage-timing.patch) in main
`5cd22d95`. It uses host steady-clock intervals and emits three stderr lines
only when `BPFTIME_CUDA_STARTUP_TIMING` is nonempty and not `0`. Compilation
options, selector logic, six PTX packets and load order are unchanged.

Root initially trimmed a trailing context blank while copying the patch,
causing a malformed final hunk count. The artifact's final count was corrected;
`git apply --reverse --check` then passed against the applied source. This
packaging repair did not change the generated C++ implementation.

Build command, from `/home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt`:

```sh
cmake --build build-pod-release-575 --target bpftime-agent -j8
```

The build exits zero with the existing `ebpf_inst` ODR warning. Before rebuilding,
root retained the original Release agent (175374144 bytes), syscall server
(172865000 bytes) and PTX entry pass (111733560 bytes), preserving their normal
relative paths under `build-pod-release-575-untimed-saved`. This is a local
runtime backup, not a claim of an independently relocatable distribution;
compiled absolute dependency paths still refer to the retained source/build
tree. The earlier Debug build was not rebuilt or removed.

## Reproduction and artifacts

From the gpu_ext repository, the instrumented command is:

```sh
env BPFTIME_CUDA_STARTUP_TIMING=1 python3 -B \
  workloads/pod-attention/run_current_runtime_cell.py \
  --output workloads/pod-attention/raw/startup-stage-575-20260907-02/block-1-pod_bpf \
  --arm pod_bpf --block 1 \
  --bpftime-build /home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt/build-pod-release-575
```

That output directory is complete and must not be overwritten or rerun.
[Attempt 01](raw/startup-stage-575-20260907-01/) and
[attempt 02](raw/startup-stage-575-20260907-02/) retain execution environments,
commands, loader/client logs and the original benchmark output. The three
stage values are directly readable from `02/block-1-pod_bpf/client.log`;
pre-Python, client-wall and operator values are in `execution.json`.

This coordinating session did not edit paper files. LMCache async prefetch is
separate implementation work, not a completed outcome of this diagnostic.
