# POD optimized host-runtime follow-up

All three new Release BPF processes completed with exit zero. The original
Debug build and its [nine-cell campaign](results-current-runtime-575-20260907.md)
remain intact. No completed Debug cell was repeated.

| Release observation | Pre-Python startup, s | Client wall, s | Operator mean, ms |
| --- | ---: | ---: | ---: |
| 1 | 220.892286 | 225.158701 | 3.328349 |
| 2 | 223.995661 | 228.160801 | 3.318939 |
| 3 | 220.542719 | 224.761693 | 3.320156 |
| Median | **220.892286** | **225.158701** | **3.320156** |

The prior Debug BPF medians were 234.967227 seconds pre-Python,
239.166103 seconds client wall, and 3.311722 ms operator time. Ratios of
the two campaigns' startup and client-wall medians are respectively
**-5.990%** and **-5.857%**. These are descriptive sequential-campaign
comparisons, not randomized paired estimates or confidence intervals.
The block numbers do not turn separately run builds into interleaved pairs.

Release does not remove the multi-minute startup cost. The recorded difference
does not identify Debug compilation as its principal cause, and the new
operator median is not better than the prior Debug median. There is no new
native/CUDA control in this follow-up; the earlier paired mechanism-cost result
remains separate rather than being recomputed against non-interleaved samples.

## Build and measurement scope

The [recorded build](setup-steady-followup/release-build-20260907.md) uses
the same bpftime worktree in a separate `build-pod-release-575` directory.
It changes the host build configuration, not the selector, workload, six
PTX packets, GPU PTX compiler options, or benchmark sample count. Effective
attach flags include `-O2` and target-specific LTO; PTX core/pass flags include
`-O3`. This is not a single-flag ablation. Existing source/dependency state,
build warnings and unchanged original-build paths are disclosed in that note.

All three runs use `bench.py --phase-study`, Llama-3-8B / decode batch 32,
ten warmups and 100 CUDA-event operator samples, with CPUs 8–15. Client wall
includes startup and the unchanged benchmark loop, not just operator execution.
Exit observation has a 0.2-second polling interval. No preflight, extra
correctness campaign, clock calibration or artificial timeout was introduced.
The original benchmark's own output remains in the raw files.

The GLM opt-in stage-timing patch has not been applied or measured. Separating
PTX rewriting, compilation and loading is the next existing-workload task;
these three Release cells are complete and are not queued for repetition.

## Artifacts

The [raw directory](raw/release-runtime-575-20260907-01/) contains exact
commands/environments, logs, operator samples and process timestamps.
Its [summary](raw/release-runtime-575-20260907-01/summary.json) is reproduced
without GPU work using:

```sh
python3 -B workloads/pod-attention/analyze_current_runtime.py \
  workloads/pod-attention/raw/release-runtime-575-20260907-01
```

All observations, including unfavorable operator values, are retained.
No paper files were edited by this coordinating session.
