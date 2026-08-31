# Plan Review: RQ4 RTX 5090 Observability Overhead

## Round 1

**Verdict: APPROVE WITH REQUIRED REPAIRS**

The experiment is decision-relevant and directly addresses mandatory revision item R6. The hypothesis, single strongest baseline choice, workload, and per-tool interpretation are appropriate. Execution must not begin until the following defects are repaired.

### Blocking required repairs

1. **Use an officially supported NVBit driver stack for the paper comparison.** The host is on driver `610.43.02`, while the current official NVBit requirements still state driver `<=575.xx`; NVBit v1.8 is latest, but does not remove that published limit. A run or failure on 610 is diagnostic only and cannot establish either fair NVBit performance or conclusive incompatibility. Run all seven configurations on the same RTX 5090 under a supported 575.x driver, or leave the experiment incomplete. Pin the NVBit artifact/version/hash and record the full driver/toolkit stack.
2. **Define and implement genuinely matched NVBit semantics before preflight.** The current runner has no NVBit path. Freeze three custom adapters atop the official NVBit release, with the same target symbol, hook point, event payload, aggregation/output volume, CUDA-graph setting, and no instrumentation of unrelated kernels. Call them “matched custom adapters using official NVBit,” not official NVBit tools.
3. **Resolve incorrect or unsafe observability semantics.** The plan calls `kernelretsnoop` per-block, but its source records thread coordinates and timestamps and does not record block identity. `threadhist` currently prints only seven counters. `launchlate` stores a single last host-launch timestamp, which can be overwritten by unrelated asynchronous launches before the selected kernel executes. Fix these semantics or narrow/rename the paper rows, then make NVBit match the corrected definitions.
4. **Make the workflow and raw path executable.** Add the real NVBit runner path and exact build, preflight, full-run, analysis, and resume commands, explicitly passing `--output-dir`. Preflight every distinct tool path.
5. **Implement the declared validity, repetition, and uncertainty protocol.** Use 10 randomized/interleaved repetition blocks; require successful execution, expected prompt count, nonzero sane events, and application-output correctness for every instrumented repetition. Preserve failures but exclude them from performance estimates. Predeclare the paired effect as `NVBit overhead - gpubpf overhead` and compute a fixed-seed paired bootstrap 95% CI.
6. **Correct the completion and timeout rule.** Attempts on unsupported driver 610 cannot complete the hard NVBit commitment. Completion requires 10 valid repetitions for all seven cells on the supported matched stack; otherwise status is incomplete/inconclusive. Reconcile the no-progress threshold with runner timeouts after preflight.

### Non-blocking suggestions

- Cite the official NVBit release/requirements, NVBit MICRO'19, and the pinned llama.cpp implementation defining tokens/s.
- Record clock, power, and temperature state and report exclusions and per-tool results separately.

### Author response

Accepted. The experiment remains blocked from paper-facing execution on driver 610, but implementation repairs and non-paper diagnostic checks may proceed. The plan and runner will be revised before a follow-up review.

## Round 2: first follow-up

**Verdict: APPROVE WITH REQUIRED REPAIRS**

The reviewer accepted the exact-symbol adapters, corrected gpubpf semantics,
seven-cell randomized blocks, paired bootstrap, safe GPU admission, and the
disclosed difference between NVBit's native CUDA launch callback and gpubpf's
exact-stub uprobe. Five remaining blockers were identified:

1. Official preflight, not only the full run, must require driver 575.x.
2. An untimed deterministic application-output checksum must match across all
   seven configurations before performance measurements.
3. The plan must not claim independent coordinate bounds or an independent
   exit count that the collectors do not measure.
4. NVBit threadhist must not use stronger atomic increments than gpubpf.
5. New runs and resumes must prevent output overwrite and verify all defining
   parameters, sources, binaries, and driver state.

### Author response and repair status

Accepted and implemented. Both official phases now reject unsupported drivers;
all seven paths must pass a fixed-prompt, fixed-seed `llama-cli` stdout checksum
and probe-engagement gate; the plan narrows the collector claims; NVBit uses the
same ordinary per-thread increment semantics; and each run preserves hashed
tool copies under its raw directory. New runs refuse nonempty output paths, and
resume verifies parameters, driver, model/llama binaries, sources, and tool
artifacts before continuing. These changes await the final permitted follow-up
review and the real runtime preflight on an idle RTX 5090 with driver 575.x.

## Round 3: final follow-up

**Verdict: REJECT**

The final permitted review accepted the supported-driver gate, narrowed probe
claims, matched threadhist increment semantics, and non-overwriting output
paths, but found two executability blockers:

1. NVBit's banner is written to stdout and would make every NVBit deterministic
   output checksum differ from the baseline unless `NOBANNER=1` is explicit.
2. Resume did not compare the bpftime git revision or hashes of
   `libbpftime-agent.so` and `libbpftime-syscall-server.so`, so rebuilt runtime
   libraries at the same paths could be mixed across blocks.

Both defects were subsequently repaired in the runner: all NVBit correctness
and timing environments now set `NOBANNER=1`, and new/resumed state records and
verifies the bpftime revision and both runtime-library hashes. These repairs
have not been independently approved because the workflow permits no further
follow-up for this proposal. The proposal is therefore closed as rejected; its
implementation and raw admission evidence are retained for a future newly
reviewed experiment after a supported driver is installed.
