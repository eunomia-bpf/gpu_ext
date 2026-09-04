# Device-verifier S0 steady-state harness

This directory implements the frozen S0 comparison in
[`verifier-on-device-plan.md`](../verifier-on-device-plan.md) without changing
the revision-rq4 full campaign or the A1 admission-latency harness. It uses the
real Table 1 `kernelretsnoop` and `threadhist` objects and one verifier-enabled
bpftime runtime for both instrumented treatments.

No GPU run was performed while implementing or testing this harness.

## Frozen matrix and metric

- For each tool, one pp32 correctness block contains exactly one fresh
  `{control, STRICT, NO_VERIFY}` client.
- After all six correctness cells pass, each tool receives ten pp512 complete
  blocks. Every block contains one fresh process for each treatment, in a
  fixed-seed randomized order. Tools are interleaved by block. Seed `1797`
  fixes the schedule.
- `llama-bench` warmup remains enabled. Its one measured `avg_ts` sample starts
  after runtime admission, attachment, and warmup.
- A control cell is a fresh uninstrumented client and therefore intentionally
  has no bpftime shared memory. Every STRICT/NO_VERIFY cell creates a new
  private `/dev/shm/rq4_*` segment and an owned loader; all 44 segment names
  must be unique and confirmed removed.
- Any invalid cell ends the campaign. There are no automatic retries. Resume
  only continues an identical plan and refuses an unrecorded partial cell
  directory.

For every tool/block, the primary paired effect is:

```text
100 * (STRICT pp_tok_s / NO_VERIFY pp_tok_s - 1)
```

The analyzer also reports STRICT-vs-control and NO_VERIFY-vs-control with the
same ratio definition. Positive means the numerator treatment has higher
throughput. It reports the ten effects and a fixed-seed, 10,000-resample 95%
bootstrap interval for their mean. No equivalence conclusion is produced
because the frozen plan does not specify a numerical equivalence margin.

Runtime `verification_elapsed_ns` is only an admission-engagement gate. It is
never used in a throughput value, ratio, or interval.

## Fail-closed evidence

Before any GPU access, the runner requires:

- `ENABLE_EBPF_VERIFIER`, CUDA attachment, and LLVM JIT enabled in the same
  build directory;
- both runtime DSOs newer than the relevant source and containing the enabled
  timing/accept/skip/map literals;
- the source-only verifier-unavailable strict guard;
- the exact Table 1 program, attach section, and map definitions in fresh tool
  copies.

Every cell reuses the revision-rq4 owned process and GPU safety machinery. The
independent analyzer does not import the runner. It reopens each target log,
`execution.json`, safety record, telemetry CSV, probe log, and private-SHM
cleanup record. It requires:

- one final signed-integer target-log exit footer with no trailing content,
  equal to the independently recorded successful return code;
- independently rederived before/after 575-driver, 400 W, idle-GPU,
  kernel-log, UVM, `struct_ops`, boot-ID, worker-CPU, and cleanup gates;
  telemetry headers, every row, throttle reasons, sample count, peaks, mean
  power, and clock range are reparsed and must reproduce the recorded summary;
- exact target-PID STRICT admission/timing/map or NO_VERIFY skip records, and
  no admission records in control;
- exactly one valid pp32 or pp512 `llama-bench` JSON row with the fixed model,
  GPU layers, prompt size, one positive sample, and warmup-enabled command;
  `avg_ts` must agree exactly with the elapsed-time-derived `pp*1e9/avg_ns`
  within the fixed six-decimal `avg_ts` tolerance. It must also agree with
  `samples_ts[0]` within half of that field's six-significant-digit print unit,
  matching `llama-bench`'s default C++ stream representation;
- exact pp-dependent `kernelretsnoop` events, coordinates, multiplicity,
  map capacity, and zero-drop gates, or complete `threadhist` readback;
- equal positive STRICT and NO_VERIFY `threadhist` event counts within every
  correctness or timing block;
- exactly 6 correctness and 60 timing cells, with no missing, extra, or
  duplicate sequence and ten complete blocks per tool. Directories must use
  canonical `<stage>/<three-digit-sequence>-<tool>-<treatment>` names, and all
  66 `(pid,start_ticks)` target identities and raw directories must be unique;
- one nonempty, identical `llama-bench` `build_commit` and positive
  `build_number` across all 66 raw benchmark rows.

The exact target command is also reconstructed: CPU binding followed by the
fixed `-r 1 -o json -p <pp> -n 0 -ngl 99` workload. Control forbids
`LD_PRELOAD` and all `BPFTIME_*` injection variables. Instrumented cells require
the exact agent DSO preload and the raw private-probe environment for their
treatment.

## Commands

Print the fixed schedule without touching the GPU or locks:

```bash
python3 run_device_verifier_s0.py --dry-run
```

Run without `sudo` in the current admitted session:

```bash
env PATH="/usr/local/cuda-12.9/bin:$PATH" \
  python3 run_device_verifier_s0.py \
  --bpftime-root /home/yunwei37/workspace/gpu/bpftime-table1-575 \
  --bpftime-build-dir /home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575-strict \
  --llama-bench /home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/build-ptx-1b/bin/llama-bench \
  --model /home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --output-dir ./raw/s0-575-01
```

The run has 66 `llama-bench` cells: 6 correctness and 60 timing. Each client
has a 300-second timeout; the conservative client-time ceiling is 5.5 hours,
plus tool build, 3-second startup for each of 44 loaders, telemetry, and cleanup.

Analyze only after `result.json` reaches `status: complete`:

```bash
python3 analyze_device_verifier_s0.py ./raw/s0-575-01
```

The runner incrementally writes machine-readable `result.json`; the analyzer
writes `analysis.json`.

## Offline tests

```bash
python3 -m unittest -v test_device_verifier_s0.py
```

The fixtures cover schedule reconstruction, disjoint admission contracts,
target-PID binding, duplicate/forbidden timing, raw throughput recomputation,
private-SHM reuse, missing/extra/duplicate cells, exact exit-footers, every
fixed safety gate, strict telemetry replay, and cross-cell build identity.
They do not access a GPU or acquire experiment locks.
