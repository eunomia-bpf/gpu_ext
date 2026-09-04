# Device-verifier A1 admission-latency harness

This directory is an independent harness for A0/A1 in
[`verifier-on-device-plan.md`](../verifier-on-device-plan.md). It does not
change the frozen revision-rq4 full throughput campaign. The harness uses the
real `kernelretsnoop` and `threadhist` sources, makes fresh private copies,
targets the Table 1 llama.cpp `kretprobe`, and builds the resulting BPF objects
against one verifier-enabled runtime tree.

No GPU run has been performed while adding this harness.

## Fixed experiment

- A baseline `llama-cli` correctness cell must first reproduce the exact
  normalized 47-byte answer.
- A0 is one fresh STRICT admission/correctness cell for each real Table 1
  object. A1 starts only after both pass.
- A1 contains 10 pairs per tool by default and refuses fewer than 10. A pair is
  exactly one fresh STRICT process and one fresh NO_VERIFY process for the same
  tool and pair index. Each four-cell block contains both modes for both tools.
  Seed `1797` fixes the randomized, within-tool order-balanced schedule.
- Every instrumented cell uses the existing revision-rq4 build adapter,
  `private_probe`, owned `run_cli_separate`, full normalized-output oracle,
  exact `kernelretsnoop` multiplicity/drop gate or complete `threadhist` map
  readback, GPU safety telemetry, and the two existing read-only leases.
- Every cell gets a unique private `/dev/shm/rq4_*` segment. Cleanup must be
  confirmed before the next cell. There are no automatic retries; an invalid
  cell stops the campaign and remains in `result.json`.

The only latency metric is the positive integer in this exact target-runtime
record:

```text
GPU eBPF verification timing: program=cuda__retprobe verification_elapsed_ns=N
```

For STRICT, the target `llama_cli.execution.json` PID must have exactly one
timing record, one STRICT accepted record with the exact program and attach,
and one exact map descriptor. For NO_VERIFY, that PID must have exactly one
skip record and no timing, accepted, map, reject, or unavailable record.
Foreign-PID or malformed admission markers fail the cell. The runner only
parses `llama_cli.log`; probe, agent, wall-clock, token/s, and whole-application
elapsed values are never substituted for verifier latency.

The strict runtime must include the timing/guard contract introduced by
bpftime commit `8eb27cf`, and its existing build must report all of
`ENABLE_EBPF_VERIFIER`, `BPFTIME_ENABLE_CUDA_ATTACH`, and `BPFTIME_LLVM_JIT` as
enabled. Before touching the GPU, the runner also checks that both
verifier-enabled runtime libraries contain every marker reachable in that build
(timing, acceptance, skip, and map) and are no older than the source that
implements the contract. The separately tested verifier-disabled build contains
the fail-closed unavailable-verifier path, which preprocessing correctly omits
from these enabled libraries. Rebuild the runtime after that commit before
running A1.

## Commands

Inspect the fixed 40-cell A1 schedule without touching the GPU:

```bash
python3 run_device_verifier_a1.py --dry-run
```

Run once from a clean, uninjected environment on the admitted 575 host. Keep
the CUDA 12.9 tools in `PATH` as required by the parent revision-rq4 runbook:

```bash
env PATH="/usr/local/cuda-12.9/bin:$PATH" \
  python3 run_device_verifier_a1.py \
  --bpftime-root /home/yunwei37/workspace/gpu/bpftime-table1-575 \
  --bpftime-build-dir /home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575-strict \
  --llama-cli /home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/build-ptx-1b/bin/llama-cli \
  --llama-bench /home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/build-ptx-1b/bin/llama-bench \
  --model /home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --output-dir ./raw/a1-575-01
```

`--resume` may continue only an identical fixed plan with unchanged runtime
agent/syscall-server metadata, built-object metadata, runtime source contract,
and relevant runtime source relative to the initially recorded Git revision.
Unrelated bpftime `HEAD` movement is allowed. It does not rerun an invalid cell
or reuse a partially created unrecorded cell directory.

After the runner reaches `status: complete`, recompute all gates and statistics
without importing runner code:

```bash
python3 analyze_device_verifier_a1.py ./raw/a1-575-01
```

The runner writes incremental machine-readable `result.json`; the analyzer
writes `analysis.json`. The analyzer independently reopens target logs,
execution records, probe summaries, safety records, and private-SHM cleanup
records. It also requires the exact fixed `taskset`/`llama-cli` argv, including
the baseline's lack of preload and the instrumented cells' exact agent preload;
extra or changed generation parameters fail closed. Per tool it reports the 10 STRICT `verification_elapsed_ns` samples,
mean, median, range, and a fixed-seed 95% bootstrap interval for the mean. The
NO_VERIFY member is a matched skip control and has no invented numeric
latency. A valid result requires all 20 pairs across both tools.

## Offline tests

```bash
python3 -m unittest -v test_device_verifier_a1.py
```

The fixtures exercise the fixed schedule, pair floor, exact PID binding,
duplicate/malformed timing, wrong attach and map, NO_VERIFY timing leakage,
missing cells, and reused private shared memory. They do not query or execute a
GPU.
