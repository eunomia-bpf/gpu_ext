# Read-only review of the Fig. 15 device-map plan

- Reviewer: OpenCode 1.18.27 with the locally configured
  `spark-gateway/qwen3.8-27b-nvfp4-200k` model.
- Session: `ses_f92f2d3adffeHNNMylunmRUvxd`.
- Isolation: pure mode with snapshot, sharing, shell, web, task, write, and edit
  capabilities denied. The review read only the plan, audit, runner, analyzer,
  tests, CUDA target, loader, and BPF source.
- Verdict: **PASS**.

The reviewer accepted the one-hypothesis admission, the distinction among
device-resident, directly host-mapped, and serialized RPC arrays, the matched
lookup/update bodies, CUDA-event batching, correctness gates, balanced paired
schedule, Bonferroni co-primary intervals, independent raw replay, and the
STOP boundary for the missing warp/eGPU comparison.

It identified two fail-closed preflight conditions:

1. Runtime engagement records can be routed to either `application.log` or
   `agent.log`. The runner and independent analyzer now search their union for
   the exact target transformation, module-load, and attach records while
   separately requiring the bootstrap records in `agent.log`. An offline test
   covers split routing. Existing trampoline-scaling raw logs also show that
   the selected runtime currently emits transformation/load/attach records to
   the application stream despite a file-valued `BPFTIME_LOG_OUTPUT`.
2. The loader must be able to read back the type-1503 device-resident array
   through its host API. This cannot be established without using the GPU and
   remains an explicit preflight STOP: failure invalidates the campaign and
   cannot be replaced with the historical aggregate.

The reviewer also suggested checking the one-hour budget from the preflight.
The full run remains fixed at 128 arm processes with no timing-based retries or
optional stopping; a deadline failure leaves the campaign incomplete.
