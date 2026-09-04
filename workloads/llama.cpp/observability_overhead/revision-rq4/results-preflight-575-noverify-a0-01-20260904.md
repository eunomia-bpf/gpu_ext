# NO_VERIFY device control preflight

This fresh RTX 5090 / driver 575.57.08 control used the same verifier-enabled CUDA/LLVM runtime as the strict campaign, after bpftime `8eb27cf`; only `BPFTIME_VERIFIER_LEVEL=NO_VERIFY` changed. The independent analyzer accepts all five correctness configurations and the complete randomized pp32 timing block with no rejected or retried cell.

Each of the four gpubpf target processes contains exactly one explicit `Skipping GPU eBPF verification for cuda__retprobe` record bound to its execution-record PID. All four contain zero timing, accepted, verified-map, rejected, foreign-PID, unexpected, or unparsed verifier records. This verifies the bypass treatment rather than inferring it from a verifier-disabled binary.

All normal correctness, probe-engagement, process, shared-memory, GPU-safety, and restoration gates passed. The one-block pp32 rates were baseline 7,059.59, gpubpf/NVBit exit recording 132.79/136.31, and gpubpf/NVBit histogram 5,323.57/6,669.64 token/s. These are preflight diagnostics, not a paired steady-state result and not paper performance estimates.

This closes the NO_VERIFY control prerequisite for A1/S0. It does not itself compare STRICT with NO_VERIFY because the two preflights are separate processes and schedules. Primary machine evidence: `raw/preflight-575-noverify-a0-01/result.json` and `raw/preflight-575-noverify-a0-01/independent-analysis.json`.
