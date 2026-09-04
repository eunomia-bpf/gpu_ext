# STRICT device actual-object gate A0-03

The final A0 campaign is complete on the RTX 5090 with driver 575.57.08. It used the verifier-enabled CUDA/LLVM runtime at bpftime `b266cf2`, the target-PID/map/cardinality gate at gpu_ext `7e8e44d`, and fresh real `kernelretsnoop` and `threadhist` objects. The independent analyzer accepts all five correctness configurations and the complete randomized pp32 timing block with no rejected or retried cell.

Each of the four gpubpf cells binds evidence only from its target application log to the PID in the corresponding execution record. Every cell contains exactly one STRICT acceptance, exactly one phase-specific expected map, and zero skipped, rejected, foreign-PID, unexpected, or unparsed verifier records:

- `kernelretsnoop`: 60 instructions; type 1527/key 4/value 32; max entries 256 in correctness and 44 in timing.
- `threadhist`: 13 instructions; type 1502/key 4/value 8/max entries 1 in correctness and timing.

All correctness cells produced the exact 47-byte application oracle. The gpubpf and matched NVBit exit paths each observed the exact 720,896 records and complete 220-launch/22,528-coordinate oracle; both histogram paths observed 720,896 samples and 22,528 active threads, and gpubpf completed its 1,048,576-entry readback. All recorded process, shared-memory, GPU-safety, and restoration gates passed.

The one-block pp32 rates were baseline 7,034.56, gpubpf/NVBit exit recording 133.21/135.56, and gpubpf/NVBit histogram 5,296.53/6,671.40 token/s. They are preflight diagnostics, not paper performance estimates. The paper-facing NVBit comparison remains the existing ten-block pp512 result; A1 admission latency and S0 STRICT-versus-NO_VERIFY steady-state pairing remain separate open gates. `launchlate` is outside this campaign because its cross-clock measurement is not yet valid.

The earlier A0-01 false-negative and A0-02 intermediate pass remain immutable rather than being overwritten. Deny-all OpenCode/Qwen review session `ses_f93941e51ffeETPVJDfC095rq7` independently returned PASS on the final target-bound evidence logic, while noting the ordinary limitation that a JSON-only analyzer assumes honest raw record generation.

Primary machine evidence: `raw/preflight-575-strict-a0-03/result.json` and `raw/preflight-575-strict-a0-03/independent-analysis.json`.
