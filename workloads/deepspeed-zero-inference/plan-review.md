# Plan Review

## Round 1

Status: **BLOCK**

The independent review found four load-bearing issues:

1. The gpubpf cell required a custom BTF-enabled UVM module while the plan
   prohibited module replacement, so the shared driver stack and exact live
   engagement gates were undefined.
2. The DeepSpeed baseline lacked pinned Torch/Transformers versions, exact CPU
   and NVMe configs, initialization order, launcher shape, and a bounded tuning
   set.
3. Correctness and engagement did not rule out MXFP4 dequantization,
   non-ZeRO/partial offload, or full-weight GPU materialization; CPU and NVMe
   outputs were not required to agree, and timing boundaries differed.
4. The inherited schedule covered four rather than five cells, and the shared
   three-attempt budget lacked NVMe allocation, timeouts, and memory/disk
   containment.

Repair: revision 2 freezes one custom-610 UVM stack for every cell, exact
software/API/config/launcher choices, strict ZeRO and MXFP4 engagement gates,
cross-tier correctness, deterministic preflight allocation, resource limits,
and a five-cell Latin-square schedule with same-slot retries.

## Round 2

Status: **BLOCK; candidate closed before execution**

The revised retry allocation, UVM stack, resource caps, correctness structure,
and schedule were coherent, but source inspection established a decisive
incompatibility. Transformers 5.16.1 deletes each MXFP4 expert projection from
the module's `_parameters` and attaches a Triton tensor object. Official
DeepSpeed ZeRO-3 offloads parameters, so it cannot include the packed expert
weights in its partition inventory. Allowing custom glue would replace the
baseline mechanism under test. A valid positive cell was therefore
structurally unavailable and no expensive preflight was authorized.

Two protocol issues would also have required repair if the baseline were
executable: direct `generate` and llama HTTP timing boundaries were not matched,
and failure of the optional NVMe tier should not veto a valid CPU baseline.
They are moot for this exact GPT-OSS-120B MXFP4 path and are retained here to
prevent reuse of the rejected protocol.

## Round 3

Status: **CLOSE-APPROVE**

The final read-only audit confirmed that the closure is source-grounded, states
that no run or result occurred, preserves the exact-model boundary, and routes
to the separately promised Expert Buffering policy implementation without
substituting a weaker model. No factual blocker remains in the availability
record.
