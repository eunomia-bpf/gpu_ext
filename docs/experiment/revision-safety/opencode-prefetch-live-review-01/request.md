You are an independent read-only systems reviewer. Do not edit files, run
commands, launch GPU work, or use the network.

Review the next Q2 live safety experiment in this repository. Read only these
files:

- docs/experiment/revision-safety/prefetch-invalid-live-plan.md
- docs/experiment/revision-safety/prefetch-invalid-live-plan-review.md
- extension/revision-prefetch/driver-diagnostic.patch
- extension/revision-prefetch/driver-diagnostic.md
- extension/revision-prefetch/driver-diagnostic-review.md
- extension/revision-prefetch/fixture.bpf.c
- extension/revision-prefetch/fixture.h
- extension/revision-prefetch/loader.c
- extension/revision-prefetch/run_safety.py
- extension/revision-prefetch/test_offline.py

The sibling source tree is /home/yunwei37/workspace/gpu/gpu_ext-kernel-575.
Read only these source files there if the read tool can access them:

- kernel-open/nvidia-uvm/uvm_bpf_struct_ops.h
- kernel-open/nvidia-uvm/uvm_bpf_struct_ops.c
- kernel-open/nvidia-uvm/uvm_perf_prefetch.c
- kernel-open/common/inc/nv-gpu-transition-validator.h

Answer four concrete questions:

1. Does the revised three-file diagnostic preserve policy, validation,
   actuation, branches, and return semantics, and does it expose any kernel
   pointer/address or mutable policy surface?
2. What is the minimum observer/loader change needed to replace the unsupported
   structure-return fentry attachment with the new void diagnostic hook?
3. Are the native/BYPASS/invalid99 gates sufficient to support only the narrow
   claim that invalid action 99 falls back to native traversal with correct
   workload output? Identify any false claim or missing reconciliation gate.
4. Give a verdict: READY, READY WITH REQUIRED FIXES, or NOT READY. Separate
   blockers from optional improvements. Do not invent measurements.

Keep the response concise and cite file paths and field/function names.
