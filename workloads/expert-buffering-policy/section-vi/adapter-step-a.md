# Adapter step A: CPU state and selector interface

2026-09-03: implemented and CPU-tested; **not yet connected to GPU transfers**.
The existing selector's nine tests / 2,131 decisions remain separate evidence.

`adapter_state.{h,cpp}` adds fixed whole-node cohort registration, one active
layer invocation with begin/end epochs, unchanged residency across batches,
snapshot revalidation, and copy-completion/admission accounting. Its owned
dynamic-library/JIT handle lets a future C++ worker call native or actual
host-uBPF directly without a Python/GIL callback. FIFO shares validation,
hit/admit/block handling and capacity, then uses oldest insertion for its victim;
native and BPF use the existing inactive-first/LIFO selector. No policy fallback.

`Validate` checks epoch, cohort, current residency, actual input immutability,
and output/status consistency. HIT requires a resident active incoming expert;
ADMIT requires an active miss and space; EVICT requires a full cache, active miss
and eligible resident victim; BLOCKED requires a full cache without an eligible
victim. INVALID, out-of-range statuses and incompatible statuses are rejected.
These checks do not choose a victim again or replace the BPF selector.

**Live boundary:** the state object has no CUDA calls or node locks. Its caller
must hold the adapter metadata lock, verify actual Node residency/eligibility
under node/execution locks, and invoke `Admitted` only after the real whole-node
copy has succeeded. The CPU tests call completion methods as fixtures; they do
not establish that a live completion or stale-commit safeguard is wired in.
The private adapter source patch/build/GPU checks are the next step.

Validation:

```sh
timeout 30s taskset -c 17 make -C workloads/expert-buffering-policy/section-vi -j1 test-adapter
```

The initial [test.log](adapter-cpu-01/test.log) passed 1,399 checks. After root
review requested explicit status-state consistency checks,
[test-02.log](adapter-cpu-01/test-02.log) passed 1,432 checks, including 650 paired
native/actual-JIT decisions, duplicate/cross-layer mapping rejection, overlapping
invocation rejection, stale epochs/residency, unchanged serial before completion,
hit-versus-insertion ordering, unavailable victims, forged statuses, serial
exhaustion, and failed JIT open. Both commands exited 0 within one second on
CPU 17; neither imported torch, rebuilt the offloader, or accessed a GPU.

This step changes only the independent section-vi sources/Makefile/tests.
The frozen FineMoE source and all prior raw experiments remain unchanged.
Preserve both original logs explicitly when committing (`*.log` is ignored).
