# Independent plan review

2026-09-03, root reviewer; implementation owner: device branch.

Admitted for implementation. Real attention operators and original SM-local
atomic selection answer a device-policy question absent from the host-policy
results. Serial/two-stream attention establish fusion context, while original
inline CUDA and same-ABI CUDA separate adapter cost from BPF policy execution.
No fusion gain may be assigned to BPF. Neutral or slower BPF is a valid result.

Explicit device output fields are appropriate because the existing GPU JIT
discards scalar r0. The actual attention branch must consume the BPF-written
operation and task ID, and the BPF program must own the atomic claims. A
counter-only probe, host callback or always-native fallback cannot pass.
Common identifier-bound and allocation/reset/lifetime fixes are required on
the sm_120 port; the 99 KiB limit must be checked using the real head-128
operator, not avoided by silently substituting an easier numerical kernel.

Before GPU execution, add the verbatim paper RQ4 and retain the separate R2
expressibility motivation. Freeze input seeds, exact build/run command and
per-cell estimator (arithmetic mean of all 100 timed operator calls, with
paired inference over five blocks, is appropriate). Keep all raw samples;
do not treat 100 calls as 100 independent experimental blocks.

Correctness diagnostics should validate each atomic ticket and logical slot,
the exhausted-operation fallback and exactly-once work. CUDA hardware may
assign different SMs across runs, so identical trace order is not a valid
cross-arm requirement. Preserve the predeclared numerical tolerance and
diagnose any original/reference failure before interpreting BPF performance.

The minimal new ABI/compiler-call adapter is justified by the missing decision
return path. Reuse the existing compiler and official numerical kernels;
do not edit the dirty bpftime repository or add a generic scheduler framework.
Build-only checks remain dependencies, not a completed operator experiment.
