# OpenCode review request: stale-state driver bridge v1

Review only the attached source artifacts. All OpenCode tools are disabled;
do not propose or perform writes, shell commands, network calls, or delegation.

Return exactly one leading verdict: `READY FOR SOURCE/BUILD GATE` or
`REQUIRED FIXES`. Then give concise, evidence-based findings, each naming the
relevant file and construct.

Audit these properties:

1. The driver patch is applyable against NVIDIA 575 revision `6a5b3bb5` and
   keeps the existing six `gpu_mem_ops` fields unchanged while appending the
   versioned callback.
2. Old six-member struct_ops policies remain compatible through a zero tail;
   flag any claim that still requires a live load test.
3. `(sequence, phase, source_mono_ns, published_mono_ns)` is one immutable,
   atomically published driver-owned snapshot with coherent generation and
   lifecycle handling.
4. Native and BPF consumers use the same snapshot, timestamp, action model,
   validator, and effect point.
5. BPF gets a read-only input and can affect state only via exactly one trusted
   action request. The new setter must not be callable from KPROBE programs.
6. Missing, stale, malformed, mutated, duplicate, conflicting, or mismatched
   decisions fail safely and are observable.
7. Common counters and address-free diagnostics can reconcile snapshot reads,
   decisions, and final effects for both arms without proxy inference.
8. CPU/ABI/BPF/module-build checks support only a source/build readiness claim;
   no text may imply module load, verifier acceptance, attach, GPU engagement,
   performance, or experiment completion.
9. Look specifically for Linux 6.15 API errors, unload/configure races,
   sleeping in an unsafe context, RCU lifetime errors, counter asymmetry,
   ABI-layout drift, and mismatch between the active analyzer schema and the
   driver fields.

Treat the repository libbpf excerpt in the patch/readiness context as the
basis for the zero-tail explanation. A live compatibility proof remains a
future gate.
