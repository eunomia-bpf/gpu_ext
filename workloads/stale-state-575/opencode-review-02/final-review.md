# OpenCode final review

Session: `ses_f94856b14ffeT6eBzjR0ET3sAC`  
Model: `opencode/ling-3.0-flash-fin-free`  
Mode: `--pure`; snapshot/share disabled; write, edit, bash, webfetch, and task
tools disabled; all permissions denied

## Verdict: READY FOR SOURCE/BUILD GATE

1. **Patch applicability and struct_ops preservation.** The patch retains all
   six existing `gpu_mem_ops` callbacks and appends
   `gpu_stale_state_prefetch_v1` at `6 * sizeof(void *)`; the kernel and BPF ABI
   mirrors both assert the layout.
2. **Zero-tail compatibility claim is correctly bounded.** The readiness
   record bases it on repository libbpf's kernel-sized zero allocation and
   member copy, but still requires a live old-policy load test.
3. **Immutable snapshot and RCU lifecycle are sound.** Publication captures
   driver monotonic time, publishes one immutable object through RCU, and
   enforces generation/order checks; exit synchronizes before final free.
4. **The consumer paths are matched.** Native and BPF receive the same
   snapshot, decision timestamp, legal bounds, model actions, validator, and
   final effect path.
5. **BPF uses a read-only context and trusted setter.** The policy copies input
   with `bpf_probe_read_kernel`; the setter is registered only for STRUCT_OPS,
   not KPROBE.
6. **Failures are observable and safe.** Missing/invalid snapshots, input
   mutation, missing/duplicate/conflicting requests, callback/request
   disagreement, and invalid actions become recorded errors rather than
   unchecked effects.
7. **Common counters and address-free diagnostics are sufficient for the
   planned reconciliation.** Both arms share snapshot, decision, effect, and
   selected/finished accounting, with separate native/BPF invocation counts.
8. **Claims remain source/build-only.** No reviewed text claims module load,
   verifier acceptance, attach, GPU engagement, performance, or experiment
   completion.
9. **Targeted source search found no Linux 6.15 API, sleeping-context, RCU,
   counter-asymmetry, or ABI-layout blocker.**

The review also noted that the full revision in the readiness record is the
expanded form of the request's abbreviated `6a5b3bb5`; this is not a mismatch.
