# Read-only preflight audit

Act as an independent systems-artifact reviewer. Inspect every attached file.
Do not use tools, edit files, run commands, or assume any device result exists.

The intended evidence is deliberately narrow: bpftime's existing standalone
five-instruction eBPF program must generate SPIR-V, execute through OpenCL on
the currently available NVIDIA RTX 5090, and compute 100 -> 142. It is not a
gpubpf attach backend, a cross-vendor port, verifier evidence, or a performance
result. The device execution has not happened yet.

Identify only blockers that could make a future retained preflight false,
unsafe, irreproducible, or overclaimed. In particular check fail-closed device
authorization, source-native path identity, positive correctness/engagement,
SPIR-V structural validation, tamper negative control, shared-lock handling,
process cleanup, GPU/kernel safety, independent analyzer coverage, and whether
the offline tests exercise the important failure modes. Do not request broader
experiments or cosmetic changes.

End with exactly one line:

`VERDICT: PASS`

or

`VERDICT: FAIL`
