# Independent review of the Q2 live prefetch plan

Date: 2026-09-03. This is a read-only review of the next experiment, not a
measurement or driver admission result.

Three independent audits ranked the live invalid-prefetch fallback above
another completed-baseline rerun. It directly addresses Q2's transition-safety
concern and is distinct from the existing strict SIMT rejection and PMM reorder
tests. RTX 5090 Table 1 remains the next hard revision commitment, but its
seven-arm correctness preflight is not yet admitted.

The proposed three-file driver seam preserves the current policy dispatch,
validation, effect-selection branches, and returned region. Its void return and
const context avoid the structure-return fentry failure that stopped attempt01,
but source inspection alone cannot establish BTF or live admission.

One blocking issue was found before implementation: the first proposal copied
the stack-context and bitmap-tree addresses into two `u64` identity fields.
Even if reports omitted them, this needlessly exposed kernel addresses to a
tracing observer. Both fields and assignments must be removed before applying
the patch. A full `pid_tgid` plus a strict one-outstanding-frame-per-task rule is
sufficient here; duplicate SELECTED, nesting, or unmatched FINISHED events must
fail the run.

Non-blocking minimization advice is to omit fixed constants and fields unused
by the stated gates. The live admission must still inspect the exact built
prototype and instrumentation, attach the real observer, execute all three
controls, verify all 131,072 values per target, and restore the prior module and
services. No diagnostic-module timing result is publishable.
