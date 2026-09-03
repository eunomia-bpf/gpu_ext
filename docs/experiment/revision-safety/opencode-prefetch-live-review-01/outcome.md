# OpenCode Q2 prefetch review outcome

These are advisory model reviews, not measurements. All attempts used new
owned CLI processes pinned to CPU 17, disabled sharing, snapshots, updates and
tools, and never launched GPU work or edited files.

## Attempts

1. The initial GLM command exited 1 because `--file` consumed the following
   positional message. [`stderr.log`](stderr.log) contains the exact CLI error;
   [`events.jsonl`](events.jsonl) is empty.
2. The corrected GLM call received only the attached request, repeatedly tried
   unavailable Read tools, then honestly declined to review unseen files.
   [`retry-events.jsonl`](retry-events.jsonl) retains the complete exchange.
3. A GLM retry attached the requested source files directly. It emitted only a
   step-start event and timed out after 600 seconds; see
   [`retry2-events.jsonl`](retry2-events.jsonl). It produced no review.
4. `spark-gateway/qwen3.8-flash-next-nvfp4-220k` received the exact request,
   plan, driver patch, current fixture header/BPF source, and loader as
   attachments. It exited 0 and returned the preserved
   [`final review`](final-review.md); the raw stream is
   [`retry3-events.jsonl`](retry3-events.jsonl).

## Useful findings and one rejected finding

The successful review independently confirmed that the driver context contains
no pointer/address token, the driver patch is observational, the three tracing
programs plus one fixed policy are a minimal replacement for the unsupported
structure-return observer, and the stated result/effect/traversal/output gates
support only the narrow fallback claim.

It labeled the observer ordering a blocker, asserting that diagnostic SELECTED
and FINISHED occur before the fexit of `uvm_bpf_call_gpu_page_prefetch`. That
reading is incorrect. In the actual driver, `compute_prefetch_region` calls
`uvm_bpf_call_gpu_page_prefetch`; the fexit executes as that wrapper returns,
then the following statements validate the request, select the effect and call
the SELECTED diagnostic. FINISHED occurs later in the same caller. Therefore
the implemented order is wrapper entry → policy → wrapper exit → SELECTED →
FINISHED, exactly the fixture state machine. The build disassembly also retains
both diagnostic calls after the wrapper call. No source change is made for this
false blocker.

The review could not verify the unattached runner and live environment. Those
are covered separately by the offline tests, source review, module BTF/static
inspection, and the still-pending real admission. It is not treated as an
approval of a live result.
