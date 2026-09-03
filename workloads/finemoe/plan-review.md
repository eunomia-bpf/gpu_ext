# Independent plan review

2026-09-03, root reviewer; implementation owner: MoE branch.

Admitted for implementation. The four-arm comparison directly tests dynamic
candidate selection and real transfer waste on the original Qwen model, then
separates the native-C/BPF mechanism cost. The all-positive arm is correctly
labeled a set-selection ablation, not a strong baseline manufactured from an
upstream defect. The full probability/embedding/search inputs distinguish this
from an EAMC-count heuristic. A frozen history still permits per-query dynamic
sets, but does not establish online adaptation to changing workload history.

The reviewer executed the unmodified upstream methods on CPU. For probabilities
[0.6, 0.3, 0.1], threshold 0.8 and K=1, the selected mass is about 0.6; after
`process_expert_map`, all three candidates become positive. Both defects affect
the question materially, so common replayable fixes are necessary. No CUDA was
initialized by this check and it is not a performance experiment.

Before the real preflight:

- Repair and test current-sequence trajectory lifecycle on the common path;
  stale historical entries must not be paired with the current sequence ID.
- Freeze the original model revision, disjoint 64/8 prompt IDs and numerical
  tolerance from demand-only repeats. Do not relax it after a BPF failure.
- Check BPF's explicit numeric representation against an independent corrected
  Python oracle, on actual inputs and threshold/tie boundary cases. Compiling
  the same C twice is insufficient algorithm validation.
- Verify selection at actual enqueue/copy events, not just the selector mask.
  Partition completed speculative bytes before cache teardown into useful,
  evicted-unused and still-resident-unused. Teardown-induced eviction is not
  evidence of wasted inference traffic; pending/canceled copies remain separate.
- Freeze the built-client command and cache budget before GPU timing. The
  download/build stage does not establish GPU fit or complete reproduction.

Keep the source patch, native/BPF selector and necessary accounting local to
this workload. Retain common demo-distance/search/eviction differences as
declared component boundaries rather than expanding this into a new runtime.
