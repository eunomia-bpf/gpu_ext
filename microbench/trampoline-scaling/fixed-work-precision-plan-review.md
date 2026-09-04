# Fixed-work precision plan review

An independent read-only reviewer examined the prospective plan against the
prior fixed-work plan, implementation, and raw result. The reviewer did not
edit files or run the GPU.

Verdict: **READY**. No blocking scientific or executability defect was found.

The review confirmed that:

- the comparison remains the identical CUDA binary with and without the
  current return-only device handler; 512 launches only lengthen the same CUDA
  event observation, while per-kernel geometry, work, hook repetitions, and
  output remain fixed;
- all three arms use the same warmup, launch count, and geometry, while the
  counter arm remains an engagement control excluded from equivalence testing;
- 48 blocks are frozen prospectively, all six arm permutations occur eight
  times, and neither precision-triggered retries nor optional stopping are
  allowed;
- the old ten-block result is used only for explicitly exploratory power
  planning and is never pooled with the confirmatory follow-up;
- the endpoint 95% interval and four Bonferroni-adjusted 98.75% guards preserve
  the original estimand, multiplicity control, margin, and decision rule; and
- even a positive result is limited to these five fixed-work organizations on
  this synthetic kernel and GPU. It cannot imply universal block-count
  independence or warp-leader execution.

The reviewer also accepted the power calculation as a prospective budget
choice rather than promised power: its square-root aggregation assumption is
unverified, so the fixed 48-block run may honestly remain inconclusive.
