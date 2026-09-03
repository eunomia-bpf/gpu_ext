# Independent plan review

2026-09-03, root reviewer; implementation owner: scheduler branch.

The question adds evidence beyond fixed GPreempt: does event-confirmed
idle-interval admission recover BE progress at retained foreground protection?
The native/fixed/idle-C/idle-BPF comparison is fair in purpose, uses real DNN
outputs, and distinguishes policy effects from C/BPF mechanism cost. The
original author scheduler is unavailable in the checked sources; the plan
properly labels source-level splitting as a paper-component port, excludes
the unrelated multi-GPU subsystem and does not substitute a host sleep for
GPU-idle evidence. Implementation preparation is admitted.

Resolve before GPU execution:

1. Mapping the final trace arrival to one nanosecond before the window closes
   forces a deadline miss. Match the periodic case's final arrival at
   59,990,000,000 ns instead, preserving scaled gaps and ties. Identify the
   exact public trace file/revision and first chronological 6,000 successful
   rows. Genuine burst overload remains part of the outcome.
2. Freeze the CUDA contexts and timeslice configuration for every arm.
   Idle-C and idle-BPF must match; explicitly disclose the intended difference
   from fixed GPreempt instead of inheriting a context policy accidentally.
3. SLO attainment uses every offered request, including never-started backlog.
   Conditional LC p99 cannot prove retained protection. Declare a recovery
   win only when paired BE ratio's lower 95% bound exceeds one, LC p99 ratio's
   upper bound is at most 1.01, and SLO-attainment change's lower bound is at
   least −1 percentage point. Otherwise report a tradeoff or uncertainty.
4. A predicted tick does not prove device completion. The proposed common
   nonblocking event-completion check is appropriate; count its waits and
   disclose that stronger fence's possible overhead rather than claiming the
   original paper's overhead. Freeze the actual built-client command and
   profile settings before the real preflight.

Keep the implementation confined to the new workload; reuse the existing
executor, model assets, leases and numerical checks. The restricted transform
must preserve 3-D CTA coordinates and exactly-once execution, with unsupported
operations reported rather than dropped. No broad scheduler framework or
frozen GPreempt changes are needed.

Follow-up on the now-explicit context setup: both idle arms use 1,000,000 µs
for LC and BE, unlike fixed GPreempt's 1 µs BE request. Add a labeled control
with the same idle-arm contexts/timeslices and ordinary unsplit, ungated model
execution. Otherwise recovered throughput could come entirely from removing
the BE timeslice restriction; the comparison would not isolate idle-interval
benefit. This control is not another main baseline. The complete matrix is
then five arms × two arrival cases × five blocks = 50 timed cells. It does not
alter the old completed GPreempt study.

Implementation review before GPU calibration: the copied DISB frontend builds
with exact patch checks; original GPreempt sources/binaries remain untouched.
The source transform, XYZ iterator, actual host JIT and profile checks pass CPU
tests, and all four client/profiler artifacts build for sm_120. The reviewer
checked the per-piece event fence, HP publication lock, no-replay consolidation,
full-model profiling passes and final output checks. These are preparation,
not GPU correctness or performance results. Actual calibration is now admitted.

Small-pattern qualification uses five randomized 10-second blocks of four
idle-C variants: none/input/output/both. Only variants with actual corresponding
launches, full LC completion coverage and paired p99 upper 95% bound <=1.01 are
eligible. Enable both only if both individual patterns and their combination
pass; otherwise prefer eligible input, then output, then neither. This fixes
the choice before observing results. Root's runner records all offered SLO
requests and late completions; its initial combination-selection bug was fixed
after independent review and covered by a regression test. A separate isolated
60-second LC run sets the SLO. Profiling, qualification and preflight are excluded
from the 50-cell performance matrix.
