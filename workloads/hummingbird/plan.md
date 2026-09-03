# Hummingbird idle-interval scheduling — implementation proposal

Prepared 2026-09-03 UTC. Calibration, 20 small-pattern qualification cells,
ten preflight cells and the full 50-cell comparison are complete. The full
result is negative for background-throughput recovery; the frozen protocol
below is retained unchanged. See [current results](results-575-20260903.md).
The completed
[GPreempt 45-cell study](../gpreempt/results-load-study-575-20260903.md), its
source/build trees and results remain frozen. Root independent review admitted
this implementation and the calibration protocol below; no further user
approval is needed. No driver replacement or independent XSched runtime rewrite.

## Question and admission

Paper RQ3, verbatim: “Does \sys improve tail latency, throughput, and resource
fairness compared to user-space and global policies in multi-tenant settings?”

Test whether published split-kernel/idle-interval scheduling can recover
background goodput lost to fixed foreground protection, and whether executing
that same decision rule through BPF preserves its C implementation's behavior.
This is supporting RQ3/policy-versus-mechanism evidence, not a claim that BPF
invents the policy. Existing heavy-supply GPreempt results establish a real
latency/goodput tradeoff but cannot explain or test this alternative scheduler.
Unlike another repetition of fixed-priority scheduling, this changes the
mechanism causing work admission and can change the next implementation choice.
A negative result bounds this policy/workload port, not the entire paper thesis.

## Verified sources and fidelity boundary

Primary specification: [Hummingbird v2, §4.2–4.3, Algorithm 1 and Appendix A](https://arxiv.org/html/2601.04071v2),
[local v2 PDF](../../docs/reference/2026-hu-hummingbird-v2.pdf). The older local
[v1 PDF](../../docs/reference/2026-hu-hummingbird.pdf) is retained, not overwritten.

No public author scheduler implementation was confirmed. Searches covered the
exact title + `github`, title + `artifact source`, author + `code`, and
`Hummingbird preemption site:github.com`; then checked the correct PKU
[first-author page](https://tiancheng-htc.github.io/Tiancheng-Hu.github.io/),
[his public repositories](https://github.com/Tiancheng-htc?tab=repositories),
[coauthor's released systems](https://wangchenxi7.github.io/home/), and
[the lab organization](https://github.com/ICTPLSys).
[microGUST_SC26](https://github.com/Tiancheng-htc/microGUST_SC26) is empty;
[kernel_benchmark](https://github.com/Tiancheng-htc/kernel_benchmark) contains
operator examples, not a released Hummingbird scheduler. Neither establishes
an official runnable artifact. Microsoft Hummingbird is a different system.
No author contact or dependency clone was performed.

The permitted claim is **a paper-described scheduling component reimplemented
in C and BPF on the GPreempt DNN frontend**, not the author's original binary,
full-system reproduction or original hardware speedups. The first version
retains actual splitting, event-confirmed bubbles, kernel-tick pacing and
large-bubble consolidation. It excludes multi-GPU memory offloading and
automatic framework-pattern discovery. It implements Algorithm 1's
threshold-based consolidation, not the later N-BEATS prediction enhancement:
the paper does not supply its fitted model/training settings. Do not substitute
an arbitrary predictor and call it the original.

## Published algorithm → local implementation

| Published operation | Necessary local implementation |
| --- | --- |
| Compute an initial split size from SM count, occupancy and threads/block; decrease the size while execution time falls (§4.2). | Profile the real ResNet kernels on the target GPU, starting from occupancy-limited resident CTAs. Record measured candidate durations, selected size and launch overhead. C/BPF consume the same frozen profile. No hard-coded 400 µs promise borrowed from A100. |
| Preserve each original CTA's coordinates when splitting (§4.2). | Use existing exported `mod.cu` plus original `host.json` launch metadata. Generate offset-capable copies in this workload only; preserve x/y/z block coordinates and full block dimensions. Partition each original grid exactly once. Source-level transformation is a declared substitute for the paper's generic PTX transformer. |
| HP work pending means no further LP launch (Algorithm 1). | Maintain actual queued HP GPU work and completed CUDA events, not request rate or a sleep timer. Recheck HP readiness under the same submission lock immediately before issuing each LP launch. Already-issued LP work completes; no dropped or repeated CUDA execution. |
| Detect small bubbles from API patterns and confirm GPU progress (§4.3/Appendix A). | Instrument actual input/output copy + stream-sync boundaries of the DNN client with start/end CUDA events and pending-work state. A host API call or empty software queue alone is not proof that HP GPU work finished. Only separately profiled, noninterfering copy/sync patterns may admit LP work. |
| LP starts after outstanding HP compute completes; stop on HP enqueue or bubble end. | Explicit event confirmation in the HP context, then a shared admission state consumed by the LP worker in its own CUDA context. Never global-device-sync behind LP work or wait on a lock needed by the HP path. |
| Kernel-tick launch pacing, estimated duration minus launch cost (§4.3). | An asynchronous LP worker issues pieces at the profiled tick, with a bounded in-flight queue and recorded completion events. Prediction is not proof of completion: guard against underprediction rather than silently allowing an unbounded device backlog. Any event-fence cost or deviation from the paper's predicted pipeline is reported. |
| Large bubbles consolidate pieces to original grid and zero offsets. | Threshold is just above measured small-bubble lengths, frozen from calibration. Consolidate only a not-yet-started original kernel; finish any partially executed kernel's remaining pieces without replay. Recheck HP readiness before issuing the consolidated launch. |
| Multiple LP tasks use round robin. | First real case has one LP task, so this rule is vacuous; do not claim multi-LP fairness. |

The existing ResNet152 asset has 307 recorded launches across 206 host
functions, including 3-D grids. Its CUDA source is present (299,940 bytes);
a read-only scan finds no `gridDim`, atomic or cooperative-grid syntax.
This is useful implementation preparation, **not** a proof of safe splitting:
validate all exported functions, shared-memory/barrier behavior and full-model
outputs. Preserve unsplittable operations explicitly; a long unsupported
operation must be reported, not silently excluded from latency measurements.

## Existing interfaces and smallest new code surface

Prefer the GPreempt frontend because its model metadata and numerical checks
already cover real DNN requests. XSched's `ready` snapshot and Level-1
Pause/Resume provide a correct deferred-launch actuator, but expose neither
per-kernel durations/offsets nor API-bubble events. Its old 50-kernel burst has
no realistic inter-request idle intervals. Making only another HPF mask would
not implement Hummingbird.

Read-only reuse points:

- `../gpreempt/deps/upstream/include/executor.h`: virtual `load_model`,
  `execute`, `launch_kernel` and protected model permit a separate executor
  subclass without modifying the frozen library.
- `../gpreempt/deps/upstream/src/executor.cpp:120`: CUDA graph mode hides
  individual launches; explicit mode reaches `launch_kernel` at line 137,
  whose 11-argument driver launch is at line 158.
- `../gpreempt/deps/upstream/src/cuda-clients/gpreemptclient.cpp:196`:
  original early protection; lines 218–244 expose copy/infer/sync boundaries.
  Its `GP_RELEASE` is issued after graph submission, **before** GPU completion;
  it cannot be reused as a Hummingbird idle-completion signal.
- `../../extension/gpreempt_bridge.cpp:97`: existing ubpf JIT loading pattern;
  `../../extension/gpreempt_hint.bpf.c` currently knows only
  PREPROCESS/DUE/INFER, so it cannot express event-confirmed idle admission as-is.
- `../gpreempt/run_load_study.py`, `load_study_measurement.h` and
  `client_checks.h`: reuse leases, request timestamps, full-output validation
  and raw analysis boundaries, not frozen experiment configuration.

New files are confined here: `split_model.py` (restricted source transform),
`idle_policy.h`, `idle_policy.c`, `idle_policy.bpf.c` (same fixed-width decision
inputs and outputs), `idle_executor.cpp/.h`, `hummingbird_client.cpp`, focused
CPU tests and a Makefile with explicit `/usr/bin/g++-13`. Link read-only existing
`libexecutor.so`/DISB libraries where practical; all generated code/binaries
go under this workload's ignored `build/` or `deps/`. No vendored source edits.

The C and BPF policy receive identical HP pending/completion state, actual
bubble markers, clock, LP progress, next split/full duration and tick deadline.
They return stop/wait, launch-split or launch-whole plus the next wakeup.
GPU pointers and CUDA execution stay in the common executor. Validate bounded
state, timestamp arithmetic and decisions in CPU differential tests and count
actual JIT decisions in GPU runs. Do not call the C policy as BPF fallback.

## One matched experiment

Use the same seeded FP32 VGG19 foreground and ResNet152 background, batch one,
200 µs preprocessing, fixed model/input bytes and correctness reference.
Two arrival cases: periodic LC 100 requests/s and a fixed BurstGPT-derived LC
trace at mean 100 requests/s; BE is continuous in both. Use the official
[BurstGPT data and timestamp definition](https://github.com/HPMLL/BurstGPT/blob/main/README.md),
stable-sort successful request rows by timestamp, select the earliest 6,000,
and scale the first-to-last span to 59,990,000,000 ns, leaving the same 10 ms
deadline guard as periodic 100 requests/s. Reject a zero-span source segment.
The exact input is [`BurstGPT_1.csv` at source revision
`d895a53bb7b8ec137d0d2fe203b335835a78c10a`](https://raw.githubusercontent.com/HPMLL/BurstGPT/d895a53bb7b8ec137d0d2fe203b335835a78c10a/data/BurstGPT_1.csv),
50,853,373 bytes and 1,429,737 rows, of which 1,404,294 are successful.
[arrivals-burstgpt.json](arrivals-burstgpt.json) records the selected original
CSV line numbers, timestamps, response-token success field and all offsets.
The 6,000 selected arrivals contain 165 tied/zero intervals; median interval
4,285,510 ns and maximum 2,375,244,017 ns. These are fixed input properties,
not reasons to select cells, discard overload, or choose favorable gaps.
The 10-second preflight uses the first 10 seconds of this list, without a second
rescaling. Never infer
source chronology from an unsorted CSV prefix; sort successful rows first and
then take the earliest 6,000, preserving source order for timestamp ties.
Do not train/calibrate on the timed trace. This reuses request timing only,
not LLM tokens or Hummingbird's original model workload.

Five arms, five randomized paired 60-second blocks per arrival case:

1. Native CUDA priority scheduling: an unprotected throughput/reference arm.
2. Original fixed GPreempt policy: strongest available protection competitor.
3. C implementation of the paper-described idle/split scheduler.
4. BPF implementation of exactly the same idle/split scheduler.
5. Equal-timeslice control: the same two-context 1,000,000/1,000,000 us
   configuration and priority-zero streams as the idle arms, but unsplit,
   ungated real kernels and no GPreempt reservation/blocking. This is a causal
   control, not an additional main baseline: it tests whether resetting the
   BE timeslice alone explains apparent throughput recovery.

All new arms use the same explicit-kernel (non-graph) model execution setting;
only the two idle arms apply the identical splitter/pacing executor. Context
configuration is intentional and fixed: native retains one primary context
with observed highest/lowest CUDA stream priorities and no timeslice override;
fixed GPreempt retains two contexts, LC/BE timeslices 1,000,000/1 us,
host-mapped flags, blocking kernels and the original 100 us early hint. Both
idle arms use two contexts with **1,000,000/1,000,000 us** timeslices and
priority-zero explicit streams, no reservation/blocking kernels, and no flag
transport. Their completion/event-controlled admission permits only the
selected compute work to be submitted; they must not accidentally inherit
the original 1 us BE timeslice. Both use the same existing owned-context
`NvRmQuery`/`NvRmModifyTS` setup and check every return. The BPF port here is
the actual host-JIT idle decision, not a new kernel timeslice policy. Report
idle/fixed as a scheduling-package comparison. Also compare idle against the
equal-timeslice control on both BE goodput and LC protection: if the control
matches both, this experiment has not established benefit from split/idle
scheduling. If it recovers goodput but loses LC protection, report the measured
protection benefit and its cost. Turning graphs
off is a declared new-workload deviation: **never subtract old 45-cell results
to claim recovered throughput**. Compare the new matched cells. Before the
matrix, run isolated LC calibration on separate arrivals to freeze the SLO at
its response p99, following Hummingbird's exclusive-execution reference, and
the shared kernel/bubble profiles. The short preflight is excluded.

Calibration is separate from the 50 timed comparison cells:

- SLO: the same `native` frontend with a single role-0 VGG task, periodic 100/s,
  60 seconds, and all-offered response p99. Only this isolated calibration
  admits one task; the comparison cells require both roles.
- Kernel profile: occupancy-resident initial CTA capacity from the actual
  function/block shape; halve while the median per-request maximum piece
  duration improves by at least 1%, stopping at the first plateau or one CTA.
  Three complete, numerically checked model passes measure each candidate.
  A predeclared 12-halving bound is fatal if still improving, not silent success.
  Run another three passes with the final mixed capacity choices. Use the
  maximum measured selected-piece duration as the tick estimate and the median
  actual host submission-path duration as launch overhead; keep every sample.
  Copy profiling uses 100 isolated VGG input/output copy-and-sync samples.
  The large-bubble threshold is their maximum host API interval plus
  `max(1 us, 1%)`. These numerical calibration settings are local choices, not
  parameters recovered from an unreleased author artifact.
- Small-pattern interference: `idle_c`, periodic LC100 and continuous BE,
  four fixed profile variants `none`, `input`, `output`, `both`, five randomized
  paired 10-second blocks (20 cells, 200 seconds). Every other profile/context
  setting is identical. An enabled variant qualifies only with 100% LC
  completion coverage, nonzero actual LP launches in each enabled API pattern,
  and a paired LC-p99 ratio 95% upper bound <=1.01 against `none`. Use the same
  five-block/10,000-draw/seed-20260903 bootstrap. Enable both only if all relevant
  individual and combined checks pass; if both singles pass but their combination
  does not, choose input by this predeclared rule, not by observed BE speed.
  Otherwise choose the qualifying single or none. Retain failed/inconclusive
  variants. This tests incremental interference on this frontend; it is not
  the paper's automatic framework/API-pattern discovery. The profiler itself
  leaves both patterns disabled until these measurements qualify them.

If no small pattern qualifies, the resulting real large-interval-only experiment
must be labelled a partial component port, not successful reproduction of useful
small-bubble filling. C and BPF share the final frozen eligibility profile.

Primary: verified BE requests completed inside the common 60-second window,
and LC SLO attainment plus scheduled-arrival-to-verified-output response p99.
SLO attainment always divides by **all offered LC requests**; never-started,
unfinished and too-late requests are misses, not omitted samples. A conditional
LC p99 cannot substantiate retained foreground protection.
Preserve FIFO backlog, offered/started/finished accounting, late completions
and conditional-p99 labels as in the completed load study. Never discard
overloaded cells or count failed numerical outputs as goodput.

Compare idle versus fixed GPreempt only at retained LC protection. A win
requires the paired BE-goodput ratio's 95% lower bound >1, the paired LC-p99
ratio's upper bound <=1.01, and the SLO-attainment difference's lower bound
>=-1 percentage point, with complete LC coverage. Report paired effects and
confidence intervals even when these targets fail; more BE work with worse
protection is a tradeoff, and intervals crossing these criteria are inconclusive,
not a win. BPF/C compares
implementation overhead, separately from idle/fixed policy benefit. Five-block
paired geometric means and 10,000 whole-block percentile bootstrap draws use
seed 20260903; no statistical-equivalence claim or median CI substitution.

## Execution readiness and stop conditions

CPU build/check commands from the repository root (no GPU initialization):

```bash
HB_CPUSET=4-7 bash workloads/hummingbird/prepare_trace.sh
make -C workloads/hummingbird -j2 clients profile-client split-cubin test-cpu
workloads/hummingbird/build/hummingbird_client --help
workloads/hummingbird/build/hummingbird_profile --help
```

The source transformer, common C/actual-ubpf-JIT policy, asynchronous LP executor,
typed CUDA client, real-GPU profiler and SM120 split cubin now build successfully.
CPU checks cover six source-transform cases, 528 C/JIT semantic/parity cases,
1,000 randomized C++ exact-once XYZ partitions and 19 invalid synthetic profiles.
These are **not GPU correctness or performance results**. The original fixed
client is newly compiled directly from the frozen GPreempt client source, with
no policy rewrite. [prepare_trace.sh](prepare_trace.sh) copies and patches DISB
into this directory; both clients use its new header and archive. Link maps
confirm `BenchmarkTask`/`BenchmarkSuite` come from that archive; the frozen
executor library exports no competing DISB benchmark/client symbols.

Built-client contracts for the root-owned exclusive runner are below; paths
named `CONFIG.json`/`PROFILE.json`/`NEW_PROFILE.json` are runner-created inputs
or fresh outputs, not existing result claims. **These commands use the GPU**:

```bash
workloads/hummingbird/build/hummingbird_profile \
  --split-cubin workloads/hummingbird/build/resnet152-split/mod.cubin \
  --output NEW_PROFILE.json
workloads/hummingbird/build/hummingbird_client CONFIG.json --mode native
workloads/hummingbird/build/hummingbird_client CONFIG.json --mode timeslice_control
workloads/hummingbird/build/fixed_client CONFIG.json true --flag-transport host_mapped
workloads/hummingbird/build/hummingbird_client CONFIG.json --mode idle_c \
  --profile PROFILE.json --split-cubin workloads/hummingbird/build/resnet152-split/mod.cubin
workloads/hummingbird/build/hummingbird_client CONFIG.json --mode idle_bpf \
  --profile PROFILE.json --split-cubin workloads/hummingbird/build/resnet152-split/mod.cubin \
  --bpf-program workloads/hummingbird/build/idle_policy.bin
```

All clients require `GPREEMPT_POLICY=original` or unset; the idle BPF arm
loads its own actual host-JIT program, not the old driver policy. No XSched,
GDRCopy flag transport or blocking kernel is used by idle/control arms.
The idle LP worker is on CPU2, matching the original fixed daemon's CPU slot;
foreground/background client threads remain CPU0/1. Do not wrap real runs in
the build-only CPU4–7 affinity. On failure the accepted LP request is drained
or the cell fails within a 120-second worker limit; no false successful sync.
Cleanup stops the LP event reader before releasing HP events/contexts.

Planned raw output is `workloads/hummingbird/raw/idle-study-575-01/`;
50 timed cells take 50 minutes plus setup, calibration and cleanup. A root-owned
runner should reuse existing leases and invoke the real DISB client, not add
another experiment-control framework. All GPU runs remain serial/exclusive.

Preflight must demonstrate full-model split/unsplit numerical agreement,
nonzero real API/event bubbles, stop-on-HP behavior, bounded LP device backlog,
both split and consolidation execution, no replayed CTA, and real C/BPF decision
parity. If the existing model's copy/sync gaps cannot safely host useful pieces,
retain that result; do not relabel `usleep(200)` as an observed GPU bubble.
Profile/tick stability settings not provided by the paper must be explicitly
recorded before the first timed cell and shared across arms. Any changed
profile, source, correctness rule or policy requires fresh affected comparisons.
Root independent review admitted implementation with the trace deadline guard,
explicit context/timeslice setup, all-offered SLO denominator and paired-effect
decision criteria above, plus the equal-timeslice causal control (50 cells).
The predicted tick plus nonblocking completion fence
is an intentional stronger guard than the underspecified paper pipeline;
record event checks/waits and their cost instead of claiming its 1.3% overhead.

Positive: improved BE goodput while retaining LC protection, plus quantified
C/BPF cost. Contradictory: no recovered progress or overhead dominates useful
work. Mixed: BE improves by sacrificing LC protection. Inconclusive: inadequate
real bubble/split engagement or unsupported kernels invalidate the comparison.
Return results/limits to the root; leave the paper and canonical survey to it.
