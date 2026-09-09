# Revision completion checklist

## Experiment-only update — 2026-09-09 PDT

No manuscript files are changed by this update. Completed cells are retained
and must not be rerun just because an older status entry describes them as pending.

- The trampoline/work-scaling comparison below is complete, including actual
  invocation counts. Its total hook-plus-callback cost is not block-independent.
- The subsequent [warp-contiguous full-record comparison](../workloads/llama.cpp/observability_overhead/revision-rq4/results-full-record-soa-warp-20260909.md)
  completes five blocks / 15 clients. Median paired throughput gain over the
  previous SoA layout is 20.565%; paired loss versus uninstrumented is 12.862%.
  All event fields remain; the post-client drain is measured separately.
  This does not replace historical Table 1 or earlier layout measurements.
- End-to-end disk/UVM serving now completes [five blocks / 15 cells](../workloads/lmcache-disk/raw/diskuvm-physical-reclaim-20260909.pXYN4F/RESULTS.md)
  after driver 95097e20 adds physical GPU-chunk release. Each cell produces
  8192 tokens with no HTTP failure and server exit zero; original module/loaders
  are restored. Stock/native/BPF median throughput is 73.575/68.608/65.423
  token/s. Paired BPF/native difference is median -4.048%, adverse in 4/5 pairs.
  This is completed but negative performance evidence, not isolated BPF
  overhead or proof of universal OOM elimination. The earlier
  [per-range reclaim attempt](../workloads/lmcache-disk/raw/diskuvm-serving-reclaim-20260909.EFVUGf/RESULTS.md)
  and its two EngineCore OOM failures remain preserved.
- XSched Level-1 remains complete; Level-2 is still under repair. The
  [current native snapshots](../workloads/xsched/raw/level2-native-resolver-snapshot-20260909.KKFqHz/README.md)
  show that the first original-layout lookup succeeds, while a matched
  original-entry control still returns CUDA 701 with META_KPARAM=1.
  The earlier shortened-buffer control below did not enable META_KPARAM;
  it does not isolate the cause of this later configuration's failure.
  The [device-consumed-argument diagnostic](../workloads/xsched/raw/level2-xg5-runtime-20260909.jgLEyd/README.md)
  now builds and runs. Host publications advance across slots, but retained
  device entries keep reading one early slot through ordinal 200, including
  its later resume state. This narrows repair to per-launch context delivery
  or its argument ABI. Three BE workers fail; one is cleaned up after failure;
  both LC workers finish. It is not a successful performance comparison.
  The subsequent [inside-callback delivery repair](../workloads/xsched/raw/level2-callback-delivery-20260909.Y7qcwZ/README.md)
  builds but still leaves BE1 reading slot 1 through ordinal 200. All four
  BE workers fail output completion; both LC workers finish. The original
  tool is restored and GPU is idle. Placement alone does not fix delivery;
  there is no new Level-2 performance result.
  The later [per-launch enable repair](../workloads/xsched/raw/level2-perlaunch-enable-20260909.bOiYh5/README.md)
  now completes all six workers / 1200 kernels with no missing output.
  Enabling instrumentation after setting each launch value makes device
  context slots advance normally. The native C guardian port is runnable;
  this does not repair the separate original cuXtra route. A
  [five-block baseline/native/BPF comparison](../workloads/xsched/raw/level2-device-policy-pair-20260909.buBjns/README.md)
  is running, reusing that completed native sample rather than repeating it.
  Native/BPF arms include both host scheduling and device guardian decisions,
  with the same NVBit actuator and retained diagnostics. Completed Level-1
  results remain separate and unchanged.

Three local OpenCode sessions continue: one Qwen 27B result-analysis task and
two existing GLM XSched tasks. The driver and callback tasks completed
naturally before their sessions were reused. No session is stopped for lack of recent output;
actual provider retries are not treated as completed implementation work.

## Experimental follow-up — 2026-09-08 PDT

This update records experiment completion, not manuscript integration. This
session does not edit the manuscript. Earlier measurements and historical
status entries remain below; completed cells must not be repeated.

- **Disk/UVM restoration is measured, no longer code-only.** The
  [GPU-promotion report](../workloads/lmcache-disk/results-disk-uvm-gpu-promotion-20260908.md)
  contains five completed 256 MiB runs. First GPU restoration has median
  210.571 ms and repeated GPU read 0.251 ms. Compared with the earlier,
  separately measured CPU-first implementation, repeated reads are 21.58x
  faster but first restoration is 23.43% slower. This CPU-staged, explicit
  read-only disk-UVM primitive is not NVMe-to-GPU P2P, automatic pressure
  offload, or end-to-end LMCache KV integration. Existing LMCache serving
  and native/BPF storage-policy results remain separate.
- **Trampoline/work scaling and actual invocations are measured.** The
  [Native/Off extension](../microbench/fig15-device/strict-warp-map-scaling/raw/off-scalability-20260908.d8sa7gog/RESULTS.md)
  completes 240 timings and six separate count observations, through 8192
  CTAs, published in `33c4b9b1`. It measures the full trampoline plus callback
  and map update, with automatic warp execution off. At 8192 CTAs, overhead
  is 349.524%, 88.351%, and 3.985% for work 0, 128, and 1024. Invocation
  counts grow with CTA count. Neither constant total cost nor universally
  small overhead is supported. The separate
  [automatic-warp extension](../microbench/fig15-device/strict-warp-map-scaling/raw/larger-grid-20260908.bv03i7zm/RESULTS.md)
  retains its 180 timings and 12 count observations in `bf4c108c`.
- **Full-record GPU-local layout optimization is measured.** The
  [SoA comparison](../workloads/llama.cpp/observability_overhead/revision-rq4/results-full-record-soa-20260908.md)
  completes five blocks / 15 cells, published in `6fbf4c37`, retaining all
  23,068,672 records per tool cell. Paired SoA/AoS throughput gain is median
  19.229%; paired baseline losses are 27.621% / 39.385%. Readback is reported
  separately. This is a layout comparison with automatic warp execution
  off, not a replacement for the original Table 1 data or a claim of
  identical record semantics across historical campaigns.
- **XSched Level-2 host/device follow-up remains unfinished.** Existing
  Level-1 results are unchanged. The
  [reactivation diagnostic](../workloads/xsched/raw/level2-replay-diagnostic-20260908.d9Gydm/README.md)
  observes real reactivation and resume launches, but two background
  processes report missing output; the native SASS route still fails with
  CUDA 700. `b83a66d7` preserves the diagnostic records and corrects the
  earlier inference from a launch counter. These failed attempts are not
  performance comparisons. Local OpenCode sessions continue implementation;
  only failed cells need retry after a concrete repair.
  The later [ordinary-launch-value attempt](../workloads/xsched/raw/level2-ordinary-launch-value-20260908.Cp8bpS/README.md)
  still loses output (`ff9bff89`). On the native route, the
  [original-entry control](../workloads/xsched/raw/level2-native-original-entry-20260908.pn3UDn/README.md)
  completes all six workers / 1,200 kernels with the same preparation but
  without redirecting execution (`f6590e80`); this is not Level-2 success
  or a policy performance sample. The subsequent
  [resume-allocation padding attempt](../workloads/xsched/raw/level2-native-entry-alignment-20260908.jOzEYp/README.md)
  still fails with CUDA 700 (`bd38f2bc`). These narrow diagnostics leave
  redirected execution and restoration unfinished; no completed baseline
  or successful control needs repeating.
  The subsequent [direct at-launch preparation attempt](../workloads/xsched/raw/level2-direct-prepare-v2-20260908.cOtHcJ/README.md)
  is retained in `c6153b3b`: direct preparation before CUDA submission still
  leaves missing BE output; it is not a performance result. The post-run
  busy state and persistence-only recovery are recorded in `58e6cf60`.
  Native launch-error propagation is now preserved as an additive
  [source patch](../workloads/xsched/level2/native/xsched-native-launch-error-propagation.patch):
  it checks the launch result instead of waiting indefinitely for a device
  entry stamp after an already-rejected launch. Reverse-application checking
  confirms it is present in the isolated native source; that check is not
  a successful Level-2 runtime result. Parameter delivery remains in repair.
  The [context-scoped replay retry](../workloads/xsched/raw/level2-context-replay-20260908.CgSRR6/README.md)
  now builds and runs in `c8eeeb2f`: BE1/2/4 each have 201 preparations and
  201 filtered callbacks, repairing the earlier 201/200 discrepancy, but
  all three still report missing output. Both LC workers finish; BE3 is
  cleaned up after the failure. The next repair concerns guardian/resume
  execution, not another unchanged callback-count retry. No complete
  Level-2 performance result follows. The raw record includes the repeated
  post-run busy state and persistence-only recovery; GPU returned to idle.
  The native [metadata-extent attempt](../workloads/xsched/raw/level2-native-meta-extent-20260908.aY3rBY/README.md)
  first exposes omitted-source and empty-debug-body build problems, now
  repaired. The subsequent actual launch still fails; a debugger records
  initial CUDA 701 followed by teardown errors and a host exit-handler
  crash. Subsequent debugger observations show the metadata wrapper is
  called, but receives an unhandled CUDA fatbin wrapper and passes it through
  unchanged. That concrete format case is now in local-model repair.
  No native Level-2 performance sample is added by this attempt.
  The subsequent [wrapped-image retry](../workloads/xsched/raw/level2-native-wrapped-meta-20260908.NmlhJN/README.md)
  builds and actually changes both extent records before module loading,
  but the first launch still returns 701. The separate
  [resume-to-guardian / volatile-access retry](../workloads/xsched/raw/level2-resume-guardian-20260908.Rbs4dz/README.md)
  also builds and runs; both LC workers finish, while all four BE workers
  report missing output despite matched preparation/callback counts.
  Neither candidate closes Level-2 performance. Both leave GPU idle without
  a driver change or persistence restart; their source and adverse raw
  records remain separate from completed Level-1 results.
  A subsequent [original-entry control with the new wrapper and argument relay](../workloads/xsched/raw/level2-wrapped-relay-entry-control-20260909.kv7jHZ/README.md)
  still returns CUDA 701 and exits 139. Thus guardian entry redirection is
  not necessary for this launch rejection; the common parameter/resource
  setup remains under repair. This single-BE diagnostic adds no performance
  sample and does not repeat the earlier, pre-relay successful control.
  The subsequent [driver relay-size control](../workloads/xsched/raw/level2-relay-size-control-20260909.HX4CV3/README.md)
  keeps that metadata and original-entry setting, but shortens only the
  launch buffer's declared size from `0x1520` to `0x20`: all 200 kernels
  complete and the worker exits normally. This localizes the rejection to
  the large relay declaration. It is not a repair or performance sample:
  the short buffer omits the guardian argument block, and full accepted
  argument delivery remains unfinished.
- **Hummingbird host/device mapping five-block comparison is complete.**
  Existing host-policy and pipeline measurements remain complete and are
  not device-BPF evidence. The opt-in follow-up must consume BPF-computed
  block coordinates in the original workload. Its existing OpenCode session
  continues through Qwen 27B after Qwen Next's actual terminal HTTP 524,
  as recorded in `71ff359c` and the
  [local-model task record](experiment/original-host-device-local-prompts-20260908.md).
  The [integrated model record](../workloads/hummingbird/hostdev/component-build-20260908.md)
  now includes the real BPF cubin and a completed 60-second client run in
  `b15ade70`: 6000 foreground and 7718 background requests, exit zero.
  The initial ioctl failure was resolved by temporarily loading the saved
  compatible scheduler core; the prior driver/services/loaders were restored.
  The subsequent [20-cell paired comparison](../workloads/hummingbird/hostdev/results-performance-20260908.md)
  completes five blocks: original/host-BPF/callable-native/device-BPF BE
  goodput medians are 133.150/132.950/129.017/128.967 requests/s. Total
  BPF/original paired geometric-mean change is -3.052%; incremental device
  BPF/native-adapter is -0.072% with interval including zero. All earlier
  data and the adverse foreground block-0-excluded sensitivity remain.
  This is an opt-in rebuilt-cubin adapter comparison, not transparent
  unchanged-binary instrumentation or formal equivalence. The two XSched sessions
  plus this session respect the three-session limit; silence is not a reason
  to stop them.
  With this comparison finished, the same local session has moved to actual
  LMCache KV integration with the measured disk-UVM backing mechanism.
  The read-only backing contract applies to completed immutable chunks,
  not the mutable live KV pool wholesale. End-to-end integration and its
  performance remain unfinished; the unused Hummingbird launcher is WIP
  and is not a prerequisite for this work.

## Current manuscript snapshot — 2026-09-07 PDT

The current paper is `ccb5cf5`, compiled from `docs/paper/main.tex` with
flat section files under `docs/paper/tex-revision/`. The build has 15 pages.
Several entries below describe earlier 16/17-page drafts that were later
restored or simplified; their statements that material is “integrated” must
not be read as the current manuscript status. Their experiment records remain.

- Present: seven published-policy comparisons with workload/native/gpubpf
  controls; a seven-row policy table describing each original policy, its
  user-space/driver/device execution requirements and corresponding gpubpf
  implementation; transition-validation pseudocode, SIMT dataflow, rejection rules,
  fallback and TCB; RTX 5090 observability results including 5.57% kernel-return
  logging overhead; storage/CXL and state-freshness design paragraphs; an
  explicit unpartitioned GPU sharing setting; a short LMCache cuFile disk-policy
  paragraph in the KV-cache case, reporting FIFO/native/gpubpf read p99 and
  the 61.9% paired reduction.
- Still absent: the original LMCache LocalDiskBackend serving comparison
  (distinct from the cuFile policy paragraph); the additional Fig. 13 composition comparison;
  the SASS-only application discussion; architectural details of tenant
  isolation; explicit trampoline-scaling and startup-versus-runtime attachment
  explanations. Portability and future-accelerator discussion need review.
- Available outside the paper: completed original LMCache local-disk and
  cuFile-backed serving comparisons, the five-block write-budget study,
  the later I/O-concurrency study, bounded SASS EXIT injection, and the fresh
  four-configuration memory/scheduling experiment. Measurement completion
  does not imply that these results are in the current paper.
- Still unavailable in the checked release locations: the original agent
  study transcripts. Public harnesses and newly authored reproduction templates
  do not complete the original-prompt release.

The submitted Aug 3 commitments and Aug 14 shepherd follow-up in
[revision-comments.md](paper/asplos-27-rebuttal/revision-comments.md), together
with Q1–Q15 of the submitted response, define the requirements. Preserve the
original paper files, figures and locked conclusion when adding missing text.
The older progress entries below are retained as history.


Updated 2026-09-07 UTC. The user requests completion of the remaining items,
not only a status audit. The [complete review archive](paper/asplos-27-rebuttal/README.md)
define the scope. The dated plan in the paper repository is not evidence that
these commitments have been met. Experimental completion, paper integration,
and public artifact availability are separate checks.

**Latest LMCache improvement:** [25-cell matched write-budget study](../workloads/lmcache-disk/results-575-gds-write-budget-20260907.md)
is complete and pushed (`a451db6a`). BPF 200/10 ms paired read p99 improves
51.011% and write throughput 15.315% at the median, both in all five pairs;
native improves 53.999% and 12.698%. This establishes a shared policy-budget
benefit on this workload, not BPF-only acceleration or tight native/BPF
latency equivalence. A scoped storage-tier paragraph is now in the revision
draft, with a successful 16-page build. The diagnostic and all historical
unfavorable cells remain. No additional LMCache repeat is queued.

**Fresh Fig. 13 result:** [five-block four-arm study](../workloads/fig13-fast/results-performance-575-20260907.md)
completes all 20 cells and driver/service/GDS-loader restoration (`10994d21`).
High/low process-duration medians are 56.548/56.595 s baseline,
25.160/26.852 s memory-only, 3.938/6.076 s scheduling-only and
3.546/6.455 s combined. Compared with scheduling alone, combination improves
high priority by paired median 10.166% but slows low priority by 6.299%.
Paper `6b34963` integrates this measured tradeoff and qualifies the old sub-1%
scheduling differences. Paper `39ceb61` now adds the four-panel figure from
main `fbe9cac1`, preserving all three historical panels; main `8a6c51dc`
publishes the reusable analyzer and matching reanalysis. The root inspected
compiled page 12. The build has 17 pages including references, with the
conclusion on page 14. The GPU matrix must not repeat. All historical data remain.

**Latest SASS implementation:** the [PTX-free application run](../workloads/sass-kretprobe/results/sass-exit-575-20260907-01/results.md)
is complete in main `9d823ede`, with exporter `08e1692e`. Local OpenCode
Qwen 27B implemented NVBit EXIT injection/embedding; GLM implemented the BPF
device-function exporter. Root's actual cubin-only vector-add run records
two EXIT sites, all 100,352 thread slots written by compiled BPF, expected
application output and return zero. This closes the bounded in-body path,
not general helpers/maps, late attach or performance parity. The revision's
implementation paragraph includes the result and builds in 17 pages.
Separately, paper `da7c547` adds FineMoE's faster
demand-only baseline (5.17 versus native/BPF 4.50/4.51 token/s) alongside the
retained all-positive improvement; no completed FineMoE cells repeat.
The [SASS evidence index](experiment/revision-sass-aot-readiness-20260904.md)
also now includes the previously omitted `8e4e64d` first-party interposition
study: five runs, steady total 5.1375/32.1015 us uninstrumented/interposed.
The root recomputed these existing raw summaries without a GPU repeat. This
separate-kernel path is not in-body SASS injection or Table 1 evidence.

**Latest execution — 2026-09-07 PDT:** the [stale-state repeated campaign](../workloads/stale-state-575/results-performance-gds-20260907.md)
completes all 21 cells and restoration of the saved GDS module (`bc0ff88a`).
Fresh native/BPF throughput medians are 272832.196/271397.413 checked words/s;
1000 ms delayed state reduces paired throughput by median 23.192%/22.684%.
Phase-aligned analysis is complete (`1ea66808`): 15,747,386 decisions across
18 policy cells. With 1000 ms delay, native/BPF wrong-phase fractions have
medians 88.708%/88.854% (decision-weighted). All driver thrashing counters
remain zero; this is delayed-state performance sensitivity, not observed
driver-classified thrashing or a tested adaptive mitigation. Paper `57e9937`
integrates these limits and builds in 16 pages. No completed GPU cells repeat.
The [LMCache admission diagnostic](../workloads/lmcache-disk/results-575-gds-admission-timing-20260907.md)
also completes 15 cells (`37299d27`): native/BPF demand-read admission medians
are 11.973/30.140 us, versus hundreds of milliseconds end-to-end. The subsequent
LMCache optimization already completed the longer live write-protection budget
study linked above, on both native and BPF. Historical pending statements below are superseded; all
earlier numbers remain.

**Latest event-driven result — 2026-09-07 UTC:** the new LMCache executor
and [five-block measurement](../workloads/lmcache-disk/results-575-gds-live-event-driven-20260907.md)
are complete: 15 cells, 960 reads and 1,440 writes, source `c2ecedcb`.
FIFO/native/BPF read-p99 medians are 279.361/431.303/255.874 ms. Paired
BPF/FIFO p99 change has median -8.407% (four of five improve), and BPF/native
has median -9.135%, range -86.876% to +123.960%. Native/BPF decisions fall
to 306--350 per cell, but performance still varies substantially; this is
not stable superiority or a tight overhead bound. All old polling/GIL data
remain. Statements below that event-driven implementation is pending are
historical. The separate GDS-compatible stale-state driver (`a2b40efd`) and
direct performance entry (`e88e1265`) have now completed the 21-cell campaign
and saved-module restoration linked above.

**Latest LMCache update — 2026-09-07 UTC:** the live-feedback provider,
executor, runner and [five-block campaign](../workloads/lmcache-disk/results-575-gds-mixed-live-feedback-20260907.md)
are complete and pushed (`674bf3d2`): 15 measurements, 960 reads and 1,440
writes. FIFO/native/BPF scheduled read-p99 medians are
**1070.424 / 473.147 / 806.274 ms**. BPF/native paired p99 increases in all
five blocks, with median **+23.601%**; this is not a low-overhead matching
result. Each native/BPF cell releases 95/96 writes only after the 10 ms budget
expires. Event-driven wakeup is now the active optimization, not another run
of the completed polling implementation. Historical unfinished-live-feedback
statements below are superseded by this update; their earlier results remain.
The [workspace collection record](workspace-cleanup-20260907.md) tracks raw
data publication and recoverable cleanup separately from experimental progress.

The separate [GIL handoff ablation](../workloads/lmcache-disk/results-575-gds-gil-handoff-20260907.md)
now completes 20 measurements / 3,200 requests in five rotated four-arm blocks,
published in `0d14fa1a`. Keeping the GIL across the same BPF ioctl has paired
read-p99 changes versus ordinary BPF of +4.923%, -28.106%, +513.574%, -50.739%,
and -42.630%; relative to native its median is +7.477%. This is not a reliable
optimization, so the constructor option remains default-off. An initial
relative-cache-path failure is retained separately; no measured cell was
discarded. The event-driven executor remains unfinished. The matched-port
figure's grouped-bar layout is separately corrected at final 7-inch width;
paper `f990f04` builds in 16 pages, with no new performance measurements.

**2026-09-06 user direction and current result:** LMCache local disk is active,
and Table 1 is evaluated only as llama.cpp pp512 prefill-throughput overhead.
The complete RTX 5090 campaign contains 10 rotated blocks and 70 successful
numeric cells. Baseline is 37,586.3225 token/s; gpubpf/NVBit overhead is
90.7051%/99.6210% for `kernelretsnoop`, 2.9653%/10.3501% for `threadhist`, and
0.2208%/8.7959% for `launchlate`. The submitted P40 values remain retained.
The separate [GPU-local array follow-up](../workloads/llama.cpp/observability_overhead/revision-rq4/results-onevalue-array-bootstrap-575-20260907/README.md)
now completes five paired blocks: mean baseline/tool throughput is
37979.2561/35861.5351 token/s and mean paired overhead is **5.5726%**
(range 4.0175%–6.7050%). All ten benchmarks and five collectors exit zero;
every tool run retains 720896 full records. Final bulk lookup averages
10.379 ms, separately reported and excluded from prefill timing. This is a
finite pp512 buffer, not an unbounded streaming result; old numbers are retained.
Paper `812cf4a` now includes this completed GPU-local buffering result in
`tex-revision/tex/eval.tex`, alongside the unchanged older observability
numbers. The text states the final readback cost, finite serial-launch scope,
and warning-mode limitation. Two LaTeX passes complete at 16 pages with no
undefined references or citations. This closes that paper-evidence gap
without rerunning any device-tool measurement.

**2026-09-06 GDS end-to-end update:** the [five-block raw summary](../workloads/lmcache-disk/raw/gds-five-arm-575-20260906-five-block-formal/summary.json)
now completes 25/25 LMCache cells. Recompute / CPU / GDS FIFO / native / BPF
median output throughput is **31.0098 / 29.7637 / 37.4099 / 37.4703 / 38.3743
token/s**; median TTFT is **66.8884 / 74.4118 / 78.0768 / 78.4502 / 77.4424 ms**.
The same-block BPF/native throughput change has median **+0.2795%** and range
**-3.4230% to +5.1679%**. Default telemetry yields immediate submission, so this
is a mechanism-floor comparison using real cuFile compatibility-mode storage;
it does not establish dynamic storage-policy gains or hardware NVMe/GPU P2P.
The separate [policy-input ablation](../workloads/lmcache-disk/results-575-gds-policy-input-ablation-20260906.md)
also completes **25/25 cells and 200 warm requests**. FIFO / BPF immediate /
native defer / native full inputs / BPF full inputs median throughput is
**38.2426 / 36.6713 / 37.2024 / 37.8628 / 36.7143 token/s**. Paired throughput
changes are **-0.1853%** median for BPF full/native full and **-3.1112%** for
BPF immediate/FIFO. Both favorable and adverse pairs remain published.
The sequence separates cold writes from warm reads and does not demonstrate
read/write contention benefits. The [mixed-storage comparison](../workloads/lmcache-disk/results-575-gds-mixed-fresh-process-20260906.md)
now completes **15/15 fresh processes**, with **960 reads and 1,440 writes**.
FIFO/native/BPF read call-to-completion p99 medians are
**225.177 / 208.939 / 218.392 ms**. Native and BPF each defer all 96 background
writes per measurement, but paired differences vary substantially and do not
show a stable benefit. This is backend traffic, not output token/s; the raw
offer field is actual dispatch. Per-process isolation avoids the preceding
seven-measurement retained-pool OOM, whose records remain retained.
Earlier failed attempts and the
one-block pilot are retained separately; none is pooled into these five blocks.

**2026-09-07 UTC scheduled-arrival update:** the runner now isolates every
measurement in a fresh subprocess and records scheduled offers separately
from dispatch. Its [complete five-rotation comparison](../workloads/lmcache-disk/results-575-gds-mixed-scheduled-20260907.md)
adds 15 successful measurements, 960 reads and 1,440 writes. FIFO/native/BPF
scheduled-arrival read p99 medians are **298.220 / 265.297 / 247.931 ms**.
BPF/FIFO paired p99 changes have median **-12.440%**, improving in all five
pairs, while paired write throughput has median **-1.673%** and read p50
medians worsen. BPF/native paired p99 has median **+0.538%**, range
**-15.884% to +6.322%**; this is not a tight mechanism-overhead bound.
This closes the runner's process/timing implementation follow-up and measures
the fixed-delay storage-policy tradeoff. Live pending-demand feedback remains
distinct unfinished policy work. Its matching native/BPF flagged-write branch
is implemented and built in main commit `f105c6fd`; live counter/executor and
runner wiring remain unfinished. The new object is now attached on the same
575 driver; no live-feedback performance result is claimed yet.
Prior dispatch-only and burst data remain.
Paper commit `277c77f` now integrates this policy-behavior comparison into
`tex-revision/tex/eval.tex`, including the adverse p50/write-throughput results,
the BPF/native range, and the storage-request versus application-TTFT distinction.
Two `pdflatex` passes complete successfully at **17 pages**; no new experiment
or figure was introduced for this integration.

The separate [final-only collector comparison](../workloads/llama.cpp/observability_overhead/revision-rq4/results-final-only-575-20260907/README.md)
completes five baseline/tool pairs with **3493.318 token/s / 90.812% overhead**
and all 720896 events per tool run retained. It does not resolve kernelretsnoop's
high overhead. The GPU-local producer-array follow-up above is now measured
and lowers prefill overhead to 5.573%; its collection cost remains explicit.
The existing three-tool Table 1 and every earlier result are unchanged.

The [end-to-end report](../workloads/lmcache-disk/results-575-lmcache-gds-five-arm-20260906.md)
and raw records are pushed. Paper commit `c254a98` adds the measured storage
comparison only under `tex-revision`; its current build succeeds with no
undefined references, at **17 pages**. The older 16-page build records below
remain historical; final page-budget fitting is still outstanding.

Current integration update: the active paper source includes safety
pseudocode/algorithm/examples/TCB, the capability table, completed policy
comparisons and four-panel scheduling figure, discussion additions and
typography repairs. It also now includes Expert Buffering, the POD phase
decomposition, bounded raw-map evidence, and the operation-matched RTX 5090
device-map placement result. The abstract/introduction
distinguish policy benefits from measured mechanism costs. The latest source is
pushed as paper commit `5941a65`. A fresh build at that revision succeeds in
16 pages with no undefined references. The current detailed
[build review](paper/asplos-27-rebuttal/revision-build-review.md), recorded at
paper commit `e22af4c`, reports a successful 16-page build with conclusion and
references beginning on page 14 and no undefined references/citations. Review archive commit
`46d2f70` adds the missing
review updates and meta-review. The standalone native/BPF LMCache-storage
executor and the end-to-end LMCache five-arm comparison now have completed
five-block campaigns. Live-input policy benefits, original agent-log release,
and final submission readiness remain open. See
[safety evidence and gaps](revision-safety-design.md) and
[remaining artifact execution](revision-remaining-artifacts.md).

LMCache is not paused. Its retained earlier failures remain historical evidence,
while a new five-block performance campaign and native/BPF storage-policy work
continue independently.
Two requested read-only OpenCode subagents have returned; their full reports
and main-thread corrections are [recorded here](experiment/revision-opencode-review-20260903.md).
The independent [status-sync review](experiment/revision-status-sync-opencode-review-20260904.md)
then rechecked the completed comparison values and open boundaries and returned
`PASS`.

## Current evidence and remaining work

| Commitment | Evidence available | Still required |
| --- | --- | --- |
| MoE-Infinity | **Complete scoped algorithm port:** baseline -> paper-native -> paper-BPF throughput is **11.8964 -> 11.2233 -> 11.1900 token/s** over five blocks / 15 accepted cells. The [full report](../workloads/moe-infinity/results-paper-v3-protected-575.md) records BPF/native 0.996540 [0.989239, 1.005508]. | This is not original-system reproduction or formal equivalence. The baseline alone has a prefill overload shortcut, so baseline/policy is not a pure policy contrast. |
| XSched | **Complete scoped Level-1 port:** native CUDA -> original XSched -> BPF HPF is **76.8780 -> 26.9784 -> 27.2502 s** LC p99 and **10.2377 -> 10.1497 -> 10.1616 kernels/s** BE throughput. See the [46-cell report](../workloads/xsched/performance-full-575-20260903.md). | BPF makes the bounded HPF decision and original XSched executes suspend/resume. This is Level-1, not Level-3, and the LC interval does not establish a difference or equivalence. |
| GPreempt policy comparison | **Complete scoped compatibility port.** At continuous BE supply, native -> original C -> BPF is **1.795937 -> 1.614817 -> 1.610008 ms** LC response p99 and **197.717 -> 179.967 -> 180.100 req/s** BE goodput in the [45-cell load study](../workloads/gpreempt/results-load-study-575-20260903.md). The separate [27-cell LC-knee sweep](../workloads/gpreempt/results-lc-knee-575-20260903.md) brackets full coverage at 625 requests/s and conditional overload at 800. | Both ports protect foreground latency by sacrificing background work. This is host-mapped compatibility, not original GDRCopy/hardware reproduction, formal BPF/C equivalence, or an all-offered p99 claim at 800 requests/s. |
| LMCache local-disk backend | The five-block performance-only campaign completes 15/15 cells: recompute/CPU/disk output throughput is **30.6422/30.1168/28.5723 token/s**, request throughput is **1.9151/1.8823/1.7858 req/s**, and median TTFT is **67.1691/72.6468/96.3280 ms**. Earlier correctness failures remain retained separately. The [async GDS control design](../workloads/lmcache-disk/gds-async-policy-plan.md) fixes the decision boundary at one logical KV chunk. The [RTX 5090 transport reference](../workloads/lmcache-disk/results-575-gds-transport-smoke-20260906.md) measures 24 MiB async-stream read/write at **1.934/3.975 GiB/s** and batch-stream read/write at **1.917/4.090 GiB/s**. The subsequent [five-block storage-policy campaign](../workloads/lmcache-disk/results-575-gds-async-policy-20260906.md) runs FIFO, matched native and live BPF control through one trusted cuFile executor. Native and BPF make the same 40 submit / 16 defer / 8 recompute decisions per 64 requests; BPF decision cost is **1.005 us/request** versus **0.063 us** natively. | This host uses cuFile compatibility mode, so the result is real asynchronous cuFile/O_DIRECT policy control, not direct NVMe-to-GPU P2P. Storage-service variance is large and the cell does not establish a native/BPF throughput difference. The end-to-end LMCache five-arm comparison now completes 25/25 cells; see the dated update above for performance, scope and the ongoing policy-input ablation. |
| Expert Buffering hot residency | **Complete matched-policy study.** [Preflight 02](../workloads/expert-buffering-policy/section-vi/correctness-results-575-02.md) exactly rechecks 65,636,352 values and 20,182 BPF/native-shadow decisions. The [replacement performance campaign](../workloads/expert-buffering-policy/section-vi/results-performance-575-20260903.md) completes all 15 cells / five randomized blocks: native and BPF improve same-K FIFO throughput by 2.55% [2.10%, 3.09%] and 1.79% [0.79%, 2.81%], respectively; BPF is 0.74% [0.20%, 1.40%] slower than the identical native algorithm. Native/BPF outcomes match, and the policy reduces logical copy payload 12.85%. [Attempt 01](../workloads/expert-buffering-policy/section-vi/full-01-abandoned.md) remains wholly excluded and retained. | Integrate this honest policy-benefit/mechanism-cost result into the paper if space permits. It is a single-GPU same-executor policy port, not the original distributed system or an equivalence/zero-overhead result. The older page-profile analogue remains separate. |
| Transition-validation pseudocode, SIMT algorithm, rejected policies, failure taxonomy, TCB | Code-grounded exposition is in the draft. [Two actual strict counter pairs pass](../workloads/bpftime-device-smoke/results-strict-575-20260903.md), independently audited: 32,768 callbacks per positive, explicit rejection and zero counters per negative. The aggregate CPU-only runner at bpftime commit `aae1f22` passes all six unsafe/control pairs: PREVAIL memory bounds and termination, plus SIMT branch uniformity, shared-map side effects, atomic-target uniformity and the registered global-synchronization helper. Its own summary explicitly marks the host Linux-verifier and driver transition-validator layers as two external `NOT_RUN` cases. The [replacement invalid-prefetch campaign](experiment/revision-safety/prefetch-invalid-575-02/result-review.md) completes native, legal BYPASS, and invalid-action-99 controls on the RTX 5090: 42,053 / 131,072 / 41,882 closed decisions, zero observer errors and data mismatches, exact native fallback for every invalid request, and exact old-UVM/service restoration. The [scheduler-init live matrix](experiment/revision-safety/sched-init-live-575-05/result.md) now passes all 16 cells across two randomized blocks and eight native/BPF transition rows after a zero-mismatch 32,768-value native preflight. Every event-join, ownership, cleanup, monitoring and safety gate passed; the known-good 575.57.08 stack and both services were restored with no recovery/finalization error. The [actual userspace loader study](experiment/revision-safety/loader-failure-cpu-575-01/results.md) adds 15/15 CPU `LD_PRELOAD`/`BPF_PROG_LOAD` cells: explicit `STRICT` rejects an out-of-stack program without consuming a program ID, while `WARNING`, `NO_VERIFY`, and the unset default admit it. Four earlier fail-closed attempts remain retained. | These are scoped functional transition and loader tests, not performance comparisons or proof of universal no-op semantics; operation-specific fallback boundaries remain. The six-pair aggregate does not itself execute either external `NOT_RUN` layer. The default userspace verifier mode is warning-only, so deployment and paper safety claims must require and report explicit strict enforcement. CPU/compile-only fixtures and separate strict controls do not retroactively verify performance runs. |
| Three-way expressibility table | The [audited inventory](experiment/policy/reference/RELATED_POLICY_EXPRESSIBILITY.md) covers **52 papers in seven policy families** and links the strongest local evidence for every measured port. The draft separates user space, modified driver and current gpubpf capabilities, with actuator/trust boundaries. The active [evaluation source](paper/tex/eval.tex) uses the corrected sequential-prefetch **573 LOC** and composite **1334 LOC** values. | No surveyed whole system is classified `FULL`; local `performance` evidence does not erase missing semantics or actuators. The current 16-page build passes; repeat it after later source changes. |
| Expand Fig. 13 and distinguish policy from mechanism | Matched-policy subsection and revised attribution are integrated. Historical Fig. 13 retains its six source CSVs and single-round/engagement limits. The [fresh independently timed comparison](../workloads/fig13-fast/results-performance-575-20260907.md) completes five blocks / 20 cells (`10994d21`), with scheduling hits/modifications in all ten relevant cells. Paper `39ceb61` includes the expanded four-panel figure (`fbe9cac1`) and the measured priority tradeoff; reusable analysis is published in `8a6c51dc`. | Figure integration is complete; no additional GPU repeat is needed. These are policy-composition timings, not native/BPF same-policy mechanism-overhead measurements or universal scheduling superiority. |
| RTX 5090 Table 1 | The current [three-tool result](../workloads/llama.cpp/observability_overhead/revision-rq4/results-table1-warp-plt-575-06/README.md) completes **10 rotated blocks / 70 cells**, all with numeric throughput and return code 0. Baseline is **37,586.3225 token/s**. gpubpf/NVBit overhead is **90.7051%/99.6210%** for `kernelretsnoop`, **2.9653%/10.3501%** for `threadhist`, and **0.2208%/8.7959%** for `launchlate`. All earlier Table 1 measurements, including verifier A1/S0 studies and the submitted P40 values **8%/85%, 3%/87%, 14%/93%**, remain retained as separate historical evidence. | The requested three rows are measured. The separate GPU-local producer follow-up completes five paired blocks with **5.5726%** mean prefill overhead and **10.379 ms** mean final lookup outside prefill timing. It retains every record for the finite pp512 workload; no earlier measurement is overwritten. |
| Agent prompts and benchmark harnesses | Public harnesses, the missing-session inventory and separately labelled newly authored reproduction templates are committed and pushed (`1e4564c`). | Recover and redact actual original transcripts before claiming the original-prompt release complete. The author has been asked for the backup. |
| Discussion and organization | Draft groups stale-state thrashing, CXL tiers, tenant scope, trampoline scaling, portability and software co-location versus static partitioning. The [stale-state repeated campaign](../workloads/stale-state-575/results-performance-gds-20260907.md) completes 21 cells, saved-GDS-module restoration and analysis of 15,747,386 decisions (`bc0ff88a`, `1ea66808`). Paper `57e9937` reports the measured sensitivity and builds in 16 pages. The [earlier deployment audit](experiment/revision-deployment-575/RESULTS.md) covers CPU-only LD_PRELOAD/Frida lifecycle, not GPU performance. The SASS AOT pipeline at bpftime `fd976ea` loads and executes a standalone generated cubin returning 42 on RTX 5090. | Delayed-state performance sensitivity is measured, but adaptive freshness mitigation and driver-classified thrashing are not demonstrated. The earlier [owner-12 record](../workloads/stale-state-575/live-formal-campaign-20260905.md) remains separate, not repeated or rewritten. Standalone SASS execution is not injection into existing PTX-free applications, application hook/helper/map semantics, or a full NVBit comparison. Proposed CXL, tenant and portability work remains distinct from implemented guarantees. |
| PTX-free application instrumentation | The newer [live SASS EXIT result](../workloads/sass-kretprobe/results/sass-exit-575-20260907-01/results.md), main `9d823ede`, executes compiled BPF inside the existing target: two EXIT sites, 100,352/100,352 BPF outputs and expected application output. Paper `e01549f` adds the implementation paragraph; the older standalone-only boundary in the row above is superseded for this bounded path. | One context, one target kernel and finite per-thread output slots. Helpers/maps, late attach, concurrent launches and performance parity remain unestablished. NVBit supplies the trusted binary instrumentation. |
| Typographic fixes | Active-source double punctuation and printed bibliography braces repaired and checked in a fresh build. | Existing bibliography metadata warnings are not a completed citation audit. |

## Additional requested experiments

| Experiment | Status and next action |
| --- | --- |
| FineMoE dynamic prefetch | **Measurements and planned plot complete:** 20 cells, five blocks. [Report](../workloads/finemoe/results-performance.md) retains reduced unused transfers versus all-positive, a throughput loss versus demand-only, and unresolved BPF/C difference. The [two-panel plot](../workloads/finemoe/figures/dynamic-set-comparison.pdf) and exact 20-row CSV reconstruct every point from saved worker records and check against the independent analysis; three focused plot tests and eight existing scheduling-plot tests pass. The 7-inch vector PDF was rendered and inspected. No new GPU measurement occurred; this new plot is a workload artifact, not yet an additional paper float. |
| Hummingbird idle scheduling | **Original 50-cell conservative-port study and separate 40-cell fixed-bound ablation complete.** The [new full report](../workloads/hummingbird/pipeline/results-575-20260903.md) reaudits all five blocks per arrival: 240,000 LC requests, zero numerical error, 106,089,548 events retired and 3,839,761,392 actual JIT decisions. Bound 2 raises C/BPF BE goodput about 15%, but unchanged LC protection is not established; BurstGPT SLO attainment falls 0.44/0.56 pp. BPF/C has a small −0.242% BE effect at BurstGPT bound 1; other BE intervals include zero change. No equivalence or full-system reproduction is claimed. The earlier 19–20% gap versus fixed GPreempt remains a separate comparison, not closed by pooling campaigns. |
| POD-Attention device task choice | **Complete:** five blocks, ten shapes, 250 operator cells / 25 arm processes; [report](../workloads/pod-attention/results-575-20260903.md) and [independent audit](../workloads/pod-attention/raw-audit.md) reconcile every cell. Nine shapes show 0.51–1.18% BPF/CUDA latency cost; substantial whole-process cost and FP32 characterization exceedances remain explicit. Strict verifier enforcement/full-system reproduction are not claimed. Earlier failures remain retained. |
| POD-Attention setup decomposition | **Complete:** the separate [formal phase campaign](../workloads/pod-attention/results-phase-full-575-01.md) has five randomized blocks and all 15 fresh-process cells. Relative to the original CUDA selector, BPF adds 1.78% [1.64%, 1.92%] CUDA-event operator time and 1.81% [1.66%, 1.96%] host-wall operator time. The larger 2.372x measurement-plus-audit loop ratio includes correctness/decision auditing and is not operator overhead. The current fresh-process BPF path also spends a median 271.225 s before the first Python statement versus 21 ms for the CUDA adapter; this bundles preload/agent/PTX/JIT work and is not labelled generic attach latency. The [three-panel plot](../workloads/pod-attention/figures/phase-full-575-01.pdf), script, caption and five tests are retained. |
| Cross-layer non-composable map state | **Complete:** the [five-block, 15-cell formal campaign](../workloads/cross-layer-raw-map/results-full-575-02.md) exactly re-parses every native, instrumented and probe log. Ten positive cells recover all 34,560 bounded coordinate/sequence tuples; five deliberate overflow cells report all 2,560 drops and reject the incomplete streams. Each cell passes lifecycle and safety cleanup. | This directly shows current-ABI raw host readback beyond reducible aggregates. It is not a latency/bandwidth result, on-chip shard test, automatic-placement result, strict-verifier run, or arbitrary/unbounded-data claim. The retained earlier failed campaign is excluded rather than selectively completed. |
| Device trampoline scaling | **Both scoped executions are complete, but the fixed-work hypothesis is inconclusive.** The repaired [original preflight](../microbench/trampoline-scaling/raw/preflight-575-04/summary.md) and [ten-block campaign](../microbench/trampoline-scaling/results-575-20260903.md) validate 30 arms / 270 measurements. The stricter [fixed-work follow-up](../microbench/trampoline-scaling/raw/fixed-work-full-575-01/fixed-work-analysis.md) independently replays all **30 arms / 150 timings** as valid while holding total threads and dynamic warps fixed across five organizations. Its endpoint estimate is -2.7049% with 95% CI [-5.4735%, 3.1331%], and both the endpoint and Bonferroni all-five guards are inconclusive against the predeclared +/-1% bound. Absolute no-op increments span **0.272--1.840 us**; counter increments span **558.8--587.7 us**. | These are synthetic, verification-disabled measurements. The fixed-work data do **not** establish block-organization or block-count independence, warp-leader execution, once-per-warp dispatch, or constant cost for arbitrary handlers; audited code uses ordinary per-thread `call`/`call.uni`. |
| STRICT warp-key map-shard scaling | **Complete and contradicted; repository evidence only.** The [full report](../microbench/fig15-device/strict-warp-map-scaling/results-full-575-01-20260904.md) and [independent audit](../microbench/fig15-device/strict-warp-map-scaling/independent-review.md) accept **160/160** fresh processes with zero nonzero return codes, **120/120** target-PID STRICT admissions, 160 correctness markers, and 120 detach markers; noop readbacks have count 0, shared readbacks count 1, and warp readbacks satisfy `value = magic XOR key` with final distinct key counts 4/4/8/16/32 by shape. Cross-shape factors from 1 to 32 warps per block are **1.0102** [0.9768, 1.0390] shared/noop, **1.0060** [0.9915, 1.0604] warp/noop and **1.0062** [0.9878, 1.0257] warp/shared, so warp-uniform keys show **no detected** cross-shape scaling advantage, with all three cross-shape intervals containing one and the only per-shape interval excluding one in the predicted direction being the warp/shared contrast at one warp, 0.9936 [0.9848, 0.9971]. A valid execution-only [preflight](../microbench/fig15-device/strict-warp-map-scaling/results-preflight-575-06-20260904.md) passes 20/20 processes and 15/15 admissions with no effect row; five earlier preflight attempts remain retained, and [attempt 05](../microbench/fig15-device/strict-warp-map-scaling/results-preflight-575-05-failed-20260904.md) aborted before any BPF load on syscall-server CUDA error 3. Bootstrap seeds are now deterministic and 18 tests plus both campaign replays pass. | This is a negative result and must stay repository evidence, not a paper-positive claim. It is single-block, scalar per-thread `call.uni`, and the helper reads physical PTX `%warpid`, so the observed key counts are hardware warp slots, not logical CTA warp IDs. It establishes no warp aggregation, once-per-warp dispatch, callback cardinality, multi-block or grid scaling, contention decomposition, or application performance, and must not be pooled with the per-lane or strict-uniform placement campaigns. |
| Device-map placement | **Complete and integrated in two scoped modes.** The per-lane [formal report](../microbench/fig15-device/results-map-tier-full-575-06-20260904.md) and raw-only review accept all **128/128 fresh processes**: host/device latency is **9.4307x** for update (97.5% CI [9.3789, 9.4896]) and **1.0904x** for lookup ([1.0797, 1.1113]). Exact-object admission subsequently accepts only its no-op under STRICT. The separate [strict-uniform report](../microbench/fig15-device/strict-uniform-map/results-full-575-01-20260904.md) and [independent review](../microbench/fig15-device/strict-uniform-map/independent-review.md) accept **72/72** fresh processes and **60/60** target-PID STRICT admissions: uniform update is **1.0008x** [0.9891, 1.0143] and lookup is **1.0778x** [1.0644, 1.0833]. | The first result is a verifier-disabled per-lane workload; the second uses constant keys/values and same-key contention. The strict lookup number covers the complete map-type path, including cache and host-coherence behavior, rather than pure PCIe/DRAM latency. Neither result establishes application performance, callback cardinality, verifier soundness, warp aggregation, or grid scaling, and the two workloads must not be pooled. |
| Device-verifier admission scaling | **Complete supporting result.** The [formal report](../workloads/llama.cpp/observability_overhead/revision-rq4/device-verifier-scaling/results-verifier-scaling-575-01-20260904.md) accepts **200/200** CPU verifier calls over 20 blocks. The log-log exponent is **1.3841** [1.3813, 1.3898] for straight-line programs and **1.0255** [1.0215, 1.0315] for uniform diamonds; at 4,096 instructions the medians are 1,899.0 and 572.0 ms. A separate default-off phase-timing preflight at bpftime commit `c1c4cf6` localizes the two 4,096-instruction samples to PREVAIL: **99.885%** of linear and **99.566%** of diamond internal wall time, while uniformity/SIMT are slightly slower for diamonds. | The preregistered near-linear hypothesis is contradicted. This establishes a program-shape-sensitive one-time CPU admission boundary and a PREVAIL-stage diagnosis, not soundness, GPU execution, attach/JIT cost, PREVAIL-internal causality, or universal complexity. The phase diagnosis is one sample per arm, not a repeated paper result. |

These additional experiments do not silently replace safety, active local-disk, Table 1,
artifact-release or paper-integration commitments. A negative but valid result
may complete a comparison; missing execution cannot.

## Execution and publication rules

The authorized paper-integration outline is: add source-backed safety
exposition beside design/implementation; add a matched-policy subsection,
execution-domain capability table and completed scheduling figure; correct
failure classification and policy/mechanism attribution; group deployment
limits in discussion; and fix the reported typography issues. Missing live
tests remain explicit TODOs, not completed claims. Preserve existing labels
and the user's paper revision. This outline is not an approval gate.

Opening correspondence: retain the existing context, workload-dependence,
agent motivation, interface requirements, related approaches, system and
challenge paragraphs in their current order. Update only the interface-contract
and result/contribution paragraphs, then derive the abstract's corresponding
claims from the revised body: headline gains compare policies; the 610 UVM
experiment measures a 3.219% same-policy cost; matched ports do not establish
equivalence; and verifier tests and enabled runtime enforcement are distinct.
No new result is introduced only in the abstract. The device-result build and
placement review is complete; repeat the whole-paper review after later edits.

GPU experiments are serialized under the existing shared leases; source review,
lightweight CPU tests and non-overlapping documentation work run in parallel.
Keep all failed and interrupted attempts, use new output directories, and do
not change frozen thresholds or omit adverse cells to obtain a favorable result.
Use actual correctness/engagement records, source revisions and explicit file
inventories, never file/content digests. Preserve unrelated worktree changes.
Commit and push scoped implementation, records and documentation after review;
do not describe local ignored artifacts as publicly released.
