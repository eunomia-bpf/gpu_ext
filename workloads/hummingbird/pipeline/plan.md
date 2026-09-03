# Hummingbird completion-fence ablation — preparation only

RQ3: “Does \sys improve tail latency, throughput, and resource fairness
compared to user-space and global policies in multi-tenant settings?”
Question: does the conservative completion-before-next-launch guard explain
some of the measured idle scheduler's background-goodput loss, and what
foreground protection is lost when that guard permits a bounded pipeline?

The old [50-cell study](../results-575-20260903.md) is complete and unchanged.
Its approximately 19–20% loss does not isolate the fence's causal effect.
This supporting ablation adds that missing discriminator; another scheduler
name or reanalysis of overlapping wait counters cannot answer it. A positive
result would justify a pipeline-specific protection/cost discussion; a negative
or mixed result bounds this change without explaining the entire old gap.

## Fixed comparison

One private client and policy implementation, four arms: native C / actual
host-uBPF JIT, each with a fixed outstanding-event bound of 1 or 2. Bound 1 is
the default and preserves completion-before-next-launch behavior. Bound 2
permits the next launch at the same profiled duration-minus-launch-overhead
tick while an earlier event is pending. There is no depth search or retuning.
Neither host policy bypasses the shared admission decision; CUDA launch and
event handling remain the common executor's responsibility.

Reuse the exact original VGG/ResNet model assets, transformed cubin, DISB
frontend, output-only selected profile, 1,811,879 ns SLO, 200 us preprocessing,
two independent contexts and 1,000,000/1,000,000 us timeslices. Use the frozen
periodic LC100 and BurstGPT-derived arrivals, continuous BE, and original
full-output checks. The profile qualified under bound 1; its safety at bound 2
is an uncertainty to measure, not an inherited qualification. No modified
profile, artificial GPU gap, model reduction, or outcome-selected exclusion.

Five randomized complete paired blocks per arrival, seed 20260903: 40 cells
of 60 seconds, with the same five-second gap between cells. A separate real
preflight covers all eight 10-second cells before full execution. Each output
directory must be new. Preserve unsuccessful/incomplete attempts. Estimated
formal GPU window: about 40 minutes of measurement plus setup/drain/cooldown;
preflight: 80 seconds measured plus setup. No GPU run is authorized by this
preparation document; root coordinates actual admission and exclusive leases.

## Evidence and interpretation

Retain original all-offered LC SLO attainment, arrival-to-verified response p99
(with completion coverage), and verified in-window BE goodput. Compare bound 2
against bound 1 separately for C and BPF, and BPF against C at each bound.
Use five-block geometric ratios and SLO percentage-point differences with
10,000 whole-block bootstrap draws, seed 20260903. Show every block; do not
claim equivalence. A BE benefit with worse LC protection is a tradeoff, not a
recovery win; the original joint protection thresholds remain explanatory.

Record each launch using a private ring event, reuse a slot only after a
successful completion query retires it, and fully drain each model request.
The BPF/C input includes the same actual outstanding count and configured
bound. HP pending publication and LP admission remain serialized by the same
short lock; HP arrival stops new submissions, not work already submitted.
Keep all original kernels, same-stream dependency order, exact-once CTA
partitions and no-op copies. Require numerical checks, actual JIT engagement,
issued/retired equality and zero final outstanding events. Record the observed
peak and launches issued before an older event retires; an unexercised bound-2
path is retained but cannot establish a pipeline effect.

The paper's v2 p8 describes predicted kernel-tick launch pacing and reports
**1.3% slowdown**, not 1.3 us. Its hardware device-queue statement is not
established by two host-side outstanding event records. This experiment does
not claim a two-kernel hardware queue, a microsecond preemption guarantee,
full original Hummingbird reproduction, or a device-side BPF policy.

## Preparation boundary

All new sources/patches, build outputs, runner and tests stay under this
directory. Original source, build products and raw records stay frozen. Private
copies reuse their ordinary Git source revision, explicit file inventory and
patch-application checks; no content hashes/checksums are used. OpenCode runs
the configured default model with snapshot=false and only read/glob/grep/list;
its complete advisory report will be retained, checked and reconciled with the
implementation. Root subsequently asked implementation to proceed in parallel
with the running consultation; it was not called a completed review beforehand.
CPU checks are preparation, never real-GPU evidence.

Status: private C/JIT policy and event-ring core implemented from source
revision `995bc62`. CPU 17 tests pass: 1,048 actual-JIT semantic/parity cases,
19 rejected synthetic profiles, two actual runtime-wrapper JIT cases, and
bound/reuse/retirement/order/drain/error ring checks. See `cpu-tests-01.log`.
Root source review and independent rerun of both targets pass; the private
client's `--help` also succeeds without GPU initialization. Core committed and
pushed as `51abaf2`. The private runner/analyzer now implement eight preflight
cells and 40 full cells; nine synthetic CPU tests pass, including corrupt
retirement/JIT/CTA records, incomplete matrices and unexercised depth 2. Source
patch-application checks pass. Root runner review, independent nine-test rerun,
and CLI help check pass. OpenCode's full
consultation and the accepted/rejected suggestions are retained in
[opencode-review.md](opencode-review.md) and [opencode-final.md](opencode-final.md).
The subsequent [real eight-cell preflight](preflight-results.md) passes the
independent audit and exercises depth 2 in every relevant cell. All 40 formal
performance cells remain **PENDING**; the short preflight is not their result.

CPU-only preparation/build entrypoints (no CUDA-kernel rebuild):

```bash
make -C workloads/hummingbird/pipeline -j1 test-cpu HB_CPUSET=17
make -C workloads/hummingbird/pipeline -j1 core HB_CPUSET=17
make -C workloads/hummingbird/pipeline -j1 test-runner HB_CPUSET=17
```

These targets write only `pipeline/build/`, reusing the existing split cubin
and native libraries without modifying them. The private client defaults to
bound 1; `--lp-inflight-bound 2` is the explicit ablation. Do not invoke the
client on a workload until root admits the new exclusive real-GPU preflight.

## Root-only next execution

The coordinator must retain normal CPU affinity so the inherited telemetry
worker can use CPU 16; do not wrap GPU runs in the CPU-17 preparation taskset.
The client preserves the original LC/BE/LP worker affinities. The existing
GPU0 and struct-ops lease inodes, clean explicit environment, exact 575.57.08
driver gate, continuous telemetry, before/after safety and owned process-group
cleanup are reused unchanged. No build occurs inside the run.

From the repository root, after admission (these commands have **not** run):

```bash
sudo python3 -B workloads/hummingbird/pipeline/run_study.py preflight \
  --output workloads/hummingbird/pipeline/raw/preflight-575-01 \
  --profile workloads/hummingbird/raw/small-pattern-575-01/profile-selected.json \
  --slo-ns 1811879
python3 -B workloads/hummingbird/pipeline/analyze_study.py \
  workloads/hummingbird/pipeline/raw/preflight-575-01
sudo python3 -B workloads/hummingbird/pipeline/run_study.py full \
  --output workloads/hummingbird/pipeline/raw/full-575-01 \
  --profile workloads/hummingbird/raw/small-pattern-575-01/profile-selected.json \
  --slo-ns 1811879 \
  --preflight workloads/hummingbird/pipeline/raw/preflight-575-01
```

Every directory must be new and below `pipeline/raw/`; failures are preserved.
Full admission requires all eight independently audited private preflight
cells, the same runtime inventory/profile, and observed peak 2 in every depth-2
cell. Original ten-cell preflight, source/runtime drift, smoke-only evidence or
an unexercised pipeline cannot admit full. Formal results retain all 40 cells;
`causal_interpretation_ready` additionally requires actual depth-2 exposure.
