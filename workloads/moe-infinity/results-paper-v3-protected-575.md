# MoE-Infinity paper-v3 policy comparison: five complete protected-executor blocks

Completed on 2026-09-03 at 02:43 UTC. **All five scheduled paired blocks and
15 cells passed independent raw-artifact audit**, with no failed, incomplete,
rejected, or unverified attempts in this campaign. This is a completed
same-frontend algorithm-reimplementation comparison, not reproduction of the
paper's original hardware/model numbers.

The BPF and native paper-policy throughput estimates are close: paired BPF/native
geometric ratio **0.996540**, 95% block-bootstrap CI **[0.989239, 1.005508]**.
This is not a formal equivalence result: no equivalence margin was prespecified,
and five paired blocks cannot establish a universal performance claim. Both
paper-policy arms have **lower primary throughput than the current baseline**,
while their secondary first-visible-text latency is lower. Prefetch waste remains
substantial in both arms; prediction-set protection did not eliminate it.

## Evidence and comparison scope

The authoritative campaign is
[timing-849ea75d-02-postboot](raw/paper-v3-575/timing-849ea75d-02-postboot/).
Its [manifest](raw/paper-v3-575/timing-849ea75d-02-postboot/manifest.json)
retains the seeded schedule, runtime file inventory and sizes, model inventory,
driver stage, corpus, and fixed protocol. The
[independent final analysis](raw/paper-v3-575/timing-849ea75d-02-postboot/audited-analysis-final.json)
recomputes accepted blocks through `paper_result_audit.audit_block`; it does not
accept the producer's summary as raw-output evidence. The retained
[producer analysis](raw/paper-v3-575/timing-849ea75d-02-postboot/analysis.json)
is supplementary.

| Mode | Algorithm and role | Policy execution |
|---|---|---|
| `native-off` | Current pinned GPT-OSS default, prediction disabled; baseline | Original dispatcher visit-count cache eviction |
| `paper-native` | Explicit paper-v3 activation-aware algorithm reimplementation | Native Python/C++ selectors |
| `paper-bpf` | The identical reimplementation and shared feature inputs | Real userspace bpftime/ubpf JIT match, rank, and eviction selectors |

The baseline is **not** the old llama.cpp native-UVM experiment. The BPF arm is
**not** the old kernel-UVM stride/LFU policy. Paper-native is not merely the
dormant upstream predictor switched on: the flattened-cosine matching,
probability/reuse formulas, and EAMC replacement follow the explicitly documented
paper-v3 port. See [algorithm and execution scope](activation-aware-port.md).

All three arms use the same GPT-OSS-120B weights, serving frontend, router,
repaired expert-row kernels, deterministic accumulation, and output transport.
The pinned author source revision is
`b766f8f1f6379fac6cd23594713ba6f4c7650ad9`. The algorithm source is the downloaded
[MoE-Infinity v3 PDF](../../docs/paper-material/policy-expressibility-papers/20-2024-moe-infinity.pdf),
sections 4.3–4.7, Algorithm 1, and appendix B.1. This model and RTX 5090 deployment
are outside the original paper's evaluated hardware/model combination.

The common native/BPF executor protects the current prediction set from
speculative eviction and rejects unstarted stale-epoch work, matching the pinned
upstream scheduling behavior. Demand bypasses this protection and retains
progress; already-issued copies can complete. This is a shared executor
eligibility rule, not a fourth BPF policy. No candidate-score/victim-score
admission threshold or top-k budget was added. The baseline retains the original
prefill extra-expert overload shortcut, whereas the two paper-policy arms enforce
their strict cache budget. Thus BPF/native isolates the decision execution
substrate much more closely than paper-policy/baseline, which also changes cache
policy and executor behavior.

## Fixed protocol and verification

- One RTX 5090, NVIDIA 575.57.08, Linux 6.15.11; declared module stage
  `/opt/gpubpf/modules/575.57.08/gpreempt-849ea75d-6.15.11`, 400 W power limit.
  Serving used CPU cores 0–7; independent offline audits used CPU 8.
- Five seeded paired blocks, each containing all three modes in the manifest's
  randomized order. Each cell starts a fresh server/cache/EAMC and runs the same
  excluded warm-up prompt. Eight measured prompts are permuted between blocks
  and kept in the same order across all three arms within each block.
- Identical memory-budget setting 0.75, 128 KV blocks, one active sequence,
  and 512-input/64-output-token requests. No concurrent GPU experiment ran in
  these measurement windows. Model loading and warm-up are excluded.
- **Primary throughput:** 512 verified output tokens divided by the entire
  eight-request wall window, including the final prefetch drain. This is not
  decode-only throughput or a mean of selected request rates.
- **Secondary TTFT:** time to first nonempty visible text, not necessarily the
  first model token. Cell TTFT and E2E values are medians over eight requests;
  the mode summaries below are medians over the five cell medians.
- Every one of the **120 measured SSE requests / 7,680 output tokens** passed
  exact same-frontend golden-text checks and strict raw SSE framing/completion,
  with 64 token frames per request. In formal timing, independent engine and
  metric counters each advanced by 512 tokens over the whole eight-request cell;
  they were not separately sampled around every individual request. Each arm
  contributes 40 measured requests and 2,560 output tokens. Warm-up is additional,
  excluded work. The enhanced canaries separately verified both counters at the
  single-SSE-request granularity of 64 tokens.
- The freshly rebuilt protected executor had already passed
  [three-arm enhanced numerical/nonstream/SSE canaries](raw/paper-v3-575/preflight-protected-849ea75d/README.md),
  including real same-snapshot native/BPF selector checking. Shadow verification
  was **off in formal timing**. Zero mismatch counters during timing therefore
  do not mean that every timed selector decision was separately oracle-checked.
- All 15 cells exited normally with code 0 and passed cleanup and raw telemetry
  checks. The retained telemetry reports peak temperatures of 52–54 C and no
  disqualifying throttle event. Samples indicating the fixed power cap remain
  recorded rather than being removed. Final cleanup reported GPU 2 MiB/0%, no
  compute process, UVM refcount 0, empty struct_ops, and no new recorded Xid or
  abnormal driver log. Both experiment leases were released by the coordinator.

## Results

These are medians over five independently audited cells per mode, not pooled
request estimates. Lower latency and higher throughput are better.

| Mode | Primary output tokens/s | First visible text, ms | Request E2E, ms | Final drain, ms |
|---|---:|---:|---:|---:|
| Baseline `native-off` | 11.8964 | 1975.94 | 5348.88 | 15.67 |
| `paper-native` | 11.2233 | 1513.44 | 5695.36 | 16.01 |
| `paper-bpf` | 11.1900 | 1535.96 | 5713.35 | 15.97 |

Paired estimates use the geometric mean of the five within-block ratios and a
10,000-draw percentile bootstrap that resamples whole paired blocks, seed
20260904. These intervals describe variation over this small fixed-corpus
campaign, not independent evidence from 40 unrelated trials. TTFT is explicitly
secondary; its lower values do not overturn the primary throughput result.

| Numerator / denominator | Primary throughput ratio [95% CI], higher is better | Secondary TTFT ratio [95% CI], lower is better |
|---|---:|---:|
| BPF / native paper policy | 0.996540 [0.989239, 1.005508] | 0.995129 [0.958082, 1.028926] |
| Native paper policy / baseline | 0.934018 [0.921858, 0.944075] | 0.778947 [0.764961, 0.793039] |
| BPF / baseline | 0.930786 [0.925380, 0.936942] | 0.775152 [0.757434, 0.795402] |

The paired BPF/native throughput point estimate is 0.35% lower, with its interval
spanning one. Relative to the baseline, native and BPF paper-policy throughput
point estimates are 6.60% and 6.92% lower, respectively; their visible-text TTFT
point estimates are 22.11% and 22.48% lower. A ratio of the separately reported
mode medians need not equal the geometric mean of paired ratios. In particular,
the BPF/native median-TTFT ordering and paired geometric-TTFT ordering differ;
neither supports a stable BPF latency advantage here.

The complete per-block primary results are retained, including the block where
BPF exceeds native and the blocks where it does not:

| Block | Mode order | Baseline tokens/s | Native paper tokens/s | BPF tokens/s |
|---|---|---:|---:|---:|
| [1](raw/paper-v3-575/timing-849ea75d-02-postboot/block-01-attempt-01/result.json) | BPF, native, baseline | 11.8618 | 11.1562 | 11.0762 |
| [2](raw/paper-v3-575/timing-849ea75d-02-postboot/block-02-attempt-01/result.json) | baseline, native, BPF | 11.8785 | 11.2368 | 11.1920 |
| [3](raw/paper-v3-575/timing-849ea75d-02-postboot/block-03-attempt-01/result.json) | BPF, baseline, native | 11.8964 | 11.2297 | 11.0547 |
| [4](raw/paper-v3-575/timing-849ea75d-02-postboot/block-04-attempt-01/result.json) | baseline, BPF, native | 12.1090 | 11.0536 | 11.1932 |
| [5](raw/paper-v3-575/timing-849ea75d-02-postboot/block-05-attempt-01/result.json) | native, baseline, BPF | 12.1043 | 11.2233 | 11.1900 |

## Actual policy engagement and remaining prefetch waste

The following counters are **sums of the five measured-window deltas**, excluding
warm-up. End-resident counts sum the final drained snapshots; initial unused
prefetch-resident counts were zero. A hit is the first demand use of a prefetched
expert; waste means eviction of a completed prefetch before its first use.
An unused resident left at cell end is reported separately, not counted as a
completed wasted eviction.

| Counter | Baseline | Native paper policy | BPF paper policy |
|---|---:|---:|---:|
| Prefetch copies started and completed | 0 | 184,068 | 184,544 |
| First-use prefetch hits | 0 | 50,286 | 50,484 |
| Unused-prefetch evictions | 0 | 132,820 | 133,116 |
| Unused prefetched residents at cell ends | 0 | 962 | 944 |
| Wasted / completed prefetches | N/A | 72.158% | 72.132% |
| Logical prefetch H2D bytes | 0 | 2,438,250,135,552 | 2,444,555,452,416 |
| Logical bytes in unused-prefetch evictions | 0 | 1,759,395,348,480 | 1,763,316,301,824 |
| Protected-resident eligibility skips | 0 | 117,494,991 | 117,850,950 |
| Stale unstarted items discarded | 0 | 75 | 85 |
| Victim recheck rejections | 0 | 13 | 11 |
| Prefetch no-victim skips | 0 | 0 | 0 |
| Actual JIT rank calls | 0 | 0 | 92,160 |
| Actual JIT EAMC match calls | 0 | 0 | 92,160 |
| Actual JIT eviction calls | 0 | 0 | 246,988 |
| Scored native/JIT eviction selections | 0 | 246,746 | 246,988 |

For both arms, `completed = first-use hits + wasted evictions + unused residents`
holds, and copy-started equals copy-completed after every real drain. These
logical transfer-byte counters count repeated expert copies; they are not a
PCIe hardware-byte measurement. For scale, the BPF arm accumulated 2,276.67 GiB
of prefetch copy bytes, of which 1,642.22 GiB later belonged to unused-prefetch
evictions, over all five cells. The wasted fractions by block were:

| Block | Native paper policy | BPF paper policy |
|---|---:|---:|
| 1 | 62.23% | 62.28% |
| 2 | 73.79% | 73.51% |
| 3 | 70.32% | 70.46% |
| 4 | 75.58% | 75.45% |
| 5 | 75.09% | 75.15% |

Thus the first block's approximately 62% waste must not stand in for the complete
campaign's approximately 72%. Protected-resident skips count encounters during
eligibility traversal, including residents that might fail other safety checks;
they are **not** counts of prevented copies or saved bytes. Both paper arms
selected the same 46,588,058 total candidate entries and made 92,160 matched
predictions. Asynchronous completion, demand interleaving, and cache evolution
still yield different transfer and eviction counts; those counts are not an
identity requirement for two separate runs.

The remaining waste is a concrete candidate for further causal study, but this
campaign does not isolate its contribution to the throughput deficit. It also
does not establish that the protection change improved performance: the previous
executor has no complete five-block paired campaign for a controlled comparison.
Avoid attributing the entire baseline gap to BPF execution, or interpreting
protection-counter activity as elimination of cache churn.

## Why the baseline is faster: observations and unresolved causes

Across all five blocks, pooled expert-cache hit rates were **84.241% baseline,
87.231% native paper policy, and 87.278% BPF**. Better hit rate therefore did not
translate into higher throughput. Median per-cell measured process CPU-time
deltas were **77.69 / 107.78 / 107.54 seconds**, respectively. These accumulate
work across threads; they are neither critical-path durations nor JIT-only costs.
Sampled GPU-memory peaks span **31,872–31,880 MiB** across the arms, providing no
evidence that the baseline used substantially more memory. Short-lived extra
residency need not appear in the 0.2-second telemetry samples.

There is nevertheless a concrete execution-path difference in
`deps/MoE-Infinity/core/parallel/expert_dispatcher.cpp`. On a full-cache miss,
lines 760–782 permit only prediction-off, multi-token prefill to use one temporary
overload slot per GPU. Lines 877–881 do not charge that slot to the cache ledger
or insert it into the resident-cache set; lines 905 and 1029–1035 arrange removal
after computation and release the slot. Both paper arms instead evict residents
until the strict budget admits the requested expert. Thus the same 0.75 setting
does not guarantee identical residency or replacement behavior. The temporary
slot could preserve existing hot experts, but is not a large permanent cache.

The existing counters do not record overload uses, bytes, waits, or phase-specific
hit/miss counts; none of the five baseline logs retains the DEBUG overload
messages. Their absence does not prove the branch was unused. Consequently,
neither its actual trigger count nor its contribution to the baseline advantage
is established. This remains a deployment comparison, not a pure policy-cost
experiment with identical executors.

Source inspection also identifies unprofiled common costs in both paper arms:
`paper_policy.py:173` constructs two default arrays for every `setdefault` call,
even when the request already exists; `distributed/expert_executor.py:218–221`
computes predictions and converts all reuse scores before enqueueing current
demand; dispatcher lines 626–631 make the sole fetch worker wait for each issued
prefetch's CUDA event. Queued demand fetches cannot bypass that transfer.
The `distributed/` path is relative to `deps/MoE-Infinity/moe_infinity/`.
These are concrete work/synchronization sites, not measured explanations of the
CPU or throughput gap. This inspection found no concrete cosine, decay, or
ranking-direction error; unused-prefetch bytes are not wasted-time measurements.

The next causal test should toggle prefetch on/off with the **same executor,
cache budget, overload setting and eviction policy**, recording prefill/decode
hit/miss, temporary-slot use, and demand-versus-prefetch copy/wait counters.

## Limits and replay

This campaign covers one GPU, model, memory setting, concurrency level, fixed
eight-prompt corpus, and short output length. Each fresh cell finishes with only
18 prefill/decode EAMC matrices, below its capacity of 1,000: full-collection
diversity replacement is CPU-tested but was not dynamically exercised by these
GPU measurements. Likewise no-victim speculative skipping did not occur in the
timed windows; its progress behavior has separate helper tests. The new canaries
and actual JIT counters establish implementation engagement, not all-workload
policy quality or kernel-driver execution of these userspace decisions.

The earlier interrupted `timing-849ea75d-01` artifacts remain intact and are not
pooled here. Its apparently successful native producer record failed real-raw
audit after reboot; the campaign has zero valid paired blocks. See the
[recovery report](recovery-20260903-013741.md). The reboot cause was not established.
All blocks in the present campaign were newly measured with the rebuilt
protected executor and durable post-writer artifact publication.

Recompute the independent report without writing or using a GPU:

```bash
cd workloads/moe-infinity
taskset -c 8 /usr/bin/python3 -B analyze_paper_comparison.py \
  raw/paper-v3-575/timing-849ea75d-02-postboot
```

Expected result: exit 0, `complete: true`, five accepted attempts, and empty
failed/incomplete/rejected/unverified lists. The analyzer uses the existing
strict raw auditor, including SSE/goldens, tokens, time windows, runtime
inventory metadata, actual engagement, telemetry, and cleanup. The declared
runtime/model/corpus files must still be present and unchanged; this is not a
portable missing-artifact bypass. `--output NEW.json` creates an additional
report exclusively and refuses to overwrite the retained final report.

Implementation changes, the protected-executor build, and enhanced canaries were
committed before formal timing. The complete raw blocks were independently
audited and pushed incrementally in commits `569f3f5`, `8fe49e0`, `971af82`,
`4827289`, and `788334d`. No partial block was promoted into a valid pair, and no
old damaged response was reconstructed to supply missing performance evidence.
