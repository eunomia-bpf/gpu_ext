# R5 Phase C: 50-event source reconciliation

Date: 2026-08-31

Disposition: `PARTIAL`. The historical table has exactly 50 uniquely numbered
rows, but four rows are not fully supported at the specificity claimed. The
original multi-session transcript corpus is also absent locally.

No file/content hashes, checksums, fingerprints, or digests were generated,
compared, or recorded. Normal repository paths and line locations are used as
the evidence keys.

## Inventory check

The numbered-row check over `docs/eval/agent/q2_safety_taxonomy.md` reported:

```text
rows=50
missing_count=0
duplicate_count=0
```

Classification rules:

- `SUPPORTED`: the cited repository source directly records the incident and
  its fix, recovery, or decision response.
- `PARTIAL`: the source records the observation but not every claimed root
  cause, exact counter, or recovery detail.
- `UNSUPPORTED`: no direct repository source was found. Derived Q1/Q3/Q5/Q6
  reports and external memory files do not by themselves qualify.

## Per-event matrix

| # | Class | Direct repository source | Reconciliation |
| ---: | --- | --- | --- |
| 1 | PARTIAL | `docs/experiment/plans/archived/xcoord_v1.md:1122-1132` | Records the 120B UVM baseline, 3/20 completion, and segfault. It does not directly record the claimed later stabilization/recovery. |
| 2 | SUPPORTED | `docs/experiment/plans/msched.md:123-125` | Records `move_head` at activation, Xid 31, the page-table timing explanation, and `move_tail`/default recovery. |
| 3 | SUPPORTED | `docs/experiment/plans/msched.md:127-140` | Records hash-map hot-path latency, Xid 31, PERCPU_ARRAY replacement, and abandonment of LFU. |
| 4 | SUPPORTED | `docs/experiment/plans/msched.md:129` | Records verifier rejection of pointer arithmetic and scalarization via `bpf_probe_read_kernel`. |
| 5 | PARTIAL | `docs/experiment/plans/msched.md:131-140` | Records MRU `18.97/9.62`, `-83%`, and abandonment. The stronger attribution specifically to list-manipulation cost is not directly retained. |
| 6 | SUPPORTED | `docs/experiment/plans/msched.md:148-164` | Records stride matching no-prefetch, the BYPASS cause, and DEFAULT-on-uncertainty fix. |
| 7 | SUPPORTED | `docs/experiment/plans/msched.md:434-440` | Records the infinite-loop rejection and O(1) next-boundary replacement. |
| 8 | PARTIAL | `workloads/llama.cpp/results/policy_sweep/EXPERIMENT_LOG.md:96-98`; `extension/README.md:278-280` | Records persistent/racy struct-ops cleanup and the supported cleanup paths. It does not directly retain the claimed dirty-shutdown hang sequence. |
| 9 | SUPPORTED | `docs/experiment/plans/archived/xcoord_v1.md:1235-1240`; `docs/experiment/plans/xcoord.md:288` | Records clang-18/UEI atomic incompatibility and the direct-libbpf workaround. |
| 10 | SUPPORTED | `docs/experiment/plans/archived/xcoord_v1.md:1631-1639` | Records `select_cpu` bypassing `enqueue` and the corrected insertion path. |
| 11 | SUPPORTED | `docs/experiment/plans/archived/xcoord_v1.md:1633-1639` | Records empty worker maps for the fit-in-VRAM case and direct PID registration. |
| 12 | SUPPORTED | `docs/experiment/plans/archived/xcoord_v1.md:1633-1639` | Records mixed FIFO/PRIQ semantics and separation into two DSQs. |
| 13 | SUPPORTED | `docs/experiment/plans/archived/xcoord_v1.md:1465-1468` | Records owner-TGID versus kworker-TGID mismatch and the worker-PID map fix. |
| 14 | SUPPORTED | `docs/experiment/plans/archived/xcoord_v1.md:1260-1263` | Records residual GPU processes, OOM, cleanup, and the 0-MiB precondition. |
| 15 | SUPPORTED | `docs/experiment/plans/cross_block_prefetch_mechanism.md:1181-1195` | Records zero kprobe/workqueue counters, the missing attach, and explicit kprobe attachment. |
| 16 | SUPPORTED | `docs/experiment/plans/cross_block_prefetch_mechanism.md:1197-1233` | Records the repaired approximately-60 versus 84 tok/s comparison, extra DMA/lock contention, and the no-XB decision for this workload. |
| 17 | SUPPORTED | `docs/experiment/plans/archived/xcoord_v1.md:1787-1812` | Records the 82.10 s old-vtime result and recovery after moving to the local fast path. |
| 18 | SUPPORTED | `docs/experiment/plans/xcoord.md:252` | Records the no-stress GNN +3% scheduler overhead and the negative screening result. |
| 19 | SUPPORTED | `docs/experiment/plans/archived/xcoord_v1.md:66` | Records the 20B/120B coexistence OOM and required start order. |
| 20 | SUPPORTED | `docs/experiment/plans/archived/xcoord_v1.md:1966-1969` | Records self-matching `pkill -f` and the `pkill -x` fix. |
| 21 | SUPPORTED | `docs/experiment/plans/archived/xcoord_v1.md:1940-1968` | Records the invalid enqueue flags crash and removal of the extra PRIQ flag. |
| 22 | SUPPORTED | `docs/experiment/plans/xcoord.md:247,328` | Records two system OOMs from the 120B+120B memory footprint and the switch to 20B+FAISS. |
| 23 | SUPPORTED | `docs/experiment/plans/xcoord.md:257`; `docs/experiment/plans/archived/xcoord_v1.md:2460-2468` | Records the vLLM -62% throughput result, global-DSQ cause, and abandonment of blind boost. |
| 24 | SUPPORTED | `docs/experiment/plans/archived/xcoord_v1.md:19,67` | Records the +67% to +759% FAISS overhead and the changed GPU-aware/local-path direction. |
| 25 | SUPPORTED | `docs/experiment/plans/archived/xcoord_v1.md:69` | Records the 1-ms +15% result, 5-ms rerun, and workload-mismatch conclusion. |
| 26 | SUPPORTED | `docs/experiment/plans/archived/xcoord_v1.md:23,70,91` | Records the one-way latch, invalid closed-loop interpretation, and FPRS rewrite requirement. |
| 27 | SUPPORTED | `docs/experiment/plans/archived/xcoord_v1.md:331` | Records the 45-second first request and removal of non-GPU backpressure. |
| 28 | SUPPORTED | `docs/experiment/plans/archived/xcoord_v1.md:327-333` | Records non-decaying stale fault rate and the two-second staleness rule. |
| 29 | SUPPORTED | `docs/experiment/plans/archived/xcoord_v1.md:333` | Records approximately 50-second response and the gain/integral changes for approximately 500 ms response. |
| 30 | SUPPORTED | `docs/experiment/plans/archived/xcoord_v1.md:493-501` | Records `used=0`, the misplaced callback logic, and migration to `chunk_activate`. |
| 31 | SUPPORTED | `docs/experiment/plans/archived/xcoord_v1.md:493-502` | Records `lc_fr=0`, server OOM, and the startup-order/smaller-dataset response. |
| 32 | SUPPORTED | `docs/experiment/plans/archived/cross_block_prefetch_v1.md:670-703` | Records the CPU preferred-location allocator change, 70/140-second comparison, and reversion to plain managed allocation. |
| 33 | SUPPORTED | `docs/experiment/plans/cross_block_prefetch.md:81-86,448-454` | Records weak directionality in search, the direction-filter regression, and phase-aware gating response. |
| 34 | SUPPORTED | `docs/experiment/plans/archived/cross_block_prefetch_v1.md:896-919` | Records wrong cwd/missing request parameters, invalidated conclusion, and six-configuration rerun. |
| 35 | SUPPORTED | `docs/experiment/plans/archived/cross_block_prefetch_v1.md:1379` | Records wrong loaded inode, zero uprobe hits, corrected path, and restored hits. |
| 36 | SUPPORTED | `docs/experiment/plans/archived/cross_block_prefetch_v1.md:830-875` | Records zero SEARCH transitions under v1 and the exact +1-block stride detector used by v2. |
| 37 | SUPPORTED | `docs/experiment/plans/archived/cross_block_prefetch_v1.md:854-875` | Records 9.78 versus 5.49 seconds, retained BUILD-hot chunks, and default-LRU response. |
| 38 | SUPPORTED | `docs/experiment/plans/archived/cross_block_prefetch_v1.md:883-892` | Records the `va_space` early return, stuck SEARCH state, and reordered phase detection. |
| 39 | SUPPORTED | `docs/experiment/plans/archived/cross_block_prefetch_v1.md:1501-1522` | Records both throttled-XB regressions and rejection of fault rate as a safe-bandwidth proxy. |
| 40 | SUPPORTED | `docs/experiment/plans/archived/cross_block_prefetch_v1.md:1332-1361` | Records all llama narrowing regressions and loss of batched-transfer efficiency. |
| 41 | SUPPORTED | `docs/experiment/plans/cross_block_prefetch.md:439-444` | Records vLLM phase/narrowing regressions and retention of large-granularity prefetch. |
| 42 | SUPPORTED | `docs/experiment/plans/archived/cross_block_prefetch_v1.md:954-961`; `docs/experiment/plans/cross_block_prefetch.md:431` | Records the K=6 -46% result, PCIe over-prefetch cause, and return to one-block lookahead. |
| 43 | SUPPORTED | `docs/experiment/plans/cross_block_prefetch.md:731-742` | Records zero hook/XB/sync counters from PID filtering and removal of kernel-thread-context filtering. |
| 44 | SUPPORTED | `docs/experiment/plans/cross_block_prefetch.md:737-742` | Records the wrong libcudart path, zero sync hits, corrected path, and eight hits. |
| 45 | SUPPORTED | `docs/experiment/plans/archived/cross_block_prefetch_v1.md:1572-1576` | Records the paged-attention target mismatch and actual FlashAttention backend. |
| 46 | SUPPORTED | `docs/experiment/plans/archived/cross_block_prefetch_v1.md:1015-1034` | Records baseline-like TPOT, 73-ms P99, 48% migrate failures, and rejection of phase gating for vLLM. |
| 47 | SUPPORTED | `docs/experiment/plans/archived/cross_block_prefetch_v1.md:993-1013` | Records correct phase engagement, approximately -28% prefill, and no-XB decision at 1.84x oversubscription. |
| 48 | SUPPORTED | `docs/experiment/plans/gpu_block_access_fix_retest.md:28-47,139-144` | Records -4.8%/-3.1%, +116% DMA, abandonment at high oversubscription, and narrower/lower-pressure follow-ups. |
| 49 | PARTIAL | `docs/experiment/plans/gpu_block_access_fix_retest.md:3-24,111-130` | Directly records that the callback never ran, affected policies, relocation, and successful post-fix counters. The Q2-specific ftrace counts are not retained in a direct repository source. |
| 50 | SUPPORTED | `extension/README.md:278-280` | Records the bpftool struct-ops segfault on 6.15.11 and both cleanup alternatives. |

Totals:

| Classification | Count |
| --- | ---: |
| `SUPPORTED` | 46 |
| `PARTIAL` | 4 |
| `UNSUPPORTED` | 0 |
| Total | 50 |

## Transcript-corpus blocker

The repository's agent-study README correctly describes Q2 and Q5 as derived
classifications requiring source logs. The historical multi-session corpus
named by the derived reports is not present in the expected local project
directory; only one unrelated top-level JSONL session file is currently there.
The three external memory notes used by the original Q2 author are not a
substitute for publishable raw sessions and were not used as sole support in
the matrix above.

Therefore this reconciliation can audit repository-retained incident records,
but cannot reproduce the original session-to-event extraction, interaction
sequence, or omission check. R7 archive recovery and privacy review remain
required before releasing prompts/logs or claiming transcript-level
reproducibility.

## Aggregate interpretation

The 50-row count is reproducible, and 46 rows have direct retained support.
That does not make “50 safety events” a raw-log-derived statistic: four rows
need narrower wording, and the underlying session corpus is unavailable. Until
those conditions are repaired, R5 safety-event evidence is `PARTIAL` and the
paper must not imply an independently replayable 50-event transcript audit.
