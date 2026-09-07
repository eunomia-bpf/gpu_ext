# LMCache async-prefetch serving: implementation recovery and measurements

## Attempt 01: real serving exposed a request-metadata type mismatch

Source: campaign `5ce0d745`, async backend `ac97d02f`; RTX 5090,
driver 575.57.08. The built main GDS policy replaced the old worktree policy
and reported `attached`; the driver module was not reloaded. The campaign
held the existing GPU and struct-ops coordination locks.

Raw: `raw/gds-async-prefetch-575-20260907-01/`. Six cells completed before
the campaign accepted SIGTERM between cells (exit 3). Every cell returned
eight warm HTTP responses, but the async cells fell back to recomputation.
These are retained observations of the integration fault, not results for
a working asynchronous disk-prefetch policy.

| Block / position | Configuration | Output token/s | Warm TTFT median, ms |
| --- | --- | ---: | ---: |
| 0 / 0 | demand FIFO | 58.3475 | 90.6689 |
| 0 / 1 | eager async | 15.7043 | 3368.4379 |
| 0 / 2 | native async | 15.6417 | 3369.2589 |
| 0 / 3 | BPF async | 15.6330 | 3367.3986 |
| 1 / 0 | eager async | 15.7426 | 3261.8342 |
| 1 / 1 | native async | 15.7300 | 3319.9486 |

The async server logs report `Expected str, got int` in `request_configs`.
Installed LMCache's
`v1/lookup_client/async_lookup_message.py:23` declares that field as
`Optional[Dict[str, str]]`. The HTTP runner supplied the deadline as a JSON
integer. The async lookup decoder rejected it, and LMCache's existing
three-second lookup timeout returned zero cached tokens so vLLM recomputed.
This explains the multi-second TTFT without attributing it to BPF overhead.

Root made the one-line wire-format fix: serialize the deadline with `str`.
The adapter already converts that value with `int`, so the numeric hint and
policy algorithm are unchanged. No vendored code, timeout or correctness
threshold was modified. The invocation convenience CLI supplied by local
GLM is appended without replacing Qwen's completed campaign functions.

Attempt 02 uses the corrected serialization identically across all four arms,
in a fresh directory, with the same five-block schedule and 50 ms hint. The
faulted attempt is retained separately, not pooled with the corrected version.
The new concurrent setup (two server sequences, four HTTP workers) also differs
from the previous serial five-arm campaign; its 58.35 token/s demand result
must not be presented as an algorithmic gain over that older campaign.

Corrected campaign invocation (from repository root):

```sh
python3 -u workloads/lmcache-disk/run_gds_async_prefetch.py --output workloads/lmcache-disk/raw/gds-async-prefetch-575-20260907-02 --blocks 5 --expected-driver 575.57.08 --prefetch-lead-ms 50
```

At this record's initial creation, attempt 02 had not started. The completed
measurement below supersedes that execution status, not the retained attempt 01.

## Attempt 02: all 20 serving cells complete; no measured policy benefit

Source `8e0a42fd`; same attached BPF object and driver as attempt 01. The
campaign process exited 0 after five rotated blocks of four configurations.
All 20 servers exited 0, with 160 warm requests and zero recorded HTTP failures.
Each server log records eight retrievals of 1536/1536 cached tokens. This is
not a claim of output equivalence or transparent SSD-backed UVM.

Raw directory: `raw/gds-async-prefetch-575-20260907-02/`, including all 20
`result.json` and `server.log` pairs, `raw.jsonl`, `campaign.json`, and
`summary.json`. The command above reproduces the configuration with a new
output directory; no completed cells need to be repeated unchanged.

| Configuration | Cells | Median output token/s | Median of warm TTFT medians, ms |
| --- | ---: | ---: | ---: |
| Demand FIFO | 5 | 58.5692 | 85.6518 |
| Eager async | 5 | 58.4366 | 99.7516 |
| Native deadline policy | 5 | 58.6243 | 116.1194 |
| BPF deadline policy | 5 | 58.1878 | 116.1211 |

The positive 50 ms hint is time from HTTP submission, not a prediction of when
the GPU will consume a KV block. Native and BPF use the same completion-derived
transfer estimate and deadline rule. These measurements do not show a benefit
from that rule: both have higher TTFT than demand FIFO and eager async. The
fixed arrival schedule also limits interpretation of throughput as saturated
serving capacity. Differences between arm medians are not paired estimates;
paired analysis is being prepared from these same raw records.

### Paired estimates from the completed raw observations

Root recomputed each percentage within the same block as
`100 * (candidate / reference - 1)`, then took the median across all five
blocks. No cells were discarded. The pending local-model CLI will supply the
reusable analysis entry point; these are already computed raw-data estimates.

| Candidate / reference | Metric | Median change | Range | Improving pairs |
| --- | --- | ---: | ---: | ---: |
| BPF / native deadline | Output token/s | -0.8465% | -1.9014% to +1.3311% | 2/5 |
| BPF / native deadline | TTFT | +3.5566% | -13.3148% to +20.4513% | 1/5 |
| Native deadline / demand FIFO | Output token/s | -0.4010% | -1.5809% to +0.1271% | 1/5 |
| Native deadline / demand FIFO | TTFT | +35.6417% | +24.3657% to +48.3889% | 0/5 |
| BPF deadline / demand FIFO | Output token/s | -0.9018% | -2.0106% to -0.2708% | 0/5 |
| BPF deadline / demand FIFO | TTFT | +35.6436% | +28.6312% to +50.7155% | 0/5 |
| Eager async / demand FIFO | Output token/s | -0.0963% | -1.2498% to +1.0448% | 2/5 |
| Eager async / demand FIFO | TTFT | +14.8007% | +7.0299% to +18.9952% | 0/5 |

The broadly adverse TTFT change already occurs in native deadline policy;
it is not attributable entirely to BPF execution. Eager async also has higher
TTFT in every pair, so removing the deadline delay alone is not yet established
as sufficient to match the synchronous demand path. The source-described
deadline is an application hint, not measured GPU-use slack, and no precise
causal decomposition of scheduling, framework overhead, and read service is
claimed. The reference-count issue below remains a limitation of this version.

The initial local-Qwen analysis CLI is now runnable:

```sh
python3 workloads/lmcache-disk/analyze_gds_async_prefetch.py workloads/lmcache-disk/raw/gds-async-prefetch-575-20260907-02 --format json
```

Root ran it over the complete file; it reports 20 numeric cells and reproduces
the four arm medians above. Its output is retained as
`raw/gds-async-prefetch-575-20260907-02/analysis-demand.json`. This initial
version includes per-cell mean E2E and same-block absolute changes against
demand FIFO, not yet the native-reference percentage estimates above. Those
two small additions are with the same local model. The campaign has one record
per block/config; the initial CLI's duplicate merge behavior must not be used
to combine separate attempts or versions.

The local model has now completed `--reference` and paired percentage output.
Root ran both JSON references and the native-reference Markdown renderer on
the same 20 records. `analysis-demand-paired.json` and
`analysis-native-paired.json` retain the expanded outputs alongside the initial
`analysis-demand.json`; no previous output was overwritten. The native-reference
output reproduces the previously reported BPF/native paired throughput
-0.8465% and TTFT +3.5566%. It additionally reports paired mean-E2E change
+0.1121% (range -2.8864% to +6.5723%, improving in 2/5 blocks). All are
observations of the same implementation version with the warning described
below, not new GPU measurements or clean final overhead estimates.

```sh
python3 workloads/lmcache-disk/analyze_gds_async_prefetch.py workloads/lmcache-disk/raw/gds-async-prefetch-575-20260907-02 --reference gds_async_native --format json
```

There is an actual integration warning, preserved rather than hidden: each of
the 15 async server logs reports 48 negative MemoryObj reference counts. None
of the five demand-FIFO logs contains that warning. The HTTP requests and cache
retrievals finish, but these async observations must not be promoted to a clean
final mechanism-overhead estimate. A local Qwen Next task is preparing a minimal
adapter ownership fix outside the live source; no vendored code or policy rule
has been changed, and the completed measurements are retained. There is no
additional correctness campaign or threshold required to collect performance.

Source inspection narrows the reference-count issue: installed
`lmcache/v1/cache_engine.py:1667` obtains the completed prefetch future through
`get_event_future`, leaving it registered. Normal retrieval releases each used
object at line 937. Then
`lmcache/integration/vllm/vllm_v1_adapter.py:1149` calls `lookup_unpin` during
`wait_for_save`; its async fallback at `cache_engine.py:1555` invokes
`cleanup_memory_objs`, which pops that same event and releases its objects
again at line 1410. This source-supported duplicate-cleanup route is the
repair target. Adding a reference unconditionally would need a corresponding
owner on aborted requests and is not an adequate explanation or fix by itself.
The local-model patch should transfer the consumed event's ownership while
retaining cleanup of unconsumed requests, without changing disk policy rules.

## Active follow-up assignments

The root confirmed all previous OpenCode tasks absent from the live endpoint
before starting three new local sessions, one per model:

- Qwen 27B: CPU-only paired analysis of this campaign and explanation of the
  observed latency tradeoff; no new GPU runs.
- GLM: source-level route connecting actual serving KV block ownership to
  completed disk writes and reclaim, not merely temporary staging buffers.
- Qwen Next: a temporary minimal patch for the observed reference-count issue;
  the root reviews and applies it only after this completed campaign.

No local session is stopped for silence or an artificial short timeout.
On September 7 at 17:34:06 UTC, the Qwen Next repair request ended with
APIError HTTP 524 after its automatic retries, without writing a patch.
The root confirmed both its completed error and absence from the live status
endpoint, then resumed the same unfinished session with local GLM at 17:36 UTC.
Qwen 27B analysis and the original GLM KV-reclaim investigation continue.
This temporarily uses two GLM sessions after a terminal provider failure;
there are still only three live sessions, no duplicate active owner for the
repair, and no root-written replacement implementation. The repair session is
`ses_f83256701ffek3qZigZd1Fj2qL`; its temporary patch remains pending.
No paper files are edited by this session. Automatic object-level offload still
requires the KV ownership/reclaim integration described in
`gds-control/disk-uvm-source-boundary-20260907.md`; completing this retrieval
batch does not complete the overall revision work.

The Qwen 27B analysis task has now finished normally (`finish=stop`, no model
error); its CLI, expanded outputs and corrected interpretation note are
published. Its freed slot is assigned to a concrete implementation dependency
for disk-aware reclaim: a minimal optional victim-selection callback in the
installed vLLM scheduler's existing preemption path. The new task owns only
`gds-control/vllm-preemption-seam.patch` and its source note, and does not apply
the patch or run a separate benchmark. Default selection must remain unchanged;
selected victims must follow the original list, token-budget and scheduled-work
rollback before `_preempt_request` performs the existing reclaim/recompute
transition. This avoids treating a recorder as the offload endpoint.

The GLM route task remains responsible for the actual LMCache-backed ownership
and recovery-cost input path; the other GLM task continues the async event
ownership repair. The intended next integrated implementation uses real disk
state to change native/BPF selection and then compares real serving behavior.
A callback without that input/action integration is not automatic offload or
a new performance result. No installed scheduler, driver, or paper is changed
by this assignment, and total live OpenCode concurrency remains three.

### Disk-backed reclaim implementation, September 7, 18:16 UTC

The original GLM route session is now also assigned the new opt-in project
module `gds-control/lmcache_kv_backing_state.py`. Its real inputs are the
`req_id` passed into `CacheEngine.store`, the corresponding stored token
extents and existing LMCache key objects, and successful GDS completion
callbacks or already-present backend entries. Unknown coverage must remain
unknown. The registry is an input to actual scheduler victim selection, not
a standalone experiment or an implementation of transparent UVM paging.
The scheduler seam and async ownership repair retain their separate owners;
no fourth OpenCode session was started.

Two source/configuration findings determine the next integrated run:

- The selected `VLLMPagedMemGPUConnectorV3.from_gpu` only synchronizes its
  store stream for a non-CUDA destination (`gpu_connectors.py:647-649`).
  Our GDS staging is CUDA memory. Python `store()` return alone therefore
  must not be used as a copy-completion event or permission to reuse source
  blocks. The registry does not free blocks; vLLM's actual preemption/finish
  path remains responsible for their lifetime. Reuse inside the fixed KV
  pool is distinct from returning physical HBM to the device allocator.
- The completed demand-FIFO server log reports an 8,192-token GPU KV pool.
  With at most two running requests, 1,536 cached prefix tokens and only
  16 output tokens each, this campaign is not a KV-capacity-pressure
  comparison. Repeating its unchanged cells would not evaluate the new
  victim policy. The follow-up must use identical capacity and arrival/
  generation settings across stock, native and BPF that actually exercise
  the existing allocator's preemption path. Prompt coverage and unsaved
  decode tails must be represented separately: the connector can omit
  decode-cache saves, so a stored prompt is not a fully backed live request.

No new performance result or automatic-offload completion is claimed here.
The existing 20-cell campaign, including adverse results and reference-count
warnings, remains unchanged. No paper files or installed vendor sources were
modified for this assignment.

### Completed scheduler seam, September 7, 18:47 UTC

The local Qwen 27B task finished normally with the final
`gds-control/vllm-preemption-seam.patch` and its accompanying source note.
It adds an optional per-instance victim selector before the actual vLLM
preemption path. Default selection remains stock; callback-selected victims
must be current running requests and, under PRIORITY scheduling, stay in
the original worst-priority class. The patch handles removal before the
scheduling cursor even for previously skipped requests, and preserves the
scheduled-work rollback and upstream `_preempt_request` transition.

The patch was applied and compiled on a scratch copy by the local model;
root's dry-run against the current installed vLLM also exits zero. It has
not yet been installed into serving or measured. This completes an
implementation dependency, not the disk-backed reclaim experiment: real
LMCache state, native/BPF selection, and pressure-serving measurements still
need integration. No old campaign cells or paper files were changed.

### Next selector implementation, September 7, 18:51 UTC

After the seam task completed normally and its CLI exited zero, a fresh
Qwen 27B session took its slot. The two GLM sessions remain live, so total
OpenCode concurrency is still three. The backing-state session was reminded
to persist its bounded implementation as its input context approached
180,490 tokens of the configured roughly 200k window; it was not stopped.

The new task owns a typed candidate selector with matched native and BPF
code. Existing storage command 82 only decides a single read/write request;
it does not provide a candidate-set selection result. The proposed additive
driver patch therefore exposes at most eight scalar candidate records and
returns a selected candidate, preserving command 82 and existing hooks.
It must not repurpose an I/O priority or timestamp as a hidden victim index.
The task produces a patch artifact against the current clean 575 source,
not a live module change, plus callable native/BPF bindings in the workload.

The proposed policy minimizes estimated recovery cost per actually freeable
KV byte, considering full recomputation versus reading a known disk-backed
prefix and recomputing its unsaved tail. Native and BPF receive the same
metadata and use the same bounded integer rule; unknown coverage/rates and
unusable candidates preserve the stock choice. The disk registry supplies
real backing information, while vLLM supplies live computed tokens and
reclaimable blocks. No chosen action, future outcome, or measured favorable
timing is embedded in the BPF input. This remains an unmeasured heuristic,
not a novelty or improvement claim.

The vLLM scheduler still performs actual block release, and LMCache/cuFile
still executes storage I/O. This planned extension is not transparent
same-address SSD paging and does not make the kernel a disk transport.
Root will integrate and measure the completed path; no driver replacement,
new serving run, or paper modification occurred for this assignment.
