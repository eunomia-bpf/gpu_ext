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

There is an actual integration warning, preserved rather than hidden: each of
the 15 async server logs reports 48 negative MemoryObj reference counts. None
of the five demand-FIFO logs contains that warning. The HTTP requests and cache
retrievals finish, but these async observations must not be promoted to a clean
final mechanism-overhead estimate. A local Qwen Next task is preparing a minimal
adapter ownership fix outside the live source; no vendored code or policy rule
has been changed, and the completed measurements are retained. There is no
additional correctness campaign or threshold required to collect performance.

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
No paper files are edited by this session. Automatic object-level offload still
requires the KV ownership/reclaim integration described in
`gds-control/disk-uvm-source-boundary-20260907.md`; completing this retrieval
batch does not complete the overall revision work.
