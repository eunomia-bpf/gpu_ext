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

At this record's creation, attempt 02 has not started. No policy-performance
benefit or complete five-block comparison is established.
