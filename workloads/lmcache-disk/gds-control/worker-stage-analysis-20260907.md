# Stage attribution of the four-worker write-workers campaign (offline)

[analyze_worker_stages.py](analyze_worker_stages.py) decomposes the already
published [write-workers campaign](../raw/gds-write-workers-575-20260907-five-block/)
into dispatch-stage intervals using only stored per-request timestamps. It
reads the 30 complete `result.json` cells, emits JSON on stdout, and never
launches, deletes, or rewrites anything. This is attribution analysis of an
existing dataset, not causal proof; the
[campaign report](../results-575-gds-write-workers-20260907.md) remains the
result of record. Every cell has all read/write timestamps and zero
incomplete records, so no duration is imputed anywhere.

## What the stamps mean

Reads: the runner stamps `offer_s` and copies it into `submitted_s` in
`read_worker` before calling `get_blocking`
([run_gds_mixed_backend.py:652-655](../run_gds_mixed_backend.py)); the reader
stamps `completed_s` only after that call returns plus acquiring the log
lock. The measured read `submitted_s - offer_s` gap is exactly 0.0 in all
1920 reads, so read post-dispatch = `completed_s - submitted_s` includes
adapter admission, the blocking get with its 64-thread arrival path,
scheduler and lock waits, and completion bookkeeping. It is not pure SSD
service time, and no causal device attribution is made.

Writes: `offer_s` is stamped immediately before `submit_put_task`
([run_gds_mixed_backend.py:706-709](../run_gds_mixed_backend.py)).
`submitted_s` is the entry stamp of the timing wrapper's
`timed_save` coroutine around `backend._async_save_bytes_to_disk`
([run_gds_mixed_backend.py:522-536](../run_gds_mixed_backend.py)), not a
`_save_gds` worker-entry stamp. `offer_s -> submitted_s` therefore covers
the adapter's deferral decisions, event-driven waits, and loop scheduling up
to that coroutine entry; `submitted_s -> completed_s` is the coroutine body
through completion, including its internal `asyncio.to_thread` handoff of
`_save_gds` — the segment where the write-io-workers executor limit lives —
and completion propagation.

`feedback_records` exist for writes only, one record per admission event
(initial decision plus each deferred re-admission); `request_id` is unique
per event and feedback carries no timestamps, so release events cannot be
joined to a specific write request.

## Where the measured four-worker read changes sit

For every cell's worst scheduled-latency read, pre-dispatch
(`offer_s - scheduled_offer_s`) is 0.022692 to 2.379748 ms while
post-dispatch is 51.536754 to 284.360144 ms; post-dispatch is at least
99.170067% of the tail's scheduled latency in all 30 cells. Per-arm
read pre-dispatch (cell p50s, median across the five cells) is 0.066 to
0.118 ms against 73.761 to 167.028 ms post-dispatch. The dispatch stamp
has no independently observed submission interval: `dispatch_gap` is 0.0
everywhere by construction. Tail latency is dominated by the segment after
dispatch, containing the blocking get and completion path. Pre-dispatch
waiting also varies; this does not establish that every between-arm change
occurs exclusively after dispatch. The tail request in block 0, FIFO default (r36,
286.739892 ms total) decomposes into 2.379748 ms pre-dispatch,
0.0 ms gap, 284.360144 ms post-dispatch.

Per-arm stage medians (cell nearest-rank p50 of each stage, then a median
across the five cells of one arm; separate requests' quantiles never add,
and these medians do not add to the total medians — only per-request
components add exactly, verified for all 30 tails):

| Arm | read pre | read post | write offer->entry | write save body | write total | worst-read max | defer decisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FIFO default | 0.118 | 167.028 | 0.145 | 134.174 | 134.509 | 286.740 | 0 |
| Native default | 0.071 | 73.761 | 2.030 | 69.479 | 114.738 | 117.353 | 462 |
| BPF default | 0.066 | 90.661 | 9.473 | 71.014 | 126.094 | 138.030 | 511 |
| FIFO 4 | 0.068 | 114.573 | 0.126 | 118.639 | 118.742 | 131.486 | 0 |
| Native 4 | 0.085 | 86.379 | 2.012 | 56.352 | 98.124 | 104.123 | 470 |
| BPF 4 | 0.066 | 74.243 | 0.512 | 54.793 | 91.004 | 124.444 | 464 |

All `p50` values are nearest-rank (the lower middle element for even
per-cell counts of 64 reads / 96 writes); the across-cells and across-blocks
medians are standard medians. The per-cell nearest-rank p99 of reads equals
its maximum, i.e. the tail request in the table below.

## Per-cell worst scheduled-latency read with its own components

| Block | Slot | Arm | Request | Total ms | Pre-dispatch ms | Post-dispatch ms | Post % of total |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 0 | 0 | FIFO default | r36 | 286.739892 | 2.379748 | 284.360144 | 99.170067 |
| 0 | 1 | Native default | r47 | 117.353023 | 0.061507 | 117.291516 | 99.947588 |
| 0 | 2 | BPF default | r6 | 138.029875 | 0.062064 | 137.967811 | 99.955036 |
| 0 | 3 | FIFO 4 | r7 | 127.094807 | 0.152472 | 126.942335 | 99.880033 |
| 0 | 4 | Native 4 | r48 | 104.122877 | 0.121057 | 104.001820 | 99.883736 |
| 0 | 5 | BPF 4 | r18 | 97.253901 | 0.139500 | 97.114401 | 99.856561 |
| 1 | 0 | Native default | r1 | 84.944985 | 0.056359 | 84.888626 | 99.933652 |
| 1 | 1 | BPF default | r47 | 102.792580 | 0.060807 | 102.731773 | 99.940845 |
| 1 | 2 | FIFO 4 | r4 | 131.485964 | 0.080630 | 131.405334 | 99.938678 |
| 1 | 3 | Native 4 | r50 | 97.095219 | 0.067476 | 97.027743 | 99.930505 |
| 1 | 4 | BPF 4 | r2 | 55.856167 | 0.057813 | 55.798354 | 99.896497 |
| 1 | 5 | FIFO default | r42 | 228.390618 | 0.129944 | 228.260674 | 99.943104 |
| 2 | 0 | BPF default | r47 | 122.472220 | 0.180195 | 122.292025 | 99.852869 |
| 2 | 1 | FIFO 4 | r47 | 114.857861 | 0.060412 | 114.797449 | 99.947403 |
| 2 | 2 | Native 4 | r18 | 95.431857 | 0.137358 | 95.294499 | 99.856067 |
| 2 | 3 | BPF 4 | r8 | 124.444175 | 0.058256 | 124.385919 | 99.953187 |
| 2 | 4 | FIFO default | r35 | 266.459606 | 1.219868 | 265.239738 | 99.542194 |
| 2 | 5 | Native default | r29 | 65.263909 | 0.064232 | 65.199677 | 99.901581 |
| 3 | 0 | FIFO 4 | r4 | 130.387777 | 0.216396 | 130.171381 | 99.834037 |
| 3 | 1 | Native 4 | r7 | 97.547839 | 0.062828 | 97.485011 | 99.935593 |
| 3 | 2 | BPF 4 | r7 | 83.960607 | 0.190305 | 83.770302 | 99.773340 |
| 3 | 3 | FIFO default | r32 | 255.883606 | 1.034021 | 254.849585 | 99.595902 |
| 3 | 4 | Native default | r6 | 53.105297 | 0.055316 | 53.049981 | 99.895837 |
| 3 | 5 | BPF default | r4 | 83.211746 | 0.309062 | 82.902684 | 99.628584 |
| 4 | 0 | Native 4 | r3 | 53.988681 | 0.062515 | 53.926166 | 99.884207 |
| 4 | 1 | BPF 4 | r40 | 51.559446 | 0.022692 | 51.536754 | 99.955989 |
| 4 | 2 | FIFO default | r33 | 235.207276 | 0.057452 | 235.149824 | 99.975574 |
| 4 | 3 | Native default | r14 | 109.033562 | 0.241071 | 108.792491 | 99.778902 |
| 4 | 4 | BPF default | r14 | 103.327622 | 0.095863 | 103.231759 | 99.907224 |
| 4 | 5 | FIFO 4 | r43 | 127.825495 | 0.057893 | 127.767602 | 99.954709 |

Each row's three components add exactly to its total. The per-request tail
totals above are also each cell's nearest-rank read p99.

## Same-block paired changes of the tail requests

Percent changes per block, numerator/denominator minus one; the paired
median is the standard median of the five block values. These tail-total
lists reproduce the report's read-p99 paired changes exactly, as expected
when p99 is the maximum.

| Comparison | Worst-read total per block (%) | Paired median (%) | Read post median (%) |
| --- | --- | ---: | ---: |
| BPF 4 / BPF default | -29.541, -45.661, +1.610, +0.900, -50.101 | -29.541 | -27.832 |
| Native 4 / Native default | -11.274, +14.304, +46.225, +83.688, -50.484 | +14.304 | +17.108 |
| FIFO 4 / FIFO default | -55.676, -42.429, -56.895, -49.044, -45.654 | -49.044 | -29.595 |
| BPF default / Native default | +17.619, +21.011, +87.657, +56.692, -5.233 | +21.011 | +19.353 |
| BPF 4 / Native 4 | -6.597, -42.473, +30.401, -13.929, -4.500 | -6.597 | -6.387 |
| Native 4 / FIFO 4 | -18.075, -26.155, -16.913, -25.186, -57.764 | -25.186 | -24.961 |
| BPF 4 / FIFO 4 | -23.479, -57.519, +8.346, -35.607, -59.664 | -35.607 | -36.984 |

The write offer-to-entry changes also vary across blocks; paired percentages
against the near-zero FIFO release medians
(0.126-0.145 ms) are unstable and are kept in the JSON only, not treated as
findings.

## Write deferral and its overlap with demand reads (recorded state)

Across all 30 cells: 6707 decision events, 4800 submit_now (including the
1920 automatic demand-read admissions, which produce no feedback records),
1907 defer decisions, 0 recompute, 0 blocking_defer_miss. Deferral appears
only in the native and BPF arms (per arm: 462/470 default/four for native,
511/464 for BPF); the two FIFO arms never defer despite recorded pending
demand reads (submit_now events with `pending_demand_reads > 0`:
414/480 in FIFO default, 279/480 in FIFO 4, versus 2, 0, 13 and 7 of 480 in
the other arms). Because reads are demand-marked and the adapter fails
closed on any non-SUBMIT_NOW demand decision, no read was ever deferred.

All 1907 defer records carry `pending_demand_reads > 0` (minimum 1, median
25, maximum 62), every one requests `requested_wait_ns = 1000000`, and the
recorded `remaining_budget_ns` against the 200 ms cumulative write budget
stays at 196.216904 ms minimum, 199.960516 ms median: the budget was far
from exhaustion at those defer-decision points. This does not describe all
decisions: 23 final `submit_now` records have zero remaining budget (13 BPF
default, 7 BPF four-worker, and 3 native default). `feedback_records` carry
no timestamps and no wake-reason field, so the exact cause of each release
(demand-read drain versus budget expiry) is not reconstructible from the
stored records; the counts above are the recorded final-decision state,
reported descriptively. The measured write release window
(`offer_s -> submitted_s`, one per write request) has a pooled maximum of
201.114902 ms (block 0, BPF default, w5): about 1.1 ms beyond the 200 ms
budget, consistent with the budget bounding the deferral wait while the
stamped window adds loop scheduling and coroutine entry.

The write save body (`submitted_s -> completed_s`) holds the large write
latency: 54.793 to 134.174 ms arm medians. This is the segment containing
the `to_thread` executor whose size the option changes; the corresponding
read latencies also live post-dispatch. Both stage-localizations are
consistent with a shared-executor concurrency effect and rule nothing in or
out about the device itself.

## Controls and negative findings preserved

All six arms and their five within-block pairs are kept above. Native
default remains the stronger read control: its four-worker tail changes are
mixed (-50.484 to +83.688% with paired median +14.304%), so four workers is
not a universal improvement. The old default-worker BPF over native tail
disadvantage is accompanied by larger post-dispatch latency (paired tail
median +21.011%, read
post +19.353%; four adverse pairs). FIFO 4 improves its default's tails in
all five pairs and stays visible as the transport control rather than being
credited to BPF. FIFO write throughput's one adverse pair from the report
is not re-derived here and stands unchanged.

## Recommendation

Keep the four-worker configuration opt-in exactly as decided in the
campaign report: this decomposition localizes the measured changes into the
post-dispatch segments on both request types but does not identify a device
or driver mechanism, and no worker count beyond 0 and 4 exists in the data.
No tuning, no further worker-count search, and no default change follow
from these observations. `analyze_worker_stages.py` can be re-run offline on
the same stored cells (`python3 gds-control/analyze_worker_stages.py
raw/gds-write-workers-575-20260907-five-block` from
[workloads/lmcache-disk](..)); the published per-cell summary analyzer
(Qwen, commit 24ec6c3c) remains the campaign analyzer of record. No paper
or README changes.
