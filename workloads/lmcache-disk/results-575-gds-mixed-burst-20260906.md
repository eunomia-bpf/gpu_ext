# Mixed cuFile burst: five repetitions

All 15 fresh-process measurements completed: 960 reads, 1,440 writes, no
request or cleanup errors. The only workload change from the
[spaced-arrival comparison](results-575-gds-mixed-fresh-process-20260906.md)
is zero requested inter-arrival spacing for reads and writes. The same
64-read/96-write, 24 MiB, 4096 MiB staging configuration and policy executor
are retained. Native and BPF each make 64 submit / 96 defer decisions per
measurement; FIFO submits all 160. Deferral is a single 10 ms wait.

## Performance

Medians of five measurements per configuration:

| Metric | FIFO | Native policy | BPF policy |
| --- | ---: | ---: | ---: |
| Read call-to-completion p50, ms | 273.711 | 518.094 | 264.242 |
| Read call-to-completion p99, ms | 548.806 | 865.548 | 428.889 |
| Completed write throughput, MiB/s | 1756.845 | 1412.809 | 2438.467 |
| Total storage throughput, MiB/s | 2829.856 | 2283.027 | 3867.083 |

Same-repetition read-p99 percentage changes, 100 * (numerator/denominator - 1):

| Comparison | Repetitions 0, 1, 2, 3, 4 (%) | Median (%) |
| --- | --- | ---: |
| Native / FIFO | -0.283, +68.430, +57.715, -47.470, +340.462 | +57.715 |
| BPF / FIFO | +7.301, +96.799, -21.851, -72.748, -7.307 | -7.307 |
| BPF / native | +7.606, +16.843, -50.449, -48.120, -78.956 | -48.120 |

The apparently large BPF/native difference is not a demonstrated mechanism
speedup. Both execute the same decisions, and individual read p99 values span
316--1592 ms for FIFO, 338--1475 ms for native and 293--1723 ms for BPF.
The first repetition has all three p99 values around 339/338/363 ms, whereas
later repetitions vary widely. No cause of that variability has been isolated.
The native policy itself is slower than FIFO in three of five pairs. Therefore
this run does not establish that fixed write deferral provides a stable benefit
or that BPF intrinsically outperforms the native decision path.

## Existing-record timing decomposition

Reopening the 15 raw records separates dispatch-to-save-coroutine admission
from the subsequent save work. Across measurements, median write admission
delay ranges from 2.234--4.932 ms for FIFO, 13.818--14.709 ms for native, and
13.560--14.782 ms for BPF. Thus the requested extra 10 ms appears in both
policy paths. Median save-coroutine-entry-to-completion durations span
262--1246 ms across all configurations. This latter interval includes
LMCache's worker-queue wait, file setup, cuFile transfer and completion, not
just physical SSD service. The roughly millisecond-scale policy delay alone
does not explain the much larger observed storage-path variation.

Although requested read spacing is zero, the first-to-last read dispatch
spans 45--97 ms because the runner starts Python threads sequentially.
This motivates preparing workers before the common offered schedule in the
next runner. It is a timing decomposition of existing data, not a discarded
experiment or a requirement to delay collection. Detailed values are in
`raw/gds-mixed-burst-fresh-575-20260906-five-block/timing-breakdown.json`.

## Measurement scope and records

Latency starts at the actual API dispatch, not scheduled arrival. With 64
reads, nearest-rank p99 is the sample maximum. Zero requested spacing does not
mean physically simultaneous submissions: the common runner launches reader
threads before dispatching writes. This is real cuFile compatibility-mode
backend traffic, not model throughput or established hardware NVMe-to-GPU P2P.
Pressure 801 permille and slack 10 ms are controlled inputs, not live telemetry.

[Collection record](raw/gds-mixed-burst-fresh-575-20260906-five-block/README.md)
records source, invocation and cyclic order. The
[summary](raw/gds-mixed-burst-fresh-575-20260906-five-block/summary.json)
retains every per-repetition value, decision count and paired difference.
Every named measurement directory contains raw.jsonl, campaign.json and
summary.json. All 15 measurements, including adverse cases, are published;
none is retried, omitted or pooled with earlier runs.

Further policy development should distinguish live pending-demand feedback
from fixed delay and retain dispatch timing separately from scheduled arrival.
Collecting more identical fixed-delay repetitions alone will not add the
missing feedback mechanism.
