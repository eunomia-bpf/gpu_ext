# LMCache GDS controlled policy-input ablation

RTX 5090, driver 575.57.08, LMCache 0.5.4 cuFile compatibility path.
Five cyclically rotated blocks completed all 25 cells, with 200 warm requests,
zero warm request failures, and process return code 0 for every cell.
All arms use 768 MiB KV capacity and 256 MiB GDS staging. Each cell runs
eight cold requests followed by eight warm requests, generating 16 tokens
per request. The warm interval excludes population and startup.

| Configuration | Median output token/s | Median TTFT (ms) |
| --- | ---: | ---: |
| FIFO | 38.2426 | 78.0864 |
| BPF immediate-submit inputs | 36.6713 | 79.4836 |
| Native defer inputs | 37.2024 | 78.5806 |
| Native full input vector | 37.8628 | 78.1183 |
| BPF full input vector | 36.7143 | 78.8732 |

Immediate-submit inputs set pressure to zero. The other arms set controlled
pressure to 801 permille and slack to 10 ms. The full vector additionally
enables speculative recomputation with transfer/recompute estimates of 5/1 ms.
These are fixed policy inputs, not measured live HBM pressure or service cost.

## Same-block comparisons

Percent changes use 100 * (BPF/native - 1), or the named comparator.
The median of paired changes differs from the ratio of arm medians above.

| Throughput comparison | Blocks 0, 1, 2, 3, 4 (%) | Median (%) | Range (%) |
| --- | --- | ---: | --- |
| BPF full / native full | -6.496, -3.033, +1.511, -0.185, +0.319 | -0.185 | [-6.496, +1.511] |
| BPF immediate / FIFO | -6.411, -3.111, -1.876, +1.040, -4.109 | -3.111 | [-6.411, +1.040] |
| Native defer / FIFO | -5.447, -5.446, -0.492, +3.151, -2.007 | -2.007 | [-5.447, +3.151] |

BPF full/native full paired TTFT changes are +4.688, +0.966, -1.396, +0.749,
and +0.060 percent; their median is +0.749 percent. These observed ranges
are not confidence intervals or equivalence bounds. The BPF immediate-submit
arm is slower in four pairs; retain that adverse observation alongside the
earlier, separate five-arm campaign. No scheduling benefit is established.

## Scope and next experiment

This sequence finishes cold population before warm demand reads. The adapter
always immediately admits demand reads, so enabling speculative recomputation
does not establish that a real KV recomputation path ran. Write deferral is
configured during population, which lies outside warm timing. The adapter
does not enforce batch hints. Thus this is a controlled-input integration
comparison, not a demonstration of live pressure adaptation, write coalescing,
or read/write contention benefits. Hardware NVMe-to-GPU DMA is not established.

The [next experiment](gds-control/next-policy-experiment.md) overlaps real
background writes with urgent reads and measures offer-to-completion latency
plus completed write bandwidth through the same native/BPF executor.

## Records

The main campaign is [raw/gds-policy-ablation-575-20260906-five-block](raw/gds-policy-ablation-575-20260906-five-block/campaign.json),
with complete cell records in raw.jsonl and medians in summary.json.
The separate [one-block pilot](raw/gds-policy-ablation-575-20260906-block1/summary.json)
is retained and not pooled. The preceding [five-arm result](results-575-lmcache-gds-five-arm-20260906.md)
also remains separate. No additional admission, clock or correctness gates
were introduced, and no cell was retried or selected out.
