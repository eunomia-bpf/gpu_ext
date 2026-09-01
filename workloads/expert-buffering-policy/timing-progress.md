# Expert-buffering paired timing progress

Status: passed (5/5 paired blocks passed)

## Block 1

The frozen `U -> O -> E -> F` configuration order and prompt order `8, 7, 5, 1, 6, 4, 2, 3` completed without request, server, CUDA, policy-setter, or thermal/power-brake failures. Each cell produced 512 verified output tokens across eight measured requests after the fixed warm-up and untimed correctness phase.

| Configuration | Output tok/s | Paired interpretation |
|---|---:|---|
| plain UVM (U) | 6.1142 | mechanism baseline |
| gpubpf observe (O) | 6.1077 | -0.106% versus U in this block |
| gpubpf profile+protect (E) | 6.0992 | -0.139% versus O in this block |
| llama.cpp `--n-cpu-moe 32` (F) | 9.0674 | +48.301% versus U; contextual framework policy, not a mechanism-overhead comparison |

The observe cell recorded 302,500 mapped activations and 1,896,737 observed accesses with no setter failures. The protect cell recorded 40,974 hot-tail activations, 258,859 cold-native activations, 2,506 shared-tail activations, 239,199 hot-tail accesses, and 11,477 shared-tail accesses, with zero cold-head placements and zero setter failures.

The repeated-hot-activation proxy was 79,496,740,864 bytes for O and 79,515,615,232 bytes for E. The first valid block therefore does not show a policy benefit on this proxy. This is an interim observation, not an aggregate claim.

The F route diagnostic covered all 1,105 graphs in each of the expected 32 routed layers, with zero incomplete graphs and zero dropped trace events. After the block, the stock UVM module was restored with no UVM BTF entry, module refcount zero, 15 MiB GPU memory use, and 0% GPU utilization.

Raw request, trace, policy, snapshot, and telemetry records remain under the ignored `raw/timing/block-01/` directory. They are retained for audit but are not committed.

## Block 2 attempt 1

The observe cell reached the sixth untimed request before prompt 3 emitted EOS immediately. The API returned HTTP success with one completion token, so the fixed 64-token workload gate correctly rejected the cell. The failed attempt is retained as `raw/timing/block-02-failed-attempt-01/`. Subsequent attempts set the llama.cpp `ignore_eos` request option so every accepted request executes the same fixed 64-token generation workload.

## Block 2

The repaired attempt completed in frozen order `O -> E -> F -> U` with prompt order `8, 2, 6, 5, 4, 3, 7, 1`. Throughput was 5.8240 output tok/s for U, 5.8252 for O, 5.8230 for E, and 6.7903 for F. The paired O/U effect was +0.020%, while E/O was -0.039%. Neither is evidence of a material effect from this single block.

O recorded 312,584 mapped activations and 1,982,609 observed accesses. E recorded 43,004 hot-tail activations, 266,927 cold-native activations, 2,646 shared-tail activations, 257,659 hot-tail accesses, and 11,849 shared-tail accesses. Both cells had zero setter failures; E also had zero cold-head placements. Repeated-hot-activation bytes were 83,504,398,336 for O and 83,552,632,832 for E, again slightly higher under E.

F covered all 1,105 graphs in every expected routed layer, with zero incomplete graphs and zero dropped events. No cell experienced thermal or power-brake throttling. Stock UVM restoration again ended with no UVM BTF entry, module refcount zero, 15 MiB GPU memory use, and 0% GPU utilization.

## Block 3

The frozen `E -> F -> U -> O` order and prompt order `8, 1, 5, 7, 3, 6, 4, 2` passed. Throughput was 6.1429 output tok/s for U, 6.1454 for O, 6.1421 for E, and 6.7961 for F. O/U was +0.040% and E/O was -0.054%, again near zero in this block.

O recorded 288,523 mapped activations and 1,774,138 observed accesses. E recorded 38,760 hot-tail activations, 248,803 cold-native activations, 2,409 shared-tail activations, 217,793 hot-tail accesses, and 11,418 shared-tail accesses. Setter failures and cold-head placements remained zero. Repeated-hot-activation bytes were 74,551,656,448 for O and 74,503,421,952 for E, a small decrease under E in this block.

F again covered all 1,105 graphs in every expected routed layer, with zero incomplete graphs and zero dropped events. No thermal or power-brake throttling occurred. Stock UVM was restored cleanly with the same idle state as prior blocks.

## Block 4

The frozen `F -> U -> O -> E` order and prompt order `8, 3, 5, 7, 6, 4, 2, 1` passed. Throughput was 5.7817 output tok/s for U, 5.9455 for O, 5.9441 for E, and 6.7505 for F. O/U was +2.832%, unlike the near-zero first three blocks, while E/O remained near zero at -0.023%. The O/U difference is retained as observed and will be interpreted only in the paired five-block aggregate.

O recorded 313,662 mapped activations and 1,985,234 observed accesses. E recorded 42,187 hot-tail activations, 268,784 cold-native activations, 2,620 shared-tail activations, 247,128 hot-tail accesses, and 12,115 shared-tail accesses. Setter failures and cold-head placements remained zero. Repeated-hot-activation bytes were 81,874,911,232 for O and 81,893,785,600 for E, slightly higher under E.

F route coverage and all thermal, policy safety, and stock-UVM restoration gates passed.

## Block 5

The frozen `F -> E -> O -> U` order and prompt order `8, 4, 7, 6, 5, 3, 1, 2` passed. Throughput was 6.3635 output tok/s for U, 6.3597 for O, 6.3527 for E, and 8.8982 for F. O/U was -0.059% and E/O was -0.111%. O recorded 286,631 mapped activations and 1,739,167 observed accesses. E retained zero setter failures and zero cold-head placements. Its repeated-hot-activation proxy was 74,450,993,152 bytes versus 74,438,410,240 for O, a +0.017% difference. All route, thermal, safety, and restoration gates passed.

## Five-block paired result

The geometric-mean O/U ratio is 1.00539: a +0.539% point estimate with a paired-bootstrap 95% interval from -0.068% to +1.686%. The interval crosses zero, so this run does not resolve either mechanism overhead or a mechanism speedup. It does bound the observed effect to a small range for this workload.

The geometric-mean E/O ratio is 0.99927, or -0.073%, with a 95% interval from -0.113% to -0.035%. Every block has the same slightly negative direction, but the magnitude is negligible. The pre-registered repeated-hot-activation paired difference, E minus O, averages +10,066,329.6 bytes with a 95% interval from -21,390,950.4 to +35,232,153.6 bytes. The secondary geometric-mean E/O ratio is 1.00011, or +0.011%, with an interval crossing zero. The protect policy therefore did not improve its intended proxy or throughput in this experiment.

F/U is 1.25579, or +25.58%, with a 95% interval from +14.20% to +39.68%. This result is context only: F uses roughly 9.5 GiB of GPU memory while U/O/E use roughly 32.1 GiB, so the difference cannot be attributed to gpubpf mechanism overhead or the protect policy.

The experiment supports the narrower claim that a profile-guided, page-granular hot-residency eviction-order analogue can be expressed through the mechanism with low observed overhead. It does not reproduce the current-batch, expert-atomic buffer of Expert Buffering, and it does not support claiming that this analogue improves GPT-OSS-120B performance.
