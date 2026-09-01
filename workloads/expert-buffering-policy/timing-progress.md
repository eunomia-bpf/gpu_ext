# Expert-buffering paired timing progress

Status: in progress (2/5 paired blocks passed)

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
