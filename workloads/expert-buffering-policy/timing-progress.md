# Expert-buffering paired timing progress

Status: in progress (1/5 paired blocks passed)

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
