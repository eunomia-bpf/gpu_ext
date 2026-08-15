# Stage 4 Trace Disabled-Path Overhead

Status: `IMPLEMENTED_NOT_EXECUTED`

The Stage 3 enhanced decision wrapper performs no allocation, string formatting, or printk, but the current private module still executes a noinline wrapper call and compiler barrier when no trace program is attached. Stage 3 measured a +1.149% mean difference from the older custom baseline, slightly above the target.

The Stage 4 runner provides 20 fresh independent timing runs with tracing disabled and 20 with tracing attached. No new measurement has been made, and no private kernel-module source was modified in this non-privileged implementation pass. A true static-key/listener optimization requires rebuilding and manually reloading the temporary custom UVM module, so it cannot be claimed complete here.

The acceptance target remains untraced overhead at or below 1%, reported with mean, median, standard deviation, p95, and a 95% confidence interval. Results must be reported even if the target is missed.
