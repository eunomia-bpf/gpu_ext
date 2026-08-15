# Stage 4 Joint Prefetch-Eviction Results

Status: `IMPLEMENTED_NOT_EXECUTED`

Static audit currently approves only `prefetch_always_max_cycle_moe` for runtime smoke among the three requested initial candidates. Runtime approval still requires:

1. 64 MiB timing and enhanced trace correctness.
2. Clean attach and detach with no residual struct_ops.
3. No Xid or kernel warning.
4. One 0.95x reduced-capacity timing run.

`eviction_fifo` is rejected because its implementation does not perform the FIFO ordering claimed by its comments, returns BYPASS from the unreliable access hook, and emits hot-path `bpf_printk`. `prefetch_cooperative` is rejected because it schedules `bpf_wq` work and invokes cross-VA-block `bpf_gpu_migrate_range`; its context and completion bounds are not established for the initial pressure run.

No joint policy has been attached or run in Stage 4.
