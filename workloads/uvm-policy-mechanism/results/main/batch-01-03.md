# Main timing batch: blocks 01--03

All six uninstrumented processes completed with zero mismatches. Each cell used
the same 8 GiB allocation, 64 KiB region stride, and 131,072 unique demand
addresses. No UVM event monitor or kprobe tracer ran during these timings.

| Block | Order | Native ms | gpubpf ms |
|---:|---|---:|---:|
| 01 | native, gpubpf | 367.400 | 386.177 |
| 02 | gpubpf, native | 357.589 | 376.865 |
| 03 | native, gpubpf | 361.734 | 364.558 |

Every gpubpf loader recorded `ready` and `detaching` and exited zero. Blocks 01
and 02 exposed a short post-exit kernel-visibility window: the immediate
struct_ops query raced detach, so execution stopped before the next reload.
Read-only follow-up confirmed no attached struct_ops, UVM refcount zero, and no
compute client. The remaining cells used a bounded post-detach poll, still
requiring link absence and refcount zero before any reload. This changed no
timed path or retained measurement.

End state: custom 610 UVM module loaded with prefetch enabled, no attached
policy, refcount zero, and no compute client.

No file/content hashes, checksums, or digests were generated, refreshed,
compared, or recorded for this batch.
