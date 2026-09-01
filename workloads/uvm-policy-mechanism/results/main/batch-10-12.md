# Main timing batch: blocks 10--12

All six uninstrumented processes completed with zero mismatches. No event
monitor or kprobe tracer ran during retained timing.

| Block | Order | Native ms | gpubpf ms |
|---:|---|---:|---:|
| 10 | gpubpf, native | 364.138 | 372.823 |
| 11 | native, gpubpf | 357.846 | 374.402 |
| 12 | gpubpf, native | 357.650 | 379.545 |

Every gpubpf loader recorded ready and detaching, exited zero, and passed the
bounded post-detach absence/refcount gate before the next module reload.

End state: custom 610 UVM module loaded with prefetch disabled after native
block 12, no attached policy, refcount zero, and no compute client.

No file/content hashes, checksums, or digests were generated, refreshed,
compared, or recorded for this batch.
