# Main timing batch: blocks 04--06

All six uninstrumented processes completed with zero mismatches. No event
monitor or kprobe tracer ran during retained timing.

| Block | Order | Native ms | gpubpf ms |
|---:|---|---:|---:|
| 04 | gpubpf, native | 364.041 | 373.382 |
| 05 | native, gpubpf | 359.188 | 375.039 |
| 06 | gpubpf, native | 366.389 | 375.125 |

Every gpubpf loader recorded ready and detaching, exited zero, and passed the
bounded post-detach absence/refcount gate before the next module reload.

End state: custom 610 UVM module loaded with prefetch disabled after native
block 06, no attached policy, refcount zero, and no compute client.

No file/content hashes, checksums, or digests were generated, refreshed,
compared, or recorded for this batch.
