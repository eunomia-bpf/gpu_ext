# Main timing batch: blocks 07--09

All six uninstrumented processes completed with zero mismatches. No event
monitor or kprobe tracer ran during retained timing.

| Block | Order | Native ms | gpubpf ms |
|---:|---|---:|---:|
| 07 | native, gpubpf | 365.559 | 373.229 |
| 08 | gpubpf, native | 369.057 | 373.870 |
| 09 | native, gpubpf | 360.900 | 383.329 |

Every gpubpf loader recorded ready and detaching, exited zero, and passed the
bounded post-detach absence/refcount gate before the next module reload.

End state: custom 610 UVM module loaded with prefetch enabled after gpubpf block
09, no attached policy, refcount zero, and no compute client.

No file/content hashes, checksums, or digests were generated, refreshed,
compared, or recorded for this batch.
