# Main timing batch: blocks 13--15

All six uninstrumented processes completed with zero mismatches. No event
monitor or kprobe tracer ran during retained timing.

| Block | Order | Native ms | gpubpf ms |
|---:|---|---:|---:|
| 13 | native, gpubpf | 365.399 | 365.255 |
| 14 | gpubpf, native | 368.669 | 373.866 |
| 15 | native, gpubpf | 361.095 | 374.766 |

Every gpubpf loader recorded ready and detaching, exited zero, and passed the
bounded post-detach absence/refcount gate before the next module reload.

End state: custom 610 UVM module loaded with prefetch enabled after gpubpf block
15, no attached policy, refcount zero, and no compute client. A separate 1 GiB
recovery smoke completed with zero mismatches.

No file/content hashes, checksums, or digests were generated, refreshed,
compared, or recorded for this batch.
