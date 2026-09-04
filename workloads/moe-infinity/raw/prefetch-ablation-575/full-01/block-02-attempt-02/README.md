# Block 02 attempt 02: retained build-contention failure

Date: 2026-09-04 UTC

Status: **failed attempt; excluded from performance analysis**.

The unchanged resume completed the scheduled `bpf-prefetch-on` and
`bpf-prefetch-off` cells. During the third arm, `native-prefetch-on`, the
predeclared no-build-contention gate detected a running `cc1` process after
four of six measured requests and aborted the block:

```
heavy compilation overlaps GPU timing: ['1774385 R    cc1']
```

The compiler came from a concurrent CPU-only build in the separately owned
stale-state driver worktree. That build stopped without loading a module or
using the GPU. The process had exited by inspection, and the owner paused all
further compilation for the duration of this timing campaign.

The interruption is a measurement-environment failure, not a correctness or
policy outcome. The two completed cells, the partial third-cell files, and all
timings from this attempt remain excluded because the four-arm block is not
complete and host compilation overlapped its timing window. No cell is reused
in a later block.

Post-abort inspection found no compute process, 15 MiB GPU memory use, UVM
reference count zero, and no attached scheduler struct-ops object. An unchanged
resume must create `block-02-attempt-03` and rerun all four arms.
