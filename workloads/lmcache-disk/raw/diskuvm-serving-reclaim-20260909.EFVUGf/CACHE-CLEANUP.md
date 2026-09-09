# Disposable cache cleanup

2026-09-09 PDT, after result publication and push in 40937690.
The five serving workers and runner had exited; original-module restoration
was complete. No cache file was Git-tracked.

Removed only the five generated KV cache directories below. Each had apparent
size 1,208,352,768 bytes; total 6,041,763,840 bytes (about 5.627 GiB).
These are disposable disk KV contents, not source, reports, logs, or model weights.
They were deleted, not moved to trash; the original workload regenerates them.

- cells/block-00/position-0-stock/cache
- cells/block-00/position-1-native/cache
- cells/block-00/position-2-bpf/cache
- cells/block-01/position-0-native/cache
- cells/block-01/position-1-bpf/cache

All raw JSON, streamed response records, server logs, source snapshots and
the failure analysis remain committed. No previous performance number was removed.

