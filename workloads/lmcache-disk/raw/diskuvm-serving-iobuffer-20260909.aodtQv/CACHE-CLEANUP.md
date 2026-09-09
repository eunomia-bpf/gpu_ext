# Completed serving cache cleanup — 2026-09-09

After publication of all small observations in c85e0bc0 (aodtQv) and
2aa3f7c6 (y4NKC1), root removed the 15 generated KV cache directories below.
Both runners had exited; original UVM restoration was recorded; no vLLM
serving process remained. Removal used explicit directory paths, not a
workspace-wide operation.

The recorded pre-removal apparent size was 18125291520 bytes
(16.880 GiB), including directory metadata. Each directory
was 1208352768 bytes. This is not a measured filesystem free-space delta.

- `workloads/lmcache-disk/raw/diskuvm-serving-iobuffer-20260909.aodtQv/cells/block-02/position-0-bpf/cache`
- `workloads/lmcache-disk/raw/diskuvm-serving-iobuffer-20260909.aodtQv/cells/block-02/position-1-stock/cache`
- `workloads/lmcache-disk/raw/diskuvm-serving-iobuffer-20260909.aodtQv/cells/block-02/position-2-native/cache`
- `workloads/lmcache-disk/raw/diskuvm-serving-iobuffer-20260909.aodtQv/cells/block-00/position-0-stock/cache`
- `workloads/lmcache-disk/raw/diskuvm-serving-iobuffer-20260909.aodtQv/cells/block-00/position-1-native/cache`
- `workloads/lmcache-disk/raw/diskuvm-serving-iobuffer-20260909.aodtQv/cells/block-00/position-2-bpf/cache`
- `workloads/lmcache-disk/raw/diskuvm-serving-iobuffer-20260909.aodtQv/cells/block-01/position-0-native/cache`
- `workloads/lmcache-disk/raw/diskuvm-serving-iobuffer-20260909.aodtQv/cells/block-01/position-1-bpf/cache`
- `workloads/lmcache-disk/raw/diskuvm-serving-iobuffer-20260909.aodtQv/cells/block-01/position-2-stock/cache`
- `workloads/lmcache-disk/raw/diskuvm-serving-iobuffer-20260909.aodtQv/cells/block-04/position-0-native/cache`
- `workloads/lmcache-disk/raw/diskuvm-serving-iobuffer-20260909.aodtQv/cells/block-03/position-0-stock/cache`
- `workloads/lmcache-disk/raw/diskuvm-serving-iobuffer-20260909.aodtQv/cells/block-03/position-1-native/cache`
- `workloads/lmcache-disk/raw/diskuvm-serving-iobuffer-20260909.aodtQv/cells/block-03/position-2-bpf/cache`
- `workloads/lmcache-disk/raw/diskuvm-serving-fixedmode-20260909.y4NKC1/cells/block-00/position-0-stock/cache`
- `workloads/lmcache-disk/raw/diskuvm-serving-fixedmode-20260909.y4NKC1/cells/block-00/position-1-native/cache`

Raw JSON, server logs, responses, reports, scripts, sources and model weights
remain. These disposable cache directories were deleted, not moved to trash;
regeneration requires running the workload again. Their generated KV payloads
were deliberately not committed. No worktree or Git metadata was removed.

