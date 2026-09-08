# Legacy generated KV cache cleanup

User authorized removal of unused files. Removed only .pt KV payloads under the old LMCache runs cache directories below. Preserved all logs, metrics, metadata, code and model weights. No payload was open before removal.

- code-smoke-prefix1-20260901-01/cache
- code-smoke-prefix1-20260901-02/cache
- code-smoke-prefix1-20260901-03/cache
- code-smoke-prefix1-20260901-04/cache
- code-smoke-prefix1-20260901-05/cache
- storage-575-correctness-01/lmcache_disk/cache
- storage-575-preflight-02/disk/cache
- storage-575-v3-correctness-01/lmcache_disk/cache
- storage-575-v3-correctness-02/lmcache_disk/cache
- storage-575-v3-performance-smoke-01/position-2-lmcache_disk/cache
- storage-575-v3-performance-smoke-02/position-1-lmcache_disk/cache
- storage-575-v3-preflight-05/disk/cache

Payloads: 366; bytes: 9210691584.

Completed removal with root permissions for root-owned generated cache files.
