# Collected workspace cleanup records

These four small records were supplied by the separate user-authorized cleanup
task and collected here by the experiment task. The experiment task did not
concurrently delete the same files. The source records were under
`/home/yunwei37/.local/state/gpubpf-disk-cleanup/`; their September 7 local
timestamps correspond to September 8 UTC.

- [Summary](SUMMARY-20260907.md): scope, approximate free-space change and preserved assets.
- [Finished LMCache payload inventory](cleanup-20260907-215933.md).
- [Legacy generated KV payload inventory](legacy-kv-cache-20260907.md).
- [Duplicate plain-log inventory](stale-log-duplicates-20260907.md): retained compressed records are the source for subsequent analysis.

The separately collected [retired-archive deletion inventory](../../../workloads/lmcache-disk/raw/retired-cache-payload-cleanup-20260908.md)
covers the 40.51 GiB temporary archive. These are cleanup records, not new
performance measurements. No cache, model or large build product is added to
Git. Historical payload-preservation statements are superseded by the relevant
deletion inventory; original measured values and logs remain preserved.
