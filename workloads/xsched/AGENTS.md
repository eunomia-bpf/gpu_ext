- Never generate, refresh, compare, or record file/content hashes, checksums,
  or digests for this workload, and never use them as experiment gates or
  evidence. Compare exact small outputs and structured semantics instead.
  Git commit IDs and upstream source revisions remain normal bookkeeping.
- The active runner uses exact small-diff comparison, semantic output checks,
  and ordinary file metadata; do not reintroduce content fingerprint logic.
