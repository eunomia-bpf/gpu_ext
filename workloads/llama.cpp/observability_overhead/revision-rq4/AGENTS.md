- Never generate, refresh, compare, or record file/content hashes, checksums,
  or digests for this workload, and never use them as experiment gates or
  evidence. Compare exact normalized output and structured semantics instead.
  Git commit IDs and upstream source revisions remain normal bookkeeping.
- The active runner preserves and compares exact short normalized output and
  uses ordinary file metadata; do not reintroduce content fingerprint logic.
