- Never generate, refresh, compare, or record file/content hashes, checksums,
  or digests for this workload, and never use them as experiment gates or
  evidence. Compare exact small outputs and structured semantics instead.
  Git commit IDs and upstream source revisions remain normal bookkeeping.
- The current legacy runner contains prohibited fingerprint logic and must not
  be executed until that logic has been replaced and independently reviewed.
