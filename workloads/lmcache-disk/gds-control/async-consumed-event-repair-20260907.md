# Async consumed-event ownership repair

The completed attempt-02 serving campaign exposed 48 negative-reference-count
warnings per async cell (720 across 15 cells); demand retrieval had none.
All old raw results remain unchanged.

Installed LMCache 0.5.4 peeks a completed LOADING future in
`_async_process_tokens_internal`, then releases the consumed objects in the
normal retrieval path. Later `lookup_unpin` still finds the same event and
`cleanup_memory_objs` releases those objects again.

The local-model patch adds opt-in wrappers to the existing async adapter.
After actual inner consumption and a successful outer `retrieve` return,
the wrapper removes the completed event without modifying object reference
counts. An unhealthy early return leaves it untouched. Partial-failure
ownership remains upstream behavior, not a newly established guarantee.
The wrappers add event lookups, a future-result read, and bookkeeping as well
as the final pop; their runtime cost has not yet been measured.

Root mechanically applied the patch and corrected its partial-failure prose.
`current-venv/bin/python -m py_compile` passed for the adapter. No driver,
installed dependency, policy algorithm, or paper file was changed.

The next measurement uses the existing four-arm runner and five rotated
blocks, the same 50 ms hint, and a new raw directory:

```sh
python3 -u workloads/lmcache-disk/run_gds_async_prefetch.py --output workloads/lmcache-disk/raw/gds-async-prefetch-575-20260907-03 --blocks 5 --expected-driver 575.57.08 --prefetch-lead-ms 50
```

This repairs the measured implementation rather than repeating an unchanged
cell. It measures demand/eager/native/BPF serving performance on the repaired
version; it does not test KV-pressure victim selection or automatic offload.
No repaired-version performance result is claimed at this commit.
