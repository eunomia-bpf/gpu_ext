# Owned-process cleanup repair

The interrupted [full attempt 01](../full-01-abandoned.md) exposed a real
`TimeoutExpired` after an empty non-zombie process group was observed. A group
being empty is not the same as its leader having been reaped.

`build_adapter.stop_owned` now succeeds only when `process.poll()` returns an
exit status **and** the exact owned process group is empty. It waits for both
within the existing SIGTERM three-second / SIGKILL three-second phases; it
does not add an early one-second `wait`, widen process scope, or change GPU
execution. It still fails if either condition remains unmet after the bounds.

The [two CPU regression tests](owned-cleanup.log) passed, and root separately
reran the same command successfully on CPU 17:

```sh
taskset -c 17 python3 -B workloads/expert-buffering-policy/section-vi/test_owned_cleanup.py
```

One test mocks the empty-group/not-yet-reaped race; the other checks an actual
already-exited child without sending a signal. These tests do not establish
real GPU interruption cleanup. No worker, private offloader, selector, golden
reference, threshold, or policy parameter was changed. This inventoried helper
change requires a new three-arm GPU preflight before full attempt 02; it does
not retroactively repair or validate attempt 01.
