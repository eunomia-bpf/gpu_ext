# Raw-map preflight 575-01 failure review

This preflight is failed evidence and must not seed a formal campaign. It
stopped in the first `small` cell because the instrumented target's JSON stdout
and bpftime diagnostics shared one file descriptor. The diagnostic
`Exiting CUDA watcher thread` split one `cuda_truth` JSON line, so the exact
target-stream validator rejected the cell.

The retained probe log independently reports 768 committed and collected raw
records, 768 aggregate callbacks, zero drops, zero malformed records, and zero
aggregate mismatches. Those values diagnose the logging fault but do not turn
the failed cell into accepted evidence. The campaign must be rerun from a new
directory after separating machine-readable stdout from diagnostics.

Owned cleanup completed: no process-group survivor, the exact private shared
segment was removed, the post-cell GPU was idle, UVM refcount was zero,
struct_ops was empty, and no recorded kernel anomaly appeared.
