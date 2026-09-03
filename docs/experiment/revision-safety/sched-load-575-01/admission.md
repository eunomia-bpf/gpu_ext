# Incomplete admission, 2026-09-03

The unprivileged coordinator failed opening the existing shared GPU lease
with `PermissionError` before collecting a GPU safety snapshot or executing
either test. No BPF object was loaded or attached. The empty output directory
is retained; the privileged retry uses a new `sched-load-575-02` directory.
