# POD phase preflight attempt 01

- Status: `failed-before-GPU`
- Date: 2026-09-03
- Command: `python3 run_phase_study.py preflight --output raw/phase-preflight-575-01 --ptx build/ptx-runtime-01`
- Failure: the inherited coordinator attempted to open
  `/tmp/gpubpf-revision-gpu0.lock` with `r+`; the pre-created lock is owned by
  root and mode `0644`, so the unprivileged process received
  `PermissionError: [Errno 13] Permission denied`.

The runner wrote only `manifest.json` during source/runtime inventory. It did
not acquire either lease or launch a CUDA process. The immediate post-failure
check found the RTX 5090 idle, NVIDIA driver 575.57.08, UVM reference count
zero, and no attached struct-ops object. Neither lock file was modified,
recreated, or removed.

This attempt contains no timing or correctness sample and is excluded from
all analysis. A retry must use a new output directory and sufficient
privilege to open the existing coordination locks; it must not alter their
ownership, mode, or inode.
