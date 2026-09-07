# Current GDS compatibility — driver-bridge forward-port (20260907)

## Result

Bounded forward-port of `driver-bridge-v1.patch` onto the GDS target tree
(`/home/yunwei37/workspace/gpu/gpu_ext-kernel-575-gds` at `8d00e964`,
"feat(uvm): add GPU storage decision hook") is complete:

- New patch: `workloads/stale-state-575/driver-bridge-gds-20260907.patch`
- Verification: `git apply --check` of the complete new patch passes on the
  unchanged target tree (exit 0). The root independently repeated the plain
  `git apply --check` successfully; the target tree remains clean.
- The old patch failed `git apply --check` at
  `kernel-open/nvidia-uvm/uvm_bpf_struct_ops.c:56` (hunks at old lines
  56/115/123/161/322, including BTF-set changes); every other section of the
  old patch already located its context in the GDS tree (with offsets).
  Their substantive changes are retained, with `index` lines omitted.

## Method

- Temporary directory `/tmp/opencode/gds-fwd-20260907/work` (no git worktree);
  copied only the source files the old patch touches, plus a pristine copy of
  `uvm_bpf_struct_ops.c` as the diff base.
- Applied the old patch with `git apply --exclude` of `uvm_bpf_struct_ops.c`;
  hand-edited the temp copy of that file to integrate the v1 changes with the
  GDS storage additions. Generated an ordinary `diff -u` for that file and
  substituted it as the patch section; all other sections retained unchanged
  with `index` lines omitted throughout the new patch.
- No driver tree edits, no modules, builds, GPU runs, benchmarks, commits, or
  hashes. Source preparation only.

## uvm_bpf_struct_ops.c adaptation (the only reworked section)

- Kept ALL GDS storage additions intact: `struct gpu_storage_ops`
  (storage-decide ABI), `uvm_storage_ops` global,
  `gpu_storage_ops__gpu_storage_decide` stub,
  `__bpf_ops_gpu_storage_ops` CFI stubs, `bpf_gpu_storage_record` kfunc,
  `gpu_storage_ops_reg/unreg`, the `gpu_storage_ops_struct_ops` bpf_struct_ops
  definition, its init registration, and the
  `uvm_bpf_call_gpu_storage_decide` wrapper. The storage cmd 82/136-byte ABI
  lives in headers outside this file and is untouched.
- `struct gpu_mem_ops` gains the versioned stale-state hook member exactly as
  in v1 (with the `offsetof(...) == 6 * sizeof(void *)` and
  `sizeof(...) == 7 * sizeof(void *)` static_asserts; the asserts scope only
  `gpu_mem_ops`, not the separate storage struct), plus the v1 stub, CFI-stub
  member, `bpf_gpu_stale_state_v1_request` kfunc, and the
  `uvm_bpf_call_gpu_stale_state_v1` RCU dispatch wrapper.
- BTF set split per v1, with storage kept where it belongs:
  - `uvm_bpf_struct_ops_kfunc_ids_set` (STRUCT_OPS hook): strstr,
    set_prefetch_region, request_reorder, `bpf_gpu_storage_record` (kept, both
    hooks), and the trusted `bpf_gpu_stale_state_v1_request` (STRUCT_OPS only).
  - `uvm_bpf_kprobe_kfunc_ids_set` (KPROBE hook): same four kfuncs without the
    stale trusted setter, matching the v1 split policy that only gpu_mem_ops
    callbacks get a trusted stale decision context.
- init() keeps the GDS `gpu_storage_ops` registration and routes every
  failure path through v1's cleanup: stale-state init and the trigger proc
  file are created before any BTF/struct_ops publication;
  `error_proc` removes the trigger file, `error_stale_state` tears down the
  stale proc/state; both struct_ops registration failures (mem and storage)
  goto `error_proc`. exit() calls `uvm_stale_state_v1_exit()` first, as in v1.
- All new files (nv-gpu-stale-state-v1.h, uvm_stale_state_v1.c/.h,
  tests/stale-state-v1/) are carried over unchanged from v1.

## Pending (not done here, by design)

- Formal 21-cells campaign and the lifecycle exact-GDS restore remain pending.
- No readiness/import/engagement runs were executed; a later step must apply
  this patch to the driver tree to continue the stale-state workstream.
