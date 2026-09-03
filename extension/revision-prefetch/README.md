# Q2: invalid initial-prefetch action, 575 preparation

Status: **actual 575 loader admission failed; all three functional controls
remain unexecuted.** [Attempt01](../../docs/experiment/revision-safety/prefetch-invalid-575-01/results.md)
stopped before releasing the initialized target because `range_enter`'s
structure-returning target is unsupported by fentry (`-EINVAL`, zero processed
instructions). Cleanup completed without recorded failure; this supplies no
invalid99 callback, native-fallback, mask, or numerical-correctness evidence.

The three synthetic record-gate tests and first independent single-CPU build passed on
2026-09-03; see the
[execution record](../../docs/experiment/revision-safety/prefetch-invalid-cpu-575-01-EXpHQx/execution.md).
Two additional cleanup regressions subsequently passed (five synthetic tests
total); [attempt02](../../docs/experiment/revision-safety/prefetch-invalid-cpu-575-02-h6EPh2/execution.md)
retains both successful runs. This supports the existing Q2/RQ2 containment
claim, not a new performance claim.
The hypothesis is that an actual callback return of 99, with a legal empty
region request, executes native tree traversal after validation and leaves
the managed-memory target numerically correct.

## Existing observation seam and evidence boundary

The 575 source is `../gpu_ext-kernel-575/` relative to the repository root.
Relevant production locations are
`kernel-open/nvidia-uvm/uvm_bpf_struct_ops.c:346`,
`kernel-open/nvidia-uvm/uvm_perf_prefetch.c:103,368`, and
`kernel-open/common/inc/nv-gpu-transition-validator.h:271,296,336`.
Action 99 is rejected; `(0,0)` is legal, deliberately isolating action rejection.

Existing 575 build inspection found `compute_prefetch_region` inlined, but
`compute_prefetch_mask` and `uvm_perf_prefetch_bitmap_tree_iter_get_range` retain
BTF, function-entry instrumentation, and actual calls from the native branch.
Attempt01 demonstrated that these properties do **not** establish live
attachment admission: the `get_range` STRUCT return was rejected during object
load, before the attachment loop. The retained fixture uses named fentry/fexit
attachments only, no instruction offsets, notrace bypass, or driver patch.
It correctly stopped before target release; do not retry unchanged inputs or
replace this failed observation with a callback counter.

The observer correlates one active `compute_prefetch_mask` frame per task,
checks the same bitmap-tree identity at every wrapper/range event, and counts
range calls only **after the actual wrapper return** and before the next
wrapper entry or mask return. Under this fixture's actions, post-return range
calls demonstrate execution of the existing native traversal: BYPASS has no
traversal; ENTER_LOOP is never submitted and its wrapper is separately watched.
The actual final output mask is read at mask-function exit, with bounds checked
and bounded samples retained. This is the compute function's output, before its
caller removes demanded/thrashing pages, not the final migration mask. It is
not reconstructed from the requested region.
If admitted, this would observe native-branch execution, not the local validator
enum directly, and would not establish completed DMA or physical PCIe bytes.
Attempt01 produced none of these observations.
Task/tree identity prevents cross-frame association but is not target-TGID or
VA-space attribution. Counters describe the globally observed exclusive window;
do not claim that this observer independently attributes every callback to the
managed target. A foreign client invalidates the assumed exclusive control.

## Fixed controls and acceptance

| Mode | Attached policy | Required wrapper result | Required per-decision traversal |
| --- | --- | --- | --- |
| `native` | none; observers only | 0, no request | at least one real range call |
| `bypass` | legal `(0,0)` | 1, attempted, no conflict | zero; every final mask empty |
| `invalid99` | legal `(0,0)` | 99, attempted, no conflict | at least one real range call |

One fresh 8 GiB / 64 KiB managed target per mode, fixed order native, bypass,
invalid99. This is a bounded functional control, not repeated performance data.
The existing `uvm_fault_stream --wait-for-monitor` initializes every expected
GPU-read region on the CPU before observers/policy attach, excluding the
preferred-location first-touch shortcut. Each run checks all 131,072 values.
Do not require identical native/invalid masks or callback counts across fresh
processes: GPU fault batching can differ.

All retained runs require nonzero matched wrapper decisions; policy calls equal
wrapper returns in the two BPF modes and zero in native; every decision satisfies
the table; entry/exit and decision totals reconcile; no iterator callback,
map failure, nesting, missing frame, identity mismatch, read failure, out-of-bound
mask, or unexpected action/request; empty active-frame map after target exit;
zero program recursion misses; all six observer links actually attached.
There is no ring buffer to drop records: aggregate counters cover all observed
events, while mask samples are explicitly illustrative. A failed attachment,
nonzero error, missing final metrics, timeout, or incomplete cleanup invalidates
the cell and stops the three-cell sequence.
An owned `BPF_ENABLE_STATS` descriptor enables actual program run-count and
recursion-miss accounting for this functional run and is closed on loader exit.

## Interface and next executable steps

Only this new directory is written: `fixture.bpf.c`, `fixture.h`, `loader.c`,
`Makefile`, `run_safety.py`, `test_offline.py`, and this runbook. Existing driver, legal control,
target, shared safety helpers, and old 610 evidence stay unchanged. The loader
follows `../prefetch_none_revision.c`; the runner reuses the project leases,
owned process-group cleanup and safety checks, rather than adding a new control
framework. Loader modes are fixed; there is no arbitrary invalid-action CLI.

The admitted CPU preparation ran these commands successfully; repeat only in a
new coordinated CPU window if the source changes:

```sh
taskset -c 17 python3 -B extension/revision-prefetch/test_offline.py
taskset -c 17 make -C extension/revision-prefetch -j1
```

The following attempt01 command was run and failed at loader admission. It is
retained for provenance, **not a ready next-run command**; do not reuse its
output directory:

```sh
sudo -n python3 extension/revision-prefetch/run_safety.py \
  --output docs/experiment/revision-safety/prefetch-invalid-575-01
```

The coordinator must have CPUs 8–16 available; only the target is pinned to
8–15, telemetry to 16. Require the recorded 575.57.08 custom ABI, Linux
6.15.11-061511-generic, prefetch enabled, both existing leases, 400 W, no
foreign compute process/policy, and no preexisting/new kernel abnormality.
Capture actual boot/version/BTF evidence, argv, paths/sizes, attach IDs,
continuous telemetry, final metrics and full target readback. No module change,
global pkill, global detach, or shared-memory deletion is part of these commands.
Target group must be gone before detaching its policy/observers; preserve those
processes and fail if an owned target survives bounded cleanup.
Use the existing lock-file inodes with the root coordinator; never chmod or
replace locks to obtain admission. The first SIGINT/TERM aborts and ignores
repeated interrupts until owned cleanup and evidence writing finish. Before/after
path, size, and mtime inventories must match; no file-content digest is used.
Cleanup requires both a reaped leader and an empty owned process group, retaining
the prior 8/5/5-second INT/TERM/KILL grace periods. Every monitor is attempted
even when another fails; each PID, return code and error is retained, and any
failure rejects the cell. The CPU regressions do not prove real signal behavior.

Expected GPU occupation is three short fault kernels plus CPU initialization
and bounded admission/cleanup. Each cell allows 60 s to reach the initialization
pause, 30 s for observer setup, and 60 s after release; these are independent
failure bounds, not a measured duration or a performance target.
The existing UVM-tools migration monitor is an optional later downstream
cross-check; its successful 610 records do not prove 575 compatibility.

The next step is a **separately, independently reviewed** read-only, noinline
Kbuild diagnostic with a void return and a pointer to a driver-filled const
context. Emit phases after initial-effect selection and completed native
traversal, carrying copied scalar action/result/effect/bounds/selected region
and invocation correlation. It must not change policy dispatch or actuation.
That would touch only
`uvm_bpf_struct_ops.h/.c` and `uvm_perf_prefetch.c`, retain actuator semantics,
and require a new driver build/reload admission. It is not implemented here;
the existing type gate is not bypassed, and the Q2 live-prefetch requirement
remains open.
