# Q2 driver safety tests: 575 results and remaining gaps

Updated 2026-09-03 from the coordinator's
[completed execution](sched-load-575-02/execution.json) and all seven verifier
logs. Both commands exited zero: the load-only runner recorded **7 attempts,
4 admissions, 3 rejections, 7 passes**; the production-shared CPU validator
recorded **12 cases and 145 assertions**. Before/after snapshots show 400 W,
UVM references zero, no compute clients or struct_ops maps/links, and no
reported Xids or abnormal kernel messages. No policy was attached.

The four positive logs report `load_error=0`; the three negative logs report
`load_error=-13` at actual stores to scheduler input offset 16, scheduler
private offset 32, and PMM private offset 56. These are verifier-load results,
not native transition execution. The earlier
[01 admission attempt](sched-load-575-01/admission.md) stopped on shared-lock
file permissions before either test and is retained, not counted as a run.
This document update only reads those records; it performs no new test or
driver operation and does not admit work alongside the current GPU owner.

## Completed scope and remaining live gaps

The existing scheduler/PMM verifier fixtures passed without a rebuild or
module change. Their source, stored ELF instructions, and current kernel BTF
match the covered ABI. **Native scheduler-init rejection/commit tests and invalid-prefetch live
fallback tests do not have a complete existing runnable fixture/runner.**
Do not substitute a successful workload or load-only result for those tests.

| Requirement | Existing path and evidence | Missing live evidence / disposition |
| --- | --- | --- |
| Scheduler verifier admission/rejection | `extension/revision_sched_verifier` and five `revision_sched_*.bpf.o`; built inputs read a 32-byte context, setters request 100 us and explicit LOW=0, negative stores are at offsets 16 and 32 | **Passed load-only in 02:** current runner also includes two PMM fixtures; observed **7 attempts, 4 admissions, 3 rejections, 7 passes**, not the historical 5/3/2/5 |
| Shared scheduler/prefetch validator semantics | `../gpu_ext-kernel-575/kernel-open/tests/transition-validator/transition_validator_test`; 12 cases/145 assertions cover scheduler identity/phase, minimum, repeat/conflict, independent fields, and prefetch action/range/translation | **Fresh CPU rerun passed in 02.** It does not execute the native constructor, resource setters, CUDA, or UVM fault path |
| Native scheduler initialization commit/rejection | Production `../gpu_ext-kernel-575/src/nvidia/src/kernel/gpu/fifo/kernel_channel_group_api.c:329`; `extension/gpu_sched_set_timeslices` supplies ordinary policy requests and callback statistics | No ready bounded negative/positive init-commit matrix. Need actual post-validation/native-setter observations for rejected and independently accepted fields; setter counters alone report recording, not commit |
| Invalid initial/iterator prefetch output | Production `../gpu_ext-kernel-575/kernel-open/nvidia-uvm/uvm_perf_prefetch.c:100`; same production header is CPU-tested | No invalid-action/range/conflict BPF variant and no live native-fallback oracle found in the existing extension/test paths. Cannot give a truthful ready-to-run completion command |
| Valid prefetch control only | `extension/prefetch_none_revision` and its BPF object use the 24-byte decision ABI and request legal `(0,0)` with BYPASS | Available control, but no invalid output and no engagement counter; it runs until signaled. It cannot close invalid-prefetch fallback by itself |
| Persistent timeslice control | `extension/.output/gpu_sched_timeslice_control_cpu_test` exercises the actual callback with mock helpers; retained context canaries checked 2048 values and 17 negatives | Different boundary (`on_timeslice_control`), not the missing initialization transition test. Existing canaries should not be repeated merely to relabel them as init coverage |

The historical scheduler report predates adding PMM objects to the runner.
Current source `extension/revision_sched_verifier.c:22` executes four positive
controls before any of three negatives, accepts only `-EACCES` as the expected
direct-write rejection, saves individual verifier logs, and only calls
`bpf_object__load()`/close, never `bpf_map__attach_struct_ops()`. `-m` selects
**PMM-only** (2/1/1/2), not scheduler-only. There is no scheduler-only switch;
use the existing seven-object invocation rather than changing the runner.

## Current ABI and build identity

Read-only checks confirm `uname -r = 6.15.11-061511-generic`; both loaded
module version files report `575.57.08`. The
[restoration record](../driver-575-linux-6.15-runtime.md) identifies the loaded
custom revision as `849ea75d`. The sibling source remains at `849ea75d`, branch
`test-sched`; only its existing untracked test executables were present.

The actual `/sys/kernel/btf/nvidia` dump contains:

- `nv_gpu_task_init_ctx`, 32 bytes; decision wrapper, 56 bytes.
- `nv_gpu_sched_ops`, 32 bytes, with the three original callbacks plus
  `on_timeslice_control`.
- `bpf_nv_gpu_set_timeslice`, `bpf_nv_gpu_set_interleave`, and
  `bpf_nv_gpu_override_timeslice`.

The loaded UVM BTF contains the 24-byte prefetch decision, the 72-byte PMM
decision, `bpf_gpu_request_reorder`, and the three-argument
`bpf_gpu_set_prefetch_region(decision, first, outer)`; no
`bpf_gpu_migrate_range` was found. This checks actual types/functions, not
merely the existence of BTF files.

The older scheduler fixture objects declare only the original three-member
`nv_gpu_sched_ops`. This is not by itself a mismatch: the linked libbpf path
allocates zeroed kernel-sized struct_ops data and maps object members by name
(`libbpf/src/libbpf.c:1135`), leaving the new optional callback null. The 32-byte
input and relevant member offsets match current BTF. All four positive load
controls subsequently passed in 02; that admission result is retained
separately from this source inspection.

| Artifact root | Inspected existing module identity | Use with current runtime |
| --- | --- | --- |
| `../gpu_ext-kernel-575/kernel-open` | Core 30,120,600 bytes; UVM 61,914,016 bytes; version 575.57.08 and vermagic 6.15.11-061511-generic | Matching build family; already staged/loaded according to the restoration record. No reload needed for the seven load fixtures |
| `/opt/gpubpf/modules/575.57.08/gpreempt-849ea75d-6.15.11` | Same listed core/UVM sizes and recorded staging identity | Existing experiment staging; do not overwrite or reload for this audit |
| `../gpu_ext-kernel-610/kernel-open` | Both module versions 610.43.02; vermagic 6.15.11-061511-generic; source `c4fd5655` | **Not interchangeable with loaded 575**, despite matching kernel version |
| `kernel-module/nvidia-module/kernel-open` | Source is `c4fd5655` (610 port), but its existing core ELF reports **575.57.08 / 6.14.0-37-generic** | Stale build artifact in a differently checked-out source tree; do not infer binary identity from the directory or source commit |

575 calls `kfifoRunlistGetMinTimeSlice_HAL()` for the scheduler validator;
610 passes `0U` because its native setter removed the minimum guard. CPU tests
exercise the explicit minimum argument, not the live GPU's native minimum.
An out-of-range outcome on one driver is therefore not automatically the same
test on the other. The current four-callback process-policy object also uses
the 575 persistent-control extension; it is not an unmodified 610 artifact.

## Existing fixture inventory

All paths below were inspected, not rebuilt:

| Path under `extension/` | Bytes |
| --- | ---: |
| `revision_sched_verifier` | 1,520,168 |
| `.output/revision_sched_immutable_read.bpf.o` | 2,744 |
| `.output/revision_sched_timeslice_setter.bpf.o` | 2,680 |
| `.output/revision_sched_interleave_low_setter.bpf.o` | 2,688 |
| `.output/revision_sched_input_write.bpf.o` | 2,416 |
| `.output/revision_sched_hidden_write.bpf.o` | 2,424 |
| `.output/revision_pmm_reorder_setter.bpf.o` | 4,376 |
| `.output/revision_pmm_hidden_write.bpf.o` | 4,128 |

The loader's existing executable exposes the current `-m` option and seven-
fixture summary format. ELF relocation targets match the named current
kfuncs. Its runtime dependencies are ordinary libelf, zlib and libc, not CUDA.
The 575 CPU transition-test executable is 16,864 bytes and newer than its
unchanged source/header; its fresh 12/145 result is retained in 02.

## Repeat-only commands and remaining coordinator work

The completed commands and their stdout are in `sched-load-575-02/execution.json`.
No rerun is needed to establish the load-only/CPU results above. If a later
change requires repetition, use a new directory and the same admission rules:

1. Wait for the current GPU owner, obtain the existing GPU/struct_ops leases, and apply the normal
   preflight: 400 W, no foreign compute client or policy, UVM reference count
   zero, no new driver/kernel errors. This audit did not grant that admission.
2. Optionally rerun the existing CPU executable below; do not rebuild merely
   to repeat its 12/145 evidence.
3. Run the seven load-only fixtures into a **new** directory. Preserve its
   console/exit code and seven per-fixture logs; require exact 7/4/3/7 totals
   and no attached/pinned state afterward. The runner overwrites log files
   when a directory already exists, hence the explicit new-directory guard.
4. Keep scheduler-init and invalid-prefetch live rows open. No new runner or
   policy was written here. Arrange their minimal missing fixtures/oracles
   before claiming completion; do not launch ad hoc unbounded policies as a
   substitute.

From `/home/yunwei37/workspace/gpu/gpu_ext`, after admission, run commands
separately and stop on any failure:

```bash
taskset -c 17 ../gpu_ext-kernel-575/kernel-open/tests/transition-validator/transition_validator_test
test ! -e docs/experiment/revision-safety/sched-load-575-03
mkdir docs/experiment/revision-safety/sched-load-575-03
sudo -n taskset -c 17 ./extension/revision_sched_verifier \
  -d extension/.output -l docs/experiment/revision-safety/sched-load-575-03 \
  > docs/experiment/revision-safety/sched-load-575-03/console.txt 2>&1
```

No command above installs or replaces modules, attaches a policy, invokes a
PMM ioctl, or runs the GPU. The unrelated `revision_pmm_ioctl` and
`uprobe_preempt_multi` artifacts are deliberately excluded.

## Why the two remaining live tests cannot be substituted

`gpu_sched_set_timeslices` calls each setter once; its CLI rejects timeslice
zero and interleave above two. It cannot express the required conflict or
invalid-interleave tests. Its `timeslice_mod`/`interleave_mod` counters count
setter success before native commit. `gpu_sched_trace` reports callback and
bind observations, not every native setter result or rollback. Retained
canaries already showed CUDA overwriting initialization requests with 2048 us
before the first kernel; later persistent-control success is a separate claim.

For prefetch, `compute_prefetch_region()` routes invalid initial actions or
regions to native behavior and invalid iterator selections to ignore. Its
checked translation happens before narrowing. Those are source-backed paths,
not observed fallback outcomes. The existing `UVM_TEST_SET_PAGE_PREFETCH_POLICY`
ioctl only enables/disables native prefetch; it does not inject a malformed BPF
decision. The only dedicated BPF-transition ioctl found is PMM-specific.
Closing the prefetch row requires actual invalid callback output, a matched
valid/native control, observed validation/fallback selection, full output
correctness, and owned cleanup. None is manufactured from the old 610 PMM log.
