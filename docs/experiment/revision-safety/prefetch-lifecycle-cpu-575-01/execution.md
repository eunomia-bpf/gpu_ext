# Q2 UVM lifecycle CPU validation

Date: 2026-09-03. This is CPU/read-only validation of the lifecycle
coordinator. No module was loaded or removed, no service was changed, no CUDA
workload ran, and this is not a live Q2 result.

## Results

- `taskset -c 17 python3 -B extension/revision-prefetch/test_lifecycle.py`:
  33 tests passed in 0.210 seconds.
- `taskset -c 17 python3 -B extension/revision-prefetch/test_offline.py`:
  18 tests passed in 0.088 seconds.
- `git diff --check` on the lifecycle runner, tests, and plan: passed.
- A source scan found no forced-removal, automatic module lookup, dependency
  regeneration, reboot, or module-install command in the runner or its tests.

The lifecycle tests include executable recovery mocks, not only pure outcome
predicates. With every recorder transition forced to fail, the physical mock
sequence still removed the candidate, inserted the old UVM, then started
nvidia-persistenced and GDM. Separate tests prove that candidate-removal,
old-insertion, and pre-service core-continuity failures prevent unsafe later
steps. Publication tests prove that the lease closes before the first possible
complete record, the first durable record is incomplete, both signal snapshots
are honored, and a failed final complete write cannot promote the hidden
summary candidate.

The lease tests use existing 0444/0644 temporary inodes, prove that acquisition
uses read-only/no-create descriptors, prove a second exclusive acquisition is
rejected, and prove that a partial two-lock failure releases the first lock.

## Read-only artifact checks

The candidate
`/home/yunwei37/workspace/gpu/gpu_ext-kernel-575/kernel-open/nvidia-uvm.ko`
was read as a 61,919,280-byte `nvidia_uvm` module with version 575.57.08,
6.15.11 vermagic, dependency `nvidia`, 53 parameters, the generic gpubpf UVM
ABI, and the diagnostic ABI. The admitted old stage was read as 61,914,016
bytes with the same version, vermagic, dependency, parameter inventory, and
generic ABI, without the diagnostic. Its 53 parameter names exactly match the
live sysfs inventory, whose prefetch-enable value is 1.

The live NVIDIA core remained loaded at version 575.57.08. Its BTF file was
125,993 bytes and exposed the required scheduling interface names. These
checks validate admission code against the current files; they do not replace
the coordinator's fresh under-lease checks at a future live launch.

## Independent root-agent rerun and review

The primary agent independently reran the lifecycle suite (33/33) and the
observer suite (18/18); the raw outputs are `root-lifecycle-tests.log` and
`root-observer-tests.log`. It also ran the CLI help path and a fresh, read-only
admission pass as UID 1000. `root-readonly-preflight.json` records a matching
6.15.11/575.57.08 runtime, 53 matching parameters, the exact candidate and
restore roles, both original services active/running/successful, only remote
SSH sessions plus the local GDM greeter, an idle 400 W GPU, zero UVM references,
empty struct-ops state, no current-boot kernel abnormality, and absent stage
and result paths.

An independent OpenCode review used direct attachments with snapshots,
sharing, tools, edits, and external tasks disabled. Its initial BTF-permission
concern was disproved by the recorded host state and unprivileged live read;
the follow-up retracted it and returned `READY`. The complete review evidence
is under `opencode-lifecycle-review-01/`. This verdict is runner readiness, not
live Q2 evidence.
