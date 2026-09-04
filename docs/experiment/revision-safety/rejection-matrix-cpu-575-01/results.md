# CPU safety rejection matrix results

Date: 2026-09-04

Run status: **valid for the stated CPU-only scope**. The tested hypothesis is
supported: all 16 selected unsafe/control pairs produced the expected opposite
outcomes, and both existing regression suites passed.

## Outcomes

| Enforcement layer | Selected pairs | Result |
| --- | ---: | --- |
| Userspace PREVAIL base checks | stack bounds; termination | 2 unsafe rejected, 2 matched controls accepted |
| GPU SIMT pass, existing | varying branch, map key, shared-map value, atomic target; prohibited synchronization helper | 5 unsafe rejected, 5 matched controls accepted |
| GPU SIMT pass, additional | direct shared-map store, varying helper output, map-update flags, host-bridge payload | 4 unsafe rejected, 4 matched controls accepted |
| Driver-shared transition validation | range, stale snapshot, conflicting request, invalid prefetch action, PMM identity | 5 invalid/stale/conflicting requests took the specified preserve/native route; 5 controls applied |

The additional SIMT runner reports four pairs through the public
`verify_gpu_program` entry point. The existing selector reports 28 assertions
in five test cases (seven unsafe/control pairs). The selected transition runner
reports five pairs and 26 assertions; the complete production-header regression
reports 12 cases and 145 assertions. Exact stdout is under [`execution/raw`](execution/raw/).

Two preflight defects were repaired before freezing the retained execution.
The first direct-store form kept its map pointer in a caller-saved register, so
PREVAIL rejected the store before the intended SIMT check. The corrected pair
uses callee-saved `r6`. The first update-flags control exposed uninitialized
bytes through an over-wide map-value descriptor; the retained pair uses the
actual eight-byte initialized value. These repairs remove unrelated rejection
causes and do not change the expected SIMT rules.

## What this demonstrates

- The userspace GPU-verifier path executes base bounds/termination checks before
  its SIMT-specific uniformity checks.
- Four additional implemented SIMT rules reject lane-varying shared side
  effects while preserving closely matched warp-uniform programs.
- The production-shared transition header distinguishes rejection from stale
  or conflicting no-ops and maps each result to an operation-specific native,
  preserve, or commit effect.

This is supporting correctness evidence, not a performance experiment. The
transition program invokes the shared validation functions but not the native
GPU actuator.

## Explicit exclusions

- **Linux host verifier and kfunc/BTF admission:** not executed. The current
  process had `CapEff: 0000000000000000`; a real test requires privileged
  `BPF_PROG_LOAD` against the loaded driver's BTF. Compilation alone would be a
  proxy, so it was excluded. Existing privileged load-only evidence remains a
  separate result.
- **Global synchronization:** the tested rule is specifically rejection of the
  registered `bpf_gpu_membar` helper as a prohibited SIMT helper. It does not
  prove rejection of every possible GPU synchronization instruction.
- **Deployment:** no strict-mode attach, GPU execution, BPF load, driver reload,
  native scheduler constructor, or UVM transition was performed here.

The appropriate paper claim is that executable paired cases cover base-program
safety, implemented SIMT side-effect rules, and operation-specific transition
validation. The result does not justify calling the GPU userspace verifier the
Linux kernel verifier or claiming universal full-stack safety.
