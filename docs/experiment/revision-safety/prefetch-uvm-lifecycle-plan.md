# Q2 UVM-only lifecycle coordinator

Status: CPU implementation and failure-gate tests only; this document does not
claim that the module swap or the three live controls ran.

## Fixed artifacts and scope

The coordinator is
`extension/revision-prefetch/run_lifecycle.py`. It holds the existing GPU 0 and
struct-ops leases once across staging, the UVM-only swap, direct calls to
`run_safety.run_cell` for `native`, `bypass`, and `invalid99`, and physical
restoration. It never calls `run_safety.main`, so it does not attempt to acquire
either lease a second time. Because the established root-owned lease files are
world-readable but not writable, it opens the verified existing inodes read
only with no-follow/no-create flags and takes the same exclusive advisory lock;
partial acquisition closes every descriptor.

The only admitted invocation is:

```sh
taskset -c 0-17 python3 -B extension/revision-prefetch/run_lifecycle.py \
  --candidate /home/yunwei37/workspace/gpu/gpu_ext-kernel-575/kernel-open/nvidia-uvm.ko \
  --restore /opt/gpubpf/modules/575.57.08/gpreempt-849ea75d-6.15.11/nvidia-uvm.ko \
  --stage /opt/gpubpf/modules/575.57.08/prefetch-diagnostic-0c109956-6.15.11 \
  --output /home/yunwei37/workspace/gpu/gpu_ext/docs/experiment/revision-safety/prefetch-invalid-575-02
```

Both the stage directory and output directory must be absent. The candidate is
expected to be 61,919,280 bytes and the old staged UVM is expected to be
61,914,016 bytes; the runner derives and records the actual values and rejects
any change. It validates ordinary file identity and metadata, module name,
version, vermagic, dependency, complete parameter-name inventory, BTF resource
ABI, and the candidate-only diagnostic ABI. Staging uses a new directory and a
lossless byte comparison. No content digest is generated or used.

## Forward gates

Before stopping a service, the coordinator records the boot, kernel, full live
NVIDIA core BTF dump and file identity, UVM BTF/interface, every UVM sysfs
parameter, GPU/kernel safety snapshot, and the initial states of GDM and
nvidia-persistenced. It records every login session and rejects any local
non-greeter session; remote SSH sessions and the GDM greeter remain admissible.
The complete session gate is repeated and persisted immediately before the
actual GDM stop, closing the staging-to-stop admission gap.

Only services that began active/running/successful are stopped. Immediately
before the ordinary UVM removal, two bounded idle snapshots must show a 400 W
575.57.08 GPU, no compute application, no residual utilization, zero UVM
references, empty struct-ops maps and links, the same boot, and no new kernel
abnormality relative to admission. `fuser` must return the unambiguous silent
no-holder result for both UVM device nodes. A queued signal is checked again
after these gates and before removal.

The core, modeset, DRM, packages, boot files, and module search path are never
changed. The runner uses only an ordinary bounded removal of `nvidia_uvm` and
an ordinary insertion of an explicit module path. Every initial UVM parameter
is passed as an argv item; a sysfs `(null)` char pointer is omitted and must
normalize back to exactly `(null)` after load. Candidate BTF, interface, and
the complete parameter dictionary must match before the first cell is released.

## Failure and restoration

SIGINT and SIGTERM are queued rather than thrown through child cleanup. Every
external command has its own process group and a bounded timeout. Forward
execution checks queued cancellation before the old removal and again between
old removal and candidate insertion. Recovery deliberately ignores queued
cancellation until physical restoration and evidence persistence finish.

The `finally` path classifies whether the original UVM remains loaded, no UVM
is loaded, or the candidate was inserted. Before removing a candidate it
repeats the operational idle, zero-reference, empty-struct-ops, and no-holder
gates. Existing kernel abnormalities make the experiment fail but do not make
an otherwise idle candidate unsafe to remove. The old UVM is then inserted
from the exact admitted stage with the captured parameters and must pass its
ABI, parameter, idle, boot, and core-continuity checks before either service is
started. Originally inactive services remain inactive; originally active
services must return to their original substate with a successful result and
unchanged enablement.

`lifecycle.json` records each recovery attempt separately. Candidate-removal,
old-insertion, service-restoration, final kernel-history, artifact-continuity,
core-BTF, no-holder, or lease-identity failure rejects the campaign. A
`summary.json` with `complete: true` is written only after all three cells pass,
the exact old runtime and services are restored, final safety passes, queued
signals are absent, and the leases close successfully. At publication, both
signals are blocked, an incomplete lifecycle record and hidden summary
candidate are written, and a second pending-signal snapshot is the commit
point. Only a clean snapshot permits the final complete lifecycle record and
summary promotion; original handlers are restored before signals are
unblocked. No force operation,
automatic module lookup, dependency regeneration, module installation, or
reboot path exists in the coordinator.

## CPU-only checks

`extension/revision-prefetch/test_lifecycle.py` covers parameter replay,
isolated command process groups, transient service polling, local-session
admission, strict `fuser` interpretation, candidate insertion failure,
post-load ABI failure, queued cell cancellation, candidate-removal failure,
old-module restoration failure, service stop/start failure, and the final
completion gate. These tests exercise construction and state-machine logic;
they are not live Q2 evidence.
