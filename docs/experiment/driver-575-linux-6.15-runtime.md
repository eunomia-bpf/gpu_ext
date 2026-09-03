# 575 runtime on Linux 6.15.11

The 2026-09-02 maintenance completed the user-authorized driver downgrade and
reboot. The new boot began at 15:30:29 PDT on Linux
`6.15.11-061511-generic`; `nvidia-smi` and the loaded module both report
`575.57.08`. SSH, GDM, and the 400 W power-limit service came back successfully.
The kernel was not downgraded. Secure Boot is disabled.

The package transaction explicitly installed 575.57.08 components and retained
`nvidia-prime=0.8.17.2`; the NVIDIA-repository driver metapackage was not used
because it conflicts with PRIME. The 24 branch/tool packages are held at their
installed versions. CUDA 12.9.1 and the container toolkit were not replaced.
The 23 exact rollback packages are retained at
`/var/cache/gpubpf-driver-rollback-610.43.02` (413,776,164 bytes).

## Separate boot and experiment modules

Stock 575 DKMS was explicitly built/installed only for the running 6.15.11
kernel; DKMS reports it installed, all five module versions/vermagic match,
and its initramfs was regenerated. The temporary `/etc/dkms/no-autoinstall`
marker used during the package transaction was removed. No 575 driver build
for the installed but unused 7.1.12 kernel is claimed.

The custom source is the sibling `gpu_ext-kernel-575` repository at commit
`28b1d30c`. It was built with:

```sh
make modules -j8 KERNEL_UNAME=6.15.11-061511-generic CC=/usr/bin/gcc-14
```

All five custom modules have matching version/vermagic and BTF. They are
staged separately at
`/opt/gpubpf/modules/575.57.08/6.15.11-061511-generic`; stock files under
`/lib/modules` were not overwritten. Custom core, modeset, DRM, and UVM were
loaded with ordinary `insmod` at 15:53 PDT after ordinary `rmmod` of stock
modules. Peermem is staged but is not needed for these experiments. A reboot
therefore returns to stock 575 rather than automatically loading gpubpf.

Live BTF exposes the six-member `gpu_mem_ops`, the memory request kfuncs, and
`nv_gpu_sched_ops`/`bpf_nv_gpu_preempt_tsg`. Kernel logs confirm both struct-ops
registrations. The post-load GPU was idle at 2 MiB/0%, UVM reference count zero,
and no NVIDIA Xid or kernel abnormality was present.

## Temporary maintenance state and required restoration

GDM and nvidia-persistenced were stopped for module replacement and exclusive
experiments. Two local k3s node labels were temporarily removed because
containerized device-plugin/DCGM processes retained UVM descriptors invisible
to ordinary host `fuser`. This paused four local device/monitoring DaemonSet
pods (NVIDIA device plugin, DCGM, node exporter, generic KVM device plugin).
k3s stayed active and all eight running business/storage pods stayed Running.

After the exclusive GPU experiments finish, restore the exact labels and
services; doing so during measurement would reintroduce GPU/module holders:

```sh
sudo -n k3s kubectl --kubeconfig /var/lib/rancher/k3s/agent/kubelet.kubeconfig \
  label node lab fleet.yunwei37.com/gpu=rtx5090 \
  monitoring.yunwei37.com/managed=true --overwrite
sudo systemctl start nvidia-persistenced.service gdm.service
```

## Real memory-policy canary

`workloads/uvm-policy-mechanism/results/safe-575-20260902-2q-03/` records a
successful 8 GiB / 64 KiB UVM correctness/engagement run: 131,072 values matched,
4,096 activation callbacks, 12,843 access callbacks, 16,939 head-reorder
requests, and zero request errors. Final UVM references, struct-ops maps/links,
compute processes, and kernel/Xid abnormal records were all zero. The
258.494 ms kernel duration is a canary observation, not a comparative result.
No promotion or eviction pressure was exercised, so this does not prove the
full two-segment policy or a performance benefit.

The preceding two runs are retained, including their failure evidence: output
was correct, but the old cleanup helper rejected a transient post-detach UVM
reference count before asynchronous BPF release completed. Attempt 02 records
the initial count of 2 and the following clean zero-count snapshot. The fix
waits only within the existing deadline when maps/links and compute apps are
empty; the final zero-reference and clean-kernel requirements are unchanged.
Runner changes present during these runs are committed alongside this note;
the recorded run Git revisions are not rewritten after the fact.

## Loaded GSP scheduling propagation and GPreempt transport

The sibling `gpu_ext-kernel-575` source now contains commit `363416c4` and the
subsequent GPreempt transport commit `e3bb2938` on `test-sched`.
The `e3bb2938` build was staged separately at
`/opt/gpubpf/modules/575.57.08/gpreempt-e3bb2938-6.15.11/` and loaded at
2026-09-03 00:35:47–49 UTC (September 2, 17:35 PDT). The historical experiments
above and the canceled generic MoE campaign used `28b1d30c`; they are not
retroactively attributed to this driver. The old staging directory and stock
files under `/lib/modules` remain unchanged.

The generic MoE experiment was stopped after one complete four-cell block
because its page-level stride/LFU policy is not the MoE-Infinity algorithm.
Block 2's UVM process was interrupted; its shutdown failure remains recorded.
After clients exited, GPU memory returned to 2 MiB and UVM references and
struct-ops attachments were zero, but utilization stayed at 100% and power near
104 W. The old kernel log also contains an RM unhandled-interrupt threshold
warning for IRQ 217 at 00:20:46 UTC during the BPF cell. No Xid does not mean
there was no abnormality. With both experiment leases held and no device-file
holders, ordinary bounded `rmmod`/`insmod` replaced core, modeset, DRM and UVM;
there was no forced unload, reboot or module installation. GPU state returned
to 2 MiB/0% and approximately 11 W. Module reload reset the power limit to
575 W despite the service still being active; it was explicitly restored to
the experiment's 400 W before further GPU checks.

Source inspection found that the open GSP-client scheduling HAL setters only
update host-side timeslice/interleave fields. Channel-group allocation does not
carry those fields to GSP. The patch forwards accepted policy requests through
the existing GSP control RPCs after successful remote allocation. Native,
default, rejected, and non-GSP paths do not gain policy RPCs. Constructor errors
explicitly free a successfully allocated remote object before local cleanup;
the original error is retained and a failed cleanup RPC is reported.

The CPU-only build succeeded using the command above, restricted to CPUs 8–15
while MoE owned the GPU. The resulting local `kernel-open/nvidia.ko` is
30,110,912 bytes after the transport addition, with version `575.57.08` and vermagic for
`6.15.11-061511-generic`. Existing transition-validator tests passed all 12 cases
and 145 assertions. These tests cover policy validation, **not execution of the
new GSP RPC or rollback path**. Independent source review found and checked the
explicit constructor-error cleanup. Known compiler-package and missing module
description warnings remain; the build had no errors.

Separate runtime canaries must check native
allocation/destruction, timeslice-only/interleave-only/combined RPC execution,
and controlled first/second-control failures with exactly one remote cleanup.
Reading the host shadow fields is not proof of hardware enforcement. Each run
must end with no UVM references, attached policy, compute client, or new Xid.
Performance requires a new campaign: this source fix does not revise the
preserved [negative driver-candidate measurements](../../workloads/xsched/driver-candidates-575-20260902.md).

The `e3bb2938` addition supplies an owned-context query and a narrow transport
for GPreempt's original 1 us / 1,000,000 us timeslice controls. It retains
ordinary control authorization, checks the calling process's retained owner
identity, and rejects ambiguous GR groups instead of choosing one. Its CPU
helper tests passed four cases and 110 assertions; actual ownership, query,
GSP control and two-context behavior remain pending hardware canaries. The
[userspace port](../../workloads/gpreempt/README.md) and
[BPF policy arm](../../extension/gpreempt-policy.md) use this ABI.

The separate official GDRCopy v2.5.2 `c91ad9f` dependency was built for
this kernel (534,936-byte `gdrdrv.ko`) and loaded at 00:37:22 UTC with persistent
mapping enabled. Its unmodified
conftest correctly detects the 6.15 `vm_flags_set` API. The earlier v2.5
535,584-byte build remains in its separate checkout. Neither build nor the BPF
hint-decision CPU tests prove GDR pin/map or GPU scheduling on the RTX 5090.
The new `/dev/gdrdrv` node is mode 0600, owned by experiment UID/GID 1000,
not world writable. The original finite smoke in
`workloads/gpreempt/raw/575-gdr-context-smoke-02/` accepted the owned-context
query and narrow timeslice request but failed at `gdr_pin_buffer`; the kernel
reported `nvidia_p2p_get_pages` returning `-22`. Cleanup left no compute process,
UVM reference, struct-ops attachment or new Xid. This is a real GDR failure,
not successful reproduction. Attempt 01 was rejected before CUDA execution
because the reload-reset power limit was still 575 W; both attempts are retained.
The independent official GDRCopy basic test then reported CUDA GPUDirect-RDMA
capability absent on this GPU and failed. A separate mapped-pinned-host flag
canary passed 64 exact roundtrips, but that is a different memory transport and
does not make the original GDR actuator available.

## Per-runlist identity and directly observable GSP completion

At 2026-09-03 01:01:22–23 UTC, a second ordinary bounded reload installed driver
revision `e7d46fa5` from the new separate directory
`/opt/gpubpf/modules/575.57.08/gpreempt-e7d46fa5-6.15.11/`. No boot-module files,
old staging files or module-probing restrictions were changed. GPU state was
2 MiB/0%, both struct-ops registrations succeeded, UVM and GDR references were
zero, and the power limit was again restored to 400 W. The existing GDR node
remained major 507, mode 0600 and UID/GID 1000.

This patch appends runlist/engine identity to the destroy context while retaining
the original TSG-ID offset. TSG hardware IDs are allocated within a per-runlist
namespace; a TSG-ID-only lookup can confuse graphics and copy-engine groups.
The matching BPF policy uses composite identity instead of accepting mismatches.
The patch also adds a Kbuild-instrumented, observation-only hook after the actual
GSP control RPC wait, carrying input value, transport status and valid firmware
status. Direct probing of the core RM `notrace` functions was rejected by Linux
6.15; that rejection is not bypassed. The completion hook cannot change policy
or return status and is not emitted for cache hits or pre-send rejection.

CPU transport/diagnostic tests passed 5 cases and 118 assertions, and transition
tests passed 12 cases and 145 assertions. The new core module is 30,112,976 bytes;
live NVIDIA BTF is 125,410 bytes and UVM BTF is 268,940 bytes. These build/load
facts do not yet prove firmware timeslice acceptance or physical scheduling
quantum; the new observer/context canary must supply its own runtime evidence.

## Persistent decisions at the authorized timeslice-control boundary

The `e7d46fa5` hardware canaries exposed an important failure: CUDA subsequently
submitted 2,048 us and overwrote both BPF initialization values before the
workload's first kernel. Initialization counters and a matching driver shadow
were therefore insufficient evidence of an effective policy. The original C
setter, called after context creation, passed this same final-GSP-value check.

At 2026-09-03 01:19:51–53 UTC, ordinary bounded reload installed `849ea75d`
from `/opt/gpubpf/modules/575.57.08/gpreempt-849ea75d-6.15.11/`. The new core
module is 30,120,600 bytes; the other four staged module sizes are unchanged.
Live core/UVM BTF sizes are 125,993/268,940 bytes. No boot files or prior staging
directories were overwritten. The first reload omitted DRM `modeset=1`, so
400 W immediately reverted to 575 W at idle; no experiment was admitted in that
state. Reloading only DRM with `modeset=1` restored the prior 2 MiB idle state
and persistent 400 W readings. An unnecessary GDR parameter was reported as
unknown and ignored; the official module explicitly reported that persistent
mapping was in use. `/dev/gdrdrv` retained major 507, mode 0600 and UID/GID 1000.

The appended optional scheduling callback runs inside the already authorized,
locked native RM timeslice-control handler. Its trusted kfunc records a bounded
decision; it does not call RM or sleep in BPF. The original physical RPC executes
after leaving the callback's RCU section, and retains its status propagation.
Incoming values outside the supported policy domain retain native behavior;
an invalid caller request is not made valid by the policy. Identity includes
RM handles, runlist, engine, hardware group, GPU and phase. Existing policies
without this callback preserve the native path. The GPreempt policy additionally
restricts decisions to the owned process and graphics-engine context roles.

The new original and BPF context canaries both passed on this loaded revision.
Each checked 2,048 integer outputs and 17 negative cases. Direct post-RPC
observations recorded LC 1,000,000 us and BE 1 us after the later CUDA controls;
unmarked native contexts retained 2,048 us. BPF reported two runtime overrides,
one per role, with zero identity, setter or mapping errors. Cleanup left zero
UVM references, no struct-ops attachment and no new Xid. This proves accepted,
persistent firmware requests for these canaries, not a measured scheduling
quantum or a full comparative performance result.

## Unexpected reboot and restoration of the verified runtime

The machine booted unexpectedly at 2026-09-03 01:37:41 UTC. The preceding boot's
journal ends at 01:36:29 UTC without a recorded shutdown, panic, Xid or OOM;
there was no new pstore record. The cause remains unknown. These observations
do not establish that a GPU experiment caused the reboot. As intended by the
temporary-loading setup, boot restored stock 575.57.08 on Linux 6.15.11, with
live core/UVM BTF sizes of 121,803/261,943 bytes rather than the custom revision.

Read-only holder and login-session checks found only the automatic GDM greeter
(UID 120, Xorg PID 2881) and nvidia-persistenced (root, PID 2002) holding NVIDIA
devices. GDM session c1 was explicitly classified as a greeter; user sessions
were remote SSH sessions, not a local desktop. Only gdm.service and
nvidia-persistenced.service were stopped to restore exclusive experiment state.
SSH, Codex, systemd-logind and the i915 display/framebuffer were not stopped.

At 01:46:10–12 UTC, ordinary bounded module removal and loading restored the
same `849ea75d` staging directory, including DRM `modeset=1` and UVM. Live
core/UVM BTF sizes returned to 125,993/268,940 bytes, and repeated idle checks
confirmed 400 W, 2 MiB and 0% GPU utilization. Official GDRCopy v2.5.2 was loaded
without the unnecessary parameter from the earlier attempt; its initialization
log confirmed persistent mapping by default. `/dev/gdrdrv` was restored with
major 507, mode 0600 and UID/GID 1000. No boot files or `/lib/modules` files were
changed, and no forced unload or further reboot was used.

Both original and BPF post-boot context canaries passed the actual GSP-completion
check, again observing LC 1,000,000 us and BE 1 us before the first kernel.
Their records are under `workloads/xsched/raw/` in
`gpreempt-context-original-849ea75d-postboot-20260903-0147/` and
`gpreempt-context-bpf-849ea75d-postboot-20260903-0147/`. Fresh XSched calibration
selected 9,511,106 repetitions for 79.968544 ms; the native, XSched, gpubpf and
bpftime HPF preflights all passed. Calibration and preflight records are in
`calibration-persistent-575-20260903/` and
`preflight-persistent-575-20260903/` under the same raw directory. These checks
re-establish runtime readiness after the reboot; they are not the full paired
performance comparison or evidence that the original GDR transport is supported.

## Three-comparison completion and service restoration

All three scoped comparisons subsequently completed: GPreempt 15/15 cells,
MoE-Infinity 15/15 cells, and XSched 40/40 mixed cells plus six controls.
XSched exited normally and its independent 46-cell raw audit passed before any
service restoration. The final pre-restoration snapshot records 400 W,
GPU 2 MiB/0%, no compute client, UVM references zero, empty struct-ops state,
and no Xid or kernel abnormality; both experiment leases were released.

At 2026-09-03 03:50 UTC, ordinary `systemctl start` restored
`nvidia-persistenced.service` and `gdm.service`. Both changed from inactive/dead
to active/running with `Result=success`; a later check confirmed those states.
Their unit enablement was unchanged. The post-restoration kernel-journal check
found no Xid, NVRM error/failure, BUG, Oops, or panic entry. No module reload,
reboot, package change, or new GPU experiment was performed.

The earlier four-paused-pod description is historical, not the final state.
Read-only checks already found the `lab` node Ready with its original labels
`fleet.yunwei37.com/gpu=rtx5090` and
`monitoring.yunwei37.com/managed=true`; neither label was rewritten. The generic
device plugin and node exporter were running; no running NVIDIA device-plugin
or DCGM pod was found on that node. The completed NVIDIA installer Job is not
a running device plugin. DaemonSet-list access was denied and no alternate
credentials or cluster changes were attempted. SSH, logind and k3s-agent stayed
active. The requested 575 driver and existing 400 W setting were not rolled back.

See the [completed comparison summary](../revision-experiment-status.md).
