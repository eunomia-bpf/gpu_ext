# Stale-state owner08 interruption and retry boundary

Date: 2026-09-05
Scope: retained excluded-preflight evidence only; no performance result

## What owner08 actually completed

The retained campaign is
`raw/stale-state-575-preflight-20260905-owner-08`. Its top-level
`campaign.json` says `complete: false`. The ordered seven-cell matrix remains
intact, but `completed` contains only:

1. `bpf_delay_100ms`;
2. `native_delay_100ms`;
3. `bpf_fresh`.

Each of those three cell directories has a complete, passing `execution.json`.
The fourth directory, `bpf_delay_1000ms`, also has a complete, passing
per-cell execution record, zero numerical mismatches, zero policy-record loss,
and clean per-cell teardown. It was not appended to the campaign's completed
list before the outer process ended. The remaining `native_delay_1000ms`,
`uvm_default`, and `native_fresh` directories do not exist.

The outer record is
`raw/stale-state-575-lifecycle-20260905-owner-08/lifecycle.json`. It is also
incomplete: its final event is `candidate_loaded`; it has no child-completion,
candidate-removal, old-UVM-restoration, service-restoration, or final-safety
event. Its child stdout and stderr logs are empty. The retained files therefore
establish that the lifecycle lost its supervising process after live work had
started, but they do not establish the exact external cause of that loss.

The excluded preflight is the indivisible admission unit. Consequently,
**zero owner08 cells are admissible preflight or paper evidence**, including
the fourth cell whose local record says passed. Owner08 must not be resumed,
completed in place, copied into another attempt, or combined with another
campaign. Its files are retained as interruption/setup evidence.

## Post-interruption recovery observation

A read-only observation at 2026-09-05 01:56 PDT found no stale-state process,
no compute client, no stale-state struct-ops map or link, UVM reference count
zero, and an idle RTX 5090 at 15 MiB, P8, and 0% utilization. The loaded UVM
reported driver 575.57.08 and its live BTF did not contain the stale-state
diagnostic. `gdm.service` and `nvidia-persistenced.service` were active/running
with successful results. The kernel and journal contained no Xid, Oops, BUG,
or panic after midnight.

This is a post-hoc observation of the recovered machine, not a substitute for
owner08's missing lifecycle-finalization record. It does not identify which
external recovery action restored the system.

## Subsequent zero-cell admission attempts

- `owner-09` failed before leases, staging, services, modules, or GPU work
  because `/tmp/gpubpf-revision-gpu0.lock` was absent. Its lifecycle state says
  `candidate_loaded: false`, `old_removed: false`, and an empty stopped-service
  list. It has no preflight output directory.
- `owner-10` failed at the lease identity/mode admission check. The recreated
  lease files had mode 0666 rather than the admitted root-owned 0644 shape.
  It likewise changed no service or module and has no preflight output.
- `owner-11` acquired the reconstructed root-owned 0644 leases, then failed
  read-only admission because the post-interruption live stock UVM has no
  `gpu_mem_ops` structure (`gpu_mem_ops ABI differs: ()`). It did not stage,
  stop a service, remove a module, or run a GPU cell.

These attempts are pre-admission infrastructure diagnostics, not experiment
cells. Their lifecycle records remain under `raw/`; no missing campaign should
be manufactured for them.

## Safe next boundary

There is no existing reviewed repository command that persistently bootstraps
the current stock UVM into the staged gpreempt UVM and then leaves that custom
module loaded. The scheduler-init lifecycle replaces the whole NVIDIA module
subset and runs a different matrix; the revision-prefetch and stale-state
lifecycle coordinators both require the live UVM already to match the admitted
gpreempt restore ABI. A manual `rmmod`/`insmod` sequence would discard their
bounded recovery and signal handling and is not an acceptable shortcut.

The stale-state lifecycle now explicitly admits a supplied stock UVM file as
its exact restore artifact. The forward candidate retains the strict
seven-member stale-state ABI. The restore parser accepts either the exact
six-member gpubpf base interface or complete absence of `gpu_mem_ops`; a
partial or other shape is rejected. Before any mutation, the explicit file's
ABI and parameter inventory must match the live UVM ABI and captured parameter
values. Recovery restores and revalidates that same file, ABI, parameters,
services, boot, idle state, and device ownership. This keeps the change
UVM-only and returns the host to its actual admission state.

The CPU suite passes 57 tests. A source-only dry run used the installed stock
575 `.ko.zst` decompressed as an ordinary temporary `nvidia-uvm.ko`; it
classified that restore as `stock_no_gpu_mem_ops`, admitted the strict
seven-member owner08 candidate, and produced the fresh owner12 seven-cell
command plan without loading a module or running GPU work.

For the actual run, prepare a fresh root-owned uncompressed restore directory
from the current installed module. The source is read only; the destination
directory must not already exist:

```sh
set -o pipefail
sudo -n mkdir --mode=0755 \
  /opt/gpubpf/modules/575.57.08/stock-dkms-6.15.11-owner12
zstd -q -d --stdout \
  /lib/modules/6.15.11-061511-generic/updates/dkms/nvidia-uvm.ko.zst |
  sudo -n dd \
    of=/opt/gpubpf/modules/575.57.08/stock-dkms-6.15.11-owner12/nvidia-uvm.ko \
    status=none
sudo -n chown root:root \
  /opt/gpubpf/modules/575.57.08/stock-dkms-6.15.11-owner12/nvidia-uvm.ko
sudo -n chmod 0644 \
  /opt/gpubpf/modules/575.57.08/stock-dkms-6.15.11-owner12/nvidia-uvm.ko
```

Then run the entire excluded preflight under a persistent systemd supervisor
with the new explicit restore and fresh owner12 paths:

```sh
sudo -n systemd-run \
  --unit=gpubpf-stale-owner12 \
  --collect \
  --property=Type=exec \
  --property=WorkingDirectory=/home/yunwei37/workspace/gpu/gpu_ext \
  /usr/bin/python3 -B \
  /home/yunwei37/workspace/gpu/gpu_ext/workloads/stale-state-575/run_module_lifecycle.py \
  execute \
  --candidate \
    /opt/gpubpf/modules/575.57.08/stale-state-v1-stage-20260905-owner-preflight-08/nvidia-uvm.ko \
  --restore \
    /opt/gpubpf/modules/575.57.08/stock-dkms-6.15.11-owner12/nvidia-uvm.ko \
  --stage \
    /opt/gpubpf/modules/575.57.08/stale-state-v1-stage-20260905-owner-preflight-12 \
  --output \
    /home/yunwei37/workspace/gpu/gpu_ext/workloads/stale-state-575/raw/stale-state-575-preflight-20260905-owner-12 \
  --record \
    /home/yunwei37/workspace/gpu/gpu_ext/workloads/stale-state-575/raw/stale-state-575-lifecycle-20260905-owner-12
```

The next live attempt must execute all seven cells from ordinal one. No
owner08--11 cell or lifecycle record may be used to satisfy its completion
gate. The retained owner08 module is only a revalidated implementation input
that the lifecycle copies into a fresh owner12 stage.
