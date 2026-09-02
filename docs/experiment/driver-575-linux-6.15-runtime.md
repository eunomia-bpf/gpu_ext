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
