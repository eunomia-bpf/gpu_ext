#!/usr/bin/env bash
# Executed once under both experiment locks. PIDs describe this invocation.
set -euo pipefail
hb_repo=/home/yunwei37/workspace/gpu/gpu_ext
hb_raw=$hb_repo/workloads/hummingbird/hostdev/raw-first-real-20260908.08lUpN
hb_saved=/var/tmp/fig13-original-modules-20260908.TkarSf
hb_changed=0
hb_stopped=0
restore_hb() {
    hb_rc=$?
    trap - EXIT
    set +e
    hb_restore_ok=1
    if [ "$hb_changed" = 1 ]; then
        for hb_mod in nvidia_drm nvidia_modeset nvidia_uvm nvidia; do
            if [ -d "/sys/module/$hb_mod" ]; then rmmod "$hb_mod" || hb_restore_ok=0; fi
        done
        modprobe nvidia || hb_restore_ok=0
        insmod "$hb_saved/nvidia-uvm.ko" uvm_enable_builtin_tests=0 || hb_restore_ok=0
    fi
    if [ "$hb_stopped" = 1 ] && [ "$hb_restore_ok" = 1 ]; then
        systemctl start nvidia-persistenced gdm || hb_restore_ok=0
        nohup "$hb_repo/workloads/lmcache-disk/gds-control/gds_policy" "$hb_repo/workloads/lmcache-disk/gds-control/gds_policy.bpf.o" > "$hb_raw/restored-gds.log" 2>&1 &
        printf 'RESTORED_GDS_PID=%s\n' "$!"
        nohup "$hb_repo/workloads/lmcache-disk/gds-control/kv_reclaim_loader" "$hb_repo/workloads/lmcache-disk/gds-control/kv_reclaim_policy.bpf.o" > "$hb_raw/restored-kv.log" 2>&1 &
        printf 'RESTORED_KV_PID=%s\n' "$!"
    fi
    chown -R yunwei37:yunwei37 "$hb_raw"
    printf 'CLIENT_EXIT=%s RESTORATION_OK=%s\n' "$hb_rc" "$hb_restore_ok"
    date -Is
    if [ "$hb_restore_ok" != 1 ]; then exit 1; fi
    exit "$hb_rc"
}
trap restore_hb EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -Is
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
kill -INT 3407414 3407415
hb_stopped=1
systemctl stop gdm nvidia-persistenced
while kill -0 3407414 2>/dev/null || kill -0 3407415 2>/dev/null; do sleep 0.2; done
while [ "$(</sys/module/nvidia_uvm/refcnt)" != 0 ]; do sleep 0.2; done
hb_changed=1
for hb_mod in nvidia_drm nvidia_modeset nvidia_uvm nvidia; do
    if [ -d "/sys/module/$hb_mod" ]; then rmmod "$hb_mod"; fi
done
insmod "$hb_saved/nvidia.ko"
insmod "$hb_saved/nvidia-uvm.ko" uvm_enable_builtin_tests=0
cd "$hb_repo"
env -u LD_PRELOAD -u CUDA_INJECTION64_PATH \
  PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin LANG=C.UTF-8 \
  CUDA_VISIBLE_DEVICES=0 GPREEMPT_POLICY=original \
  LD_LIBRARY_PATH="$hb_repo/extension/.output:$hb_repo/workloads/gpreempt/build/load-study:$hb_repo/workloads/gpreempt/deps/gdrcopy-2.5.2/src:/usr/local/cuda-12.9/lib64:/usr/local/lib" \
  workloads/hummingbird/build/hummingbird_client \
  workloads/hummingbird/raw/idle-study-575-01/block-00/periodic/idle_bpf/config.json \
  --mode idle_bpf \
  --profile workloads/hummingbird/raw/idle-study-575-01/profile-frozen.json \
  --split-cubin /tmp/hummingbird-hostdev-build-20260908.nxilLJ/resnet152-callable/mod-bpf.cubin \
  --bpf-program workloads/hummingbird/build/idle_policy.bin \
  > "$hb_raw/full-bpf-compatible-core.log" 2>&1
