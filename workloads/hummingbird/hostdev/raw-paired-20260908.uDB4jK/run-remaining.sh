#!/usr/bin/env bash
# Recorded remaining-block invocation; all block-00 cells already complete.
# Explicit loader PIDs describe this invocation only.
set -euo pipefail
hb_repo=/home/yunwei37/workspace/gpu/gpu_ext
hb_raw=$hb_repo/workloads/hummingbird/hostdev/raw-paired-20260908.uDB4jK
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
        nohup "$hb_repo/workloads/lmcache-disk/gds-control/gds_policy" "$hb_repo/workloads/lmcache-disk/gds-control/gds_policy.bpf.o" > "$hb_raw/remaining-restored-gds.log" 2>&1 &
        printf 'RESTORED_GDS_PID=%s\n' "$!"
        nohup "$hb_repo/workloads/lmcache-disk/gds-control/kv_reclaim_loader" "$hb_repo/workloads/lmcache-disk/gds-control/kv_reclaim_policy.bpf.o" > "$hb_raw/remaining-restored-kv.log" 2>&1 &
        printf 'RESTORED_KV_PID=%s\n' "$!"
    fi
    chown -R yunwei37:yunwei37 "$hb_raw"
    printf 'PAIRED_EXIT=%s RESTORATION_OK=%s\n' "$hb_rc" "$hb_restore_ok"
    date -Is
    if [ "$hb_restore_ok" != 1 ]; then exit 1; fi
    exit "$hb_rc"
}
trap restore_hb EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -Is
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
kill -INT 4107386 4107388
hb_stopped=1
systemctl stop gdm nvidia-persistenced
while kill -0 4107386 2>/dev/null || kill -0 4107388 2>/dev/null; do sleep 0.2; done
while [ "$(</sys/module/nvidia_uvm/refcnt)" != 0 ]; do sleep 0.2; done
hb_changed=1
for hb_mod in nvidia_drm nvidia_modeset nvidia_uvm nvidia; do
    if [ -d "/sys/module/$hb_mod" ]; then rmmod "$hb_mod"; fi
done
insmod "$hb_saved/nvidia.ko"
insmod "$hb_saved/nvidia-uvm.ko" uvm_enable_builtin_tests=0
cd "$hb_repo"
# Python random.Random(20260908), shuffling the four-arm list in place once
# per block. Order is expanded here so the exact executed sequence is visible.
for hb_block in 01 02 03 04; do
    case "$hb_block" in
      01) hb_order='native_adapter bpf_device original_inline inline_bpfhost' ;;
      02) hb_order='original_inline inline_bpfhost native_adapter bpf_device' ;;
      03) hb_order='bpf_device original_inline inline_bpfhost native_adapter' ;;
      04) hb_order='inline_bpfhost bpf_device native_adapter original_inline' ;;
    esac
    for hb_arm in $hb_order; do
        hb_mode=idle_bpf
        hb_cubin=$hb_repo/workloads/hummingbird/build/resnet152-split/mod.cubin
        if [ "$hb_arm" = original_inline ]; then hb_mode=idle_c; fi
        if [ "$hb_arm" = native_adapter ]; then
            hb_cubin=/tmp/hummingbird-hostdev-build-20260908.nxilLJ/resnet152-callable/mod-native.cubin
        fi
        if [ "$hb_arm" = bpf_device ]; then
            hb_cubin=/tmp/hummingbird-hostdev-build-20260908.nxilLJ/resnet152-callable/mod-bpf.cubin
        fi
        printf 'START block-%s %s\n' "$hb_block" "$hb_arm"
        env -u LD_PRELOAD -u CUDA_INJECTION64_PATH \
          PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin LANG=C.UTF-8 \
          CUDA_VISIBLE_DEVICES=0 GPREEMPT_POLICY=original \
          LD_LIBRARY_PATH="$hb_repo/extension/.output:$hb_repo/workloads/gpreempt/build/load-study:$hb_repo/workloads/gpreempt/deps/gdrcopy-2.5.2/src:/usr/local/cuda-12.9/lib64:/usr/local/lib" \
          workloads/hummingbird/build/hummingbird_client \
          "workloads/hummingbird/raw/idle-study-575-01/block-$hb_block/periodic/idle_bpf/config.json" \
          --mode "$hb_mode" \
          --profile workloads/hummingbird/raw/idle-study-575-01/profile-frozen.json \
          --split-cubin "$hb_cubin" \
          --bpf-program workloads/hummingbird/build/idle_policy.bin \
          > "$hb_raw/block-$hb_block-$hb_arm.log" 2>&1
        printf 'DONE block-%s %s exit=0\n' "$hb_block" "$hb_arm"
    done
done
