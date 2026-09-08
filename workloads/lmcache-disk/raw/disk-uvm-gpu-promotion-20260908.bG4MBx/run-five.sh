#!/usr/bin/env bash
set -euo pipefail
disk_raw=/home/yunwei37/workspace/gpu/gpu_ext/workloads/lmcache-disk/raw/disk-uvm-gpu-promotion-20260908.bG4MBx
disk_repo=/home/yunwei37/workspace/gpu/gpu_ext
disk_cache=/var/tmp/disk-uvm-gpu-promotion-20260908.njuwc4
disk_restore=/var/tmp/fig13-original-modules-20260908.TkarSf/nvidia-uvm.ko
disk_module=/home/yunwei37/workspace/gpu/gpu_ext-kernel-575-gds/kernel-open/nvidia-uvm.ko
disk_changed=0
disk_loaders_stopped=0
restore_disk_run() {
    disk_rc=$?
    trap - EXIT
    set +e
    disk_restore_ok=1
    if [ "$disk_changed" = 1 ]; then
        rmmod nvidia_uvm || disk_restore_ok=0
        if [ "$disk_restore_ok" = 1 ]; then
            insmod "$disk_restore" uvm_enable_builtin_tests=0 || disk_restore_ok=0
        fi
    fi
    if [ "$disk_loaders_stopped" = 1 ] && [ "$disk_restore_ok" = 1 ]; then
        nohup "$disk_repo/workloads/lmcache-disk/gds-control/gds_policy" "$disk_repo/workloads/lmcache-disk/gds-control/gds_policy.bpf.o" > "$disk_raw/restored-gds.log" 2>&1 &
        printf 'RESTORED_GDS_PID=%s\n' "$!"
        nohup "$disk_repo/workloads/lmcache-disk/gds-control/kv_reclaim_loader" "$disk_repo/workloads/lmcache-disk/gds-control/kv_reclaim_policy.bpf.o" > "$disk_raw/restored-kv.log" 2>&1 &
        printf 'RESTORED_KV_PID=%s\n' "$!"
    fi
    chown -R yunwei37:yunwei37 "$disk_raw"
    printf 'DISK_RUN_EXIT=%s RESTORATION_OK=%s\n' "$disk_rc" "$disk_restore_ok"
    date -Is
    if [ "$disk_restore_ok" != 1 ]; then exit 1; fi
    exit "$disk_rc"
}
trap restore_disk_run EXIT
date -Is
kill -INT 3146830 3146831
disk_loaders_stopped=1
while kill -0 3146830 2>/dev/null || kill -0 3146831 2>/dev/null; do sleep 0.2; done
while [ "$(</sys/module/nvidia_uvm/refcnt)" != 0 ]; do sleep 0.2; done
rmmod nvidia_uvm
disk_changed=1
insmod "$disk_module" uvm_enable_builtin_tests=0
for disk_repeat in 01 02 03 04 05; do
    mkdir "$disk_raw/repeat-$disk_repeat"
    printf 'START_REPEAT=%s\n' "$disk_repeat"
    set +e
    (cd "$disk_raw/repeat-$disk_repeat" && stdbuf -oL -eL "$disk_repo/workloads/lmcache-disk/gds-control/disk-uvm/disk_uvm_perf" --gpu-promotion --size 256MiB --backing-file "$disk_cache/backing.bin") > "$disk_raw/repeat-$disk_repeat/client.log" 2>&1
    disk_cell_rc=$?
    set -e
    printf 'REPEAT=%s EXIT=%s\n' "$disk_repeat" "$disk_cell_rc"
    if [ "$disk_cell_rc" != 0 ]; then exit "$disk_cell_rc"; fi
done
