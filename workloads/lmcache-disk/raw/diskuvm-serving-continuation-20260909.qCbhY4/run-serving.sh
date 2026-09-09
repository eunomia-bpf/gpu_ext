#!/usr/bin/env bash
# Root-owned experiment lifecycle, adapted from the completed disk-UVM run.
# Invoke under both GPU and struct-ops leases. PID values belong to this run.
set -euo pipefail
disk_repo=/home/yunwei37/workspace/gpu/gpu_ext
disk_raw=$disk_repo/workloads/lmcache-disk/raw/diskuvm-serving-continuation-20260909.qCbhY4
disk_restore=/var/tmp/fig13-original-modules-20260908.TkarSf/nvidia-uvm.ko
disk_module=/home/yunwei37/workspace/gpu/gpu_ext-kernel-575-gds/kernel-open/nvidia-uvm.ko
disk_changed=0
disk_loaders_stopped=0
disk_gds_pid=
disk_kv_pid=
restore_disk_run() {
    disk_rc=$?
    trap - EXIT
    set +e
    disk_restore_ok=1
    for disk_pid in "$disk_gds_pid" "$disk_kv_pid"; do
        if [ -n "$disk_pid" ]; then kill -INT "$disk_pid" 2>/dev/null; fi
    done
    for disk_pid in "$disk_gds_pid" "$disk_kv_pid"; do
        if [ -n "$disk_pid" ]; then wait "$disk_pid"; fi
    done
    if [ "$disk_changed" = 1 ]; then
        while [ "$(</sys/module/nvidia_uvm/refcnt)" != 0 ]; do sleep 0.2; done
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
kill -INT 344198 344199
disk_loaders_stopped=1
while kill -0 344198 2>/dev/null || kill -0 344199 2>/dev/null; do sleep 0.2; done
while [ "$(</sys/module/nvidia_uvm/refcnt)" != 0 ]; do sleep 0.2; done
rmmod nvidia_uvm
disk_changed=1
insmod "$disk_module" uvm_enable_builtin_tests=0
"$disk_repo/workloads/lmcache-disk/gds-control/gds_policy" "$disk_repo/workloads/lmcache-disk/gds-control/gds_policy.bpf.o" > "$disk_raw/active-gds.log" 2>&1 &
disk_gds_pid=$!
"$disk_repo/workloads/lmcache-disk/gds-control/kv_reclaim_loader" "$disk_repo/workloads/lmcache-disk/gds-control/kv_reclaim_policy.bpf.o" > "$disk_raw/active-kv.log" 2>&1 &
disk_kv_pid=$!
printf 'ACTIVE_GDS_PID=%s ACTIVE_KV_PID=%s\n' "$disk_gds_pid" "$disk_kv_pid"
cd "$disk_repo"
runuser -u yunwei37 -- "$disk_repo/workloads/lmcache-disk/current-venv/bin/python" -B -u \
    workloads/lmcache-disk/run_gds_kv_reclaim.py \
    --resume --blocks 2 --output "$disk_repo/workloads/lmcache-disk/raw/diskuvm-serving-20260909.9uBMtq/cells" \
    --calibration-result workloads/lmcache-disk/raw/kv-reclaim-recompute-calibration-575-20260907-01/calibration.json \
    --disk-uvm \
    --disk-uvm-fault-lib /tmp/lmcache-diskuvm-serving-build-20260909.7dZdC8/libdiskuvm_fault.so \
    > "$disk_raw/runner.log" 2>&1
