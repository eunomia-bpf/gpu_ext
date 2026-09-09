#!/usr/bin/env bash
# Retry original-UVM restoration after the completed serving process teardown.
set -euo pipefail
disk_repo=/home/yunwei37/workspace/gpu/gpu_ext
disk_raw=$disk_repo/workloads/lmcache-disk/raw/diskuvm-serving-20260909.9uBMtq
date -Is
printf 'UVM_REFCNT='
cat /sys/module/nvidia_uvm/refcnt
rmmod nvidia_uvm
insmod /var/tmp/fig13-original-modules-20260908.TkarSf/nvidia-uvm.ko uvm_enable_builtin_tests=0
nohup "$disk_repo/workloads/lmcache-disk/gds-control/gds_policy" "$disk_repo/workloads/lmcache-disk/gds-control/gds_policy.bpf.o" > "$disk_raw/restored-gds.log" 2>&1 &
printf 'RESTORED_GDS_PID=%s\n' "$!"
nohup "$disk_repo/workloads/lmcache-disk/gds-control/kv_reclaim_loader" "$disk_repo/workloads/lmcache-disk/gds-control/kv_reclaim_policy.bpf.o" > "$disk_raw/restored-kv.log" 2>&1 &
printf 'RESTORED_KV_PID=%s\n' "$!"
chown yunwei37:yunwei37 "$disk_raw/restored-gds.log" "$disk_raw/restored-kv.log"
printf 'RESTORATION_OK=1\n'
date -Is
