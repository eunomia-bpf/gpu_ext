#!/usr/bin/env bash
set -euo pipefail
sweep_root=/home/yunwei37/workspace/gpu/gpu_ext/docs/eval/multi-tenant-memory/results_oversubscription_20260908.iftn85ji
sweep_repo=/home/yunwei37/workspace/gpu/gpu_ext
sweep_modules=/var/tmp/fig13-original-modules-20260908.TkarSf
sweep_compute=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)
if [ -n "$sweep_compute" ]; then
    echo "GPU busy: $sweep_compute"
    exit 1
fi
sweep_changed=0
sweep_loaders_stopped=0
restore_sweep() {
    sweep_rc=$?
    trap - EXIT
    set +e
    sweep_restore_ok=1
    if [ "$sweep_changed" = 1 ]; then
        for sweep_module in nvidia_drm nvidia_modeset nvidia_uvm nvidia; do
            if [ -d "/sys/module/$sweep_module" ]; then
                rmmod "$sweep_module" || sweep_restore_ok=0
            fi
        done
        modprobe nvidia || sweep_restore_ok=0
        insmod "$sweep_modules/nvidia-uvm.ko" uvm_enable_builtin_tests=0 || sweep_restore_ok=0
    fi
    if [ "$sweep_loaders_stopped" = 1 ] && [ "$sweep_restore_ok" = 1 ]; then
        systemctl start nvidia-persistenced gdm || sweep_restore_ok=0
        nohup "$sweep_repo/workloads/lmcache-disk/gds-control/gds_policy" "$sweep_repo/workloads/lmcache-disk/gds-control/gds_policy.bpf.o" > "$sweep_root/restored-gds.log" 2>&1 &
        sweep_gds=$!
        nohup "$sweep_repo/workloads/lmcache-disk/gds-control/kv_reclaim_loader" "$sweep_repo/workloads/lmcache-disk/gds-control/kv_reclaim_policy.bpf.o" > "$sweep_root/restored-kv.log" 2>&1 &
        sweep_kv=$!
        sleep 1
        kill -0 "$sweep_gds" "$sweep_kv" || sweep_restore_ok=0
        systemctl is-active nvidia-persistenced gdm
        echo "RESTORED_LOADER_PIDS=$sweep_gds,$sweep_kv"
    fi
    chown -R yunwei37:yunwei37 "$sweep_root"
    echo "RUNNER_EXIT=$sweep_rc RESTORATION_OK=$sweep_restore_ok"
    date -Is
    if [ "$sweep_restore_ok" != 1 ]; then exit 1; fi
    exit "$sweep_rc"
}
trap restore_sweep EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set -x
date -Is
kill -INT 2474310 2474311
sweep_loaders_stopped=1
systemctl stop gdm nvidia-persistenced
sleep 1
sweep_changed=1
for sweep_module in nvidia_drm nvidia_modeset nvidia_uvm nvidia; do
    if [ -d "/sys/module/$sweep_module" ]; then rmmod "$sweep_module"; fi
done
insmod "$sweep_modules/nvidia.ko"
insmod "$sweep_modules/nvidia-uvm.ko" uvm_enable_builtin_tests=0
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$sweep_root/gpu.csv"
python3 - <<'PY'
import ctypes,json
from pathlib import Path
cuda=ctypes.CDLL('/usr/local/cuda/lib64/libcudart.so')
free,total=ctypes.c_size_t(),ctypes.c_size_t()
rc=cuda.cudaMemGetInfo(ctypes.byref(free),ctypes.byref(total))
assert rc==0,rc
p=Path('/home/yunwei37/workspace/gpu/gpu_ext/docs/eval/multi-tenant-memory/results_oversubscription_20260908.iftn85ji/gpu-memory.json')
p.write_text(json.dumps({'total_bytes':total.value,'free_bytes_before_measurement':free.value},indent=2)+'\n')
print('CUDA GPU capacity:',total.value)
PY
python3 -u "$sweep_root/run_sweep.py"
