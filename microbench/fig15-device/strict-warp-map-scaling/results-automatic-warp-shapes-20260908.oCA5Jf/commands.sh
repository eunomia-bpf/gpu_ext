exec 9</tmp/gpubpf-revision-gpu0.lock
flock -n 9 || exit
warp_formal_dir=$(mktemp -d /home/yunwei37/workspace/gpu/gpu_ext/microbench/fig15-device/strict-warp-map-scaling/results-automatic-warp-shapes-20260908.XXXXXX)
printf 'RESULT_ROOT=%s\n' "$warp_formal_dir"
warp_build=/home/yunwei37/workspace/gpu/bpftime-auto-warp/build-auto-warp-575
export PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64 CUDA_VISIBLE_DEVICES=0
export BPFTIME_MAP_GPU_THREAD_COUNT=128 BPFTIME_SHM_MEMORY_MB=256 BPFTIME_MAX_FD_COUNT=1024
export BPFTIME_LOG_OUTPUT=console SPDLOG_LEVEL=info BPFTIME_SM_ARCH=sm_120 BPFTIME_VERIFIER_LEVEL=STRICT
export CUDA_HOME=/usr/local/cuda-12.9 BPFTIME_CUDA_ROOT=/usr/local/cuda-12.9
for warp_threads in 32 1024; do
export BPFTIME_MAP_GPU_THREAD_COUNT=$warp_threads
for warp_block in {1..5}; do
  case $(( (warp_block-1)%3 )) in 0) warp_order='native off on';; 1) warp_order='off on native';; 2) warp_order='on native off';; esac
  warp_position=0
  for warp_arm in $warp_order; do
    warp_cell="$warp_formal_dir/threads-${warp_threads}-block-${warp_block}-${warp_position}-${warp_arm}"
    mkdir "$warp_cell"
    warp_run_id=$(( 92000 + warp_threads*10 + warp_block*3 + warp_position ))
    if [[ "$warp_arm" == native ]]; then
      .output/warp-map-bench --threads "$warp_threads" --warmup 8 --launches 128 --run-id "$warp_run_id" 9<&- > "$warp_cell/application.log" 2>&1
      warp_app_rc=$?
      warp_loader_rc=0
    else
      export BPFTIME_GPU_AUTO_WARP_EXECUTION=0
      [[ "$warp_arm" == on ]] && export BPFTIME_GPU_AUTO_WARP_EXECUTION=1
      export BPFTIME_GLOBAL_SHM_NAME="fig15_auto_shapes_20260908_$_${warp_threads}_${warp_block}_${warp_arm}"
      LD_PRELOAD="$warp_build/runtime/syscall-server/libbpftime-syscall-server.so" .output/warp-map-loader .output/warp-map-probe.bpf.o shared_update 3600 9<&- > "$warp_cell/loader.log" 2>&1 &
      warp_pid=$!
      while ! rg -q '^FIG15_WARP_READY' "$warp_cell/loader.log"; do
        kill -0 "$warp_pid" 2>/dev/null || break
        sleep 0.1
      done
      LD_PRELOAD="$warp_build/runtime/agent/libbpftime-agent.so" BPFTIME_LOG_OUTPUT="$warp_cell/agent.log" BPFTIME_CUDA_DEFER_PTX_EXTRACTION=1 BPFTIME_CUDA_TARGETED_LATE_BOOTSTRAP=1 .output/warp-map-bench --threads "$warp_threads" --warmup 8 --launches 128 --run-id "$warp_run_id" 9<&- > "$warp_cell/application.log" 2>&1
      warp_app_rc=$?
      kill -INT "$warp_pid" 2>/dev/null
      wait "$warp_pid"
      warp_loader_rc=$?
      if [[ -f "/dev/shm/$BPFTIME_GLOBAL_SHM_NAME" && ! -L "/dev/shm/$BPFTIME_GLOBAL_SHM_NAME" ]]; then unlink "/dev/shm/$BPFTIME_GLOBAL_SHM_NAME"; fi
    fi
    printf 'threads=%s block=%s position=%s arm=%s app_rc=%s loader_rc=%s run_id=%s\n' "$warp_threads" "$warp_block" "$warp_position" "$warp_arm" "$warp_app_rc" "$warp_loader_rc" "$warp_run_id" > "$warp_cell/execution.log"
    cat "$warp_cell/execution.log"
    rg --no-line-number '^FIG15_MEASUREMENT' "$warp_cell/application.log"
    warp_position=$((warp_position+1))
  done
done
done
exec 9<&-
printf 'CAMPAIGN_FINISHED\n'
