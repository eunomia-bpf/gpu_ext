set pagination off
set breakpoint pending on
set environment LD_LIBRARY_PATH /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/install/lib
unset environment LD_PRELOAD
set environment XSCHED_SCHEDULER GLB
set environment XSCHED_AUTO_XQUEUE ON
set environment XSCHED_AUTO_XQUEUE_LEVEL 2
set environment XSCHED_AUTO_XQUEUE_PRIORITY 0
set environment XSCHED_AUTO_XQUEUE_THRESHOLD 4
set environment XSCHED_AUTO_XQUEUE_BATCH_SIZE 2
set environment XSCHED_CUDA_LV2_PORT_120 1
set environment XSCHED_LEVEL2_TOOL_ACTUATOR 0
set environment XG_SERVICE_ONLY 1
set environment XG_NATIVE_META_EXTEND 1
set environment XG_NATIVE_ORIGINAL_ENTRY_CONTROL 1
set $xg_shortened = 0
break cuLaunchKernel
commands
silent
set $xg_extra = *(void ***)($rsp + 40)
if $xg_extra != 0
  if (unsigned long long)$xg_extra[0] == 1 && (unsigned long long)$xg_extra[2] == 2
    set $xg_size = (unsigned long long *)$xg_extra[3]
    if *$xg_size == 0x1520
      printf "CONTROL relay bytes 0x1520 -> 0x20\n"
      set *$xg_size = 0x20
      set $xg_shortened = $xg_shortened + 1
    end
  end
end
continue
end
run be 1 4 50 9511106 340 256 1 0 < workloads/xsched/raw/level2-native-meta-extent-20260908.aY3rBY/debug-go.txt
printf "CONTROL total shortened launches: %d\n", $xg_shortened
thread apply all bt 4
