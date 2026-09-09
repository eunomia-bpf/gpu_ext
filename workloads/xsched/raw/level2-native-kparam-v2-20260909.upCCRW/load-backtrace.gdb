set pagination off
set confirm off
set environment LD_LIBRARY_PATH /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/level2-build/.output/hal-native-20260908.i8na51/install/lib
unset environment LD_PRELOAD
unset environment XG_NATIVE_ORIGINAL_ENTRY_CONTROL
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
set environment XG_NATIVE_META_KPARAM 1
run be 1 4 50 9511106 340 256 1 0 < workloads/xsched/raw/level2-native-meta-extent-20260908.aY3rBY/debug-go.txt
thread apply all bt 12
x/8i $pc
