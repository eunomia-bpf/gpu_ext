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
set debuginfod enabled off
set breakpoint pending on
break '(anonymous namespace)::GetXgOriginalLayout(CUfunc_st*, unsigned long, (anonymous namespace)::XgParamLayout const*, (anonymous namespace)::XgParamLayout*)' if $rsi == 5
run be 2 4 50 9511106 340 256 1 0 < /home/yunwei37/workspace/gpu/gpu_ext/workloads/xsched/raw/level2-native-lookup-backtrace-20260909.yNZVZG/debug-go.txt
set $layout_out = (unsigned long)$rcx
set $layout_in = (unsigned long)$rdx
bt 4
x/6gx $layout_in
x/5gx ($layout_in + 264)
disable 1
finish
p/x $rax
x/6gx $layout_out
x/5gx ($layout_out + 264)
quit

