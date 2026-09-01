# XSched CUDA build smoke

- Date: 2026-08-31
- Upstream: `https://github.com/XpuOS/xsched.git`
- Commit: `f49289f0220931df78de948ed841ecbaf960a919`
- Commit date: `2026-08-19T19:16:15+08:00`
- Host compiler: `/usr/bin/gcc` and `/usr/bin/g++` 13.3.0. The explicit
  paths avoid this host's nonstandard `/usr/bin/c++` Python wrapper.
- Host driver at smoke time: NVIDIA 610.43.02
- Command: `make PLATFORM=cuda`
- Result: success; installed `xserver`, `xcli`, `libpreempt.so`,
  `libhalcuda.so`, `libshimcuda.so`, and the `libcuda.so[.1]` shim links under
  `deps/xsched/output/`.
- Official example compile: `make cuda` in
  `examples/Linux/1_transparent_sched` also succeeds with nvcc 12.9. The
  example has not been executed because the GPU is occupied.
- The reviewed passive engagement patch applies and reverses cleanly against
  the pinned commit. A clean CUDA rebuild with the patch succeeds. Admission
  compares the exact small source diff and records ordinary path, size, inode,
  and modification/change-time metadata for required runtime files.
- Architecture scope verified from upstream `platforms/cuda/hal/src/arch/arch.cpp`:
  sm_120 takes `CudaQueueLv1`; `Guardian::Instance` and
  `TarpHandler::Instance` return null for sm_120.

This is a build-only smoke test, not evidence that interception, suspension,
or a paper workload works on the RTX 5090.

## Finite runtime smoke

On 2026-09-01, a minimal finite run exposed the clock-domain defect already
identified in the closed experiment review: a CUPTI timestamp taken before a
launch was about 287 ms ahead of the kernel's `%globaltimer` timestamp. The
kernel itself completed and all 32 computed values passed the host recurrence
check, but the old direct subtraction correctly rejected the timestamp.

The harness now runs a 16-sample clock probe in a separate CUDA process and
selects the narrowest host-before/device/host-after bracket. The runner invokes
this probe before starting XSched or a gpubpf policy, then supplies the measured
CUPTI-to-`%globaltimer` offset to every workload process. The first repaired
native smoke bounded the selected offset to within 2,557 ns and completed one
kernel with all 32 outputs validated. Its mapped submission preceded device
entry by 85,491 ns.

A second finite smoke exercised XSched's official CUDA shim and global HPF
server with one high-priority process. Two kernels completed, all 64 outputs
passed the recurrence check, and the process audit reported one Level-1 XQueue
with priority 1, threshold 16, and batch size 8. The server independently
logged creation, priority assignment, and destruction of that queue. No
suspend or resume was expected with only one client, and none was observed.

These are code-path and interception smokes only. They do not establish
preemption engagement, tail latency, throughput, or a gpubpf comparison. The
closed paper experiment remains closed, and the currently loaded driver still
lacks the gpubpf scheduling hooks required for a three-way run.

The next minimal multi-client smoke used the same finite harness with two LC
and four BE processes, four streams per process, and only two tasks per stream.
All six processes completed. Each process validated 696,320 output values, for
4,177,920 checked values in total. The clock probe's selected bracket had
2,515 ns uncertainty. All 48 task samples satisfied
`submission <= device entry <= device exit`, and LC release followed the last
reported active BE task by 5,000,207 ns.

The audit established 24 unique Level-1 XQueues: four per process. All 16 BE
queues used priority 0, threshold 4, and batch size 2; all eight LC queues used
priority 1, threshold 16, and batch size 8. Every BE queue recorded one
successful suspend and one successful resume, for 16 of each transition. The
HPF server and all six workers exited, and the GPU returned to 15 MiB and zero
utilization. The server emitted one IPC warning while handling its final
SIGINT, after all 24 queues were destroyed and all six clients had closed.

This establishes multi-client Level-1 suspend/resume path engagement and
semantic completion on this RTX 5090. It does not establish GPU hardware
preemption. It remains a deliberately short code smoke, not a performance
comparison: the task duration was not frozen through the reviewed calibration,
there was one execution rather than randomized repeated blocks, and neither
native nor gpubpf comparison cells ran.

## Supported-driver preparation

The gpubpf-enabled 575.57.08 open-kernel source now builds all five modules for
the installed Linux 6.14.0-37 kernel using GCC 13. Relinking against that
kernel's extracted base BTF produces split BTF that exposes
`struct nv_gpu_sched_ops`, `bpf_nv_gpu_preempt_tsg`, and the UVM
`struct gpu_mem_ops` interface. The driver preparation is recorded at
`kernel-module/nvidia-module/GPUBPF-RUNTIME-575.md`.

No module was installed or loaded. The machine still runs Linux 7.1.12 and
driver 610.43.02; activating the prepared stack requires matching 575.57.08
userspace, installing the modules for 6.14.0-37, and a reboot in an explicitly
authorized maintenance window.
