# ASPLOS 2027 revision experiment handoff — 2026-08-31

The active work is implementation and evaluation against
[the revision plan](paper/asplos-27-rebuttal/revision-plan.md). No new complete
GPU experiment or paper-facing performance result is claimed by this handoff.
The driver port and CPU tests are dependencies, not research results.

## Persisted work

| Work | Evidence and current status |
| --- | --- |
| NVIDIA 610 port | [port branch](https://github.com/eunomia-bpf/gpu_ext-kernel-modules/tree/port/nvidia-610.43.02), including [build and runtime notes](https://github.com/eunomia-bpf/gpu_ext-kernel-modules/blob/port/nvidia-610.43.02/GPUBPF-PORT.md). All five modules build for 6.15.11 and 7.1.12; BTF verifies both struct_ops and all nine kfuncs. No runtime replacement/attach test yet. |
| MoE-Infinity | [Plan](../workloads/moe-infinity/plan.md) and [review](../workloads/moe-infinity/plan-review.md) approved; host stride/LFU policy, native read-only counters, UVM eviction monitor, workload, and runner persisted. All 26 offline tests pass. Real preflight and five complete valid blocks remain pending. |
| LMCache NVMe | [Plan](../workloads/lmcache-disk/plan.md) and [review](../workloads/lmcache-disk/plan-review.md) approved. All six CPU tests pass. O_DIRECT preflight, correctness smoke, and complete comparison remain pending. |
| XSched | [Review](../workloads/xsched/plan-review.md) closed the first proposal after three rounds. The CUPTI/globaltimer epoch proof and interpretation categories require repair in a new proposal. The CUDA workload builds; no GPU result. |
| RTX 5090/NVBit | [Review](../workloads/llama.cpp/observability_overhead/revision-rq4/plan-review.md) closed the first proposal. Final runner defects were repaired but not independently approved. The frozen NVBit comparison requires a supported 575.x stack; a 610 diagnostic cannot satisfy R6. |

## Live state and next actions

The host changed from Linux `6.15.11-061511-generic` to
`7.1.12-070112-generic` during this work. The installed driver remains official
NVIDIA Open Kernel Modules 610.43.02 via DKMS. The 610 port was rebuilt for the
new kernel; GCC 14 matches the installed DKMS build, but compiler/objtool
warnings remain documented in the port notes. Its BTF was generated with the
7.1 native script and the running kernel's base BTF, without editing system
headers or installed modules.

The GPU was briefly free, but the final module-switch check found unrelated
SGLang processes and a nonzero UVM reference count. No process was killed and
no module was unloaded. GDM/Xorg also holds the core NVIDIA module. A request
for permission to temporarily stop GDM for full scheduling-module validation
has not been answered; that would not by itself authorize stopping SGLang.

After compute users release the GPU, memory-hook validation may replace only
the unused `nvidia_uvm` module while retaining the matching official 610 core.
Full scheduling validation needs an idle GPU and an authorized display
maintenance window. Use temporary `insmod`, with system `modprobe` as recovery;
never install custom modules persistently.

The approved MoE and LMCache plans still freeze 575.57.08. Before the first
610 run, explicitly record a uniform driver-stack deviation for all compared
cells and requalify the driver-specific artifacts and UVM Tools ABI; do not
silently relax an admission check or pool results across driver stacks.
The NVBit 575 requirement is separate and cannot be waived by the 610 port.

Publication now happens after each validated scoped change, as requested.
Dependencies, virtual environments, compiled modules, and old July diagnostic
logs remain outside source commits. Unrelated paper-submodule, FAISS, and
PyTorch worktree changes are preserved and were not staged.
