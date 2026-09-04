# SPIR-V OpenCL capability attempt 02: invalid in-process cleanup

Date: 2026-09-04  
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08  
Result directory: `raw/spirv-opencl-575-02`

This attempt is **invalid**. The capability query correctly observed an empty
`CL_DEVICE_IL_VERSION`, no `CL_DEVICE_ILS_WITH_VERSION` entries, and no
`cl_khr_il_program`; it therefore did not start the demo or submit a kernel.
However, the query loaded the NVIDIA OpenCL library directly into the runner.
Its UVM references could not reach zero while that same process was still
executing its post-run safety gate, so the 60-second cleanup gate failed and
the attempt lacks accepted before/after snapshots.

The repair moves only the capability query into a short-lived child process.
The parent retains both experiment leases, parses the child's structured
capability result, and performs the unchanged fail-closed gate after the child
has exited and released its OpenCL context. No timeout was relaxed and this
attempt contributes no portability result.
