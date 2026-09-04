# SPIR-V OpenCL capability attempt 03: valid unsupported boundary

Date: 2026-09-04  
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08  
Result directory: `raw/spirv-opencl-575-03`

The isolated capability query completed with clean safety snapshots before and
after it: UVM reference count was zero, no compute process survived, and no
kernel/GPU abnormality appeared. The device reported OpenCL 3.0, but an empty
`CL_DEVICE_IL_VERSION`, no `CL_DEVICE_ILS_WITH_VERSION` entries, and no
`cl_khr_il_program` extension.

The fail-closed gate therefore returned the expected `unsupported` status
(runner exit 2) before starting the demo process. The independent analyzer
reports `complete=true`, `run_status=valid`, and
`tested_hypothesis=contradicted`. This is a valid capability-boundary result:
the current NVIDIA OpenCL stack cannot execute the generated SPIR-V module
through this route. It is not a successful device-side SPIR-V execution and
provides no performance result.
