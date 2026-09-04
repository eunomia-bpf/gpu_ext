# SPIR-V standalone device preflight

This directory freezes a narrow portability preflight around bpftime's existing
`vm/llvm-jit/example/spirv/spirv_opencl_test.cpp` program.  The source-native
demo translates a five-instruction eBPF program to SPIR-V, loads the module
through OpenCL, and checks `100 + 42 = 142` on one device invocation.

The preflight is intentionally **not** evidence of a gpubpf device-hook attach
backend, cross-layer maps, the SIMT verifier, policy execution in an application
kernel, or an AMD/Intel port.  On this host OpenCL currently exposes only the
NVIDIA RTX 5090, so success is a standards-path/code-generation check on the
same vendor, not a cross-vendor result.

## Build audit

Use a build directory outside both the production Table 1 build and the
`vm/llvm-jit` source submodule:

```bash
cmake -S /home/yunwei37/workspace/gpu/bpftime-table1-575/vm/llvm-jit \
  -B /home/yunwei37/workspace/gpu/bpftime-spirv-build-20260904 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DLLVMBPF_ENABLE_SPIRV=ON \
  -DBPFTIME_ENABLE_UNIT_TESTING=OFF \
  -DBUILD_LLVM_AOT_CLI=OFF
cmake --build /home/yunwei37/workspace/gpu/bpftime-spirv-build-20260904 \
  --target spirv_opencl_test -j4
```

This configure/build is CPU-only.  It does not execute the OpenCL program.

## Device capability gate and preflight

Do not run while another GPU experiment owns the shared revision leases.  The
runner opens the two pre-created lock files read-only, requires an idle RTX
5090 with driver 575.57.08 and the fixed 400 W power service, and runs only when
the explicit device flag is present. Before starting the source-native demo, it
records `CL_DEVICE_IL_VERSION`, `CL_DEVICE_ILS_WITH_VERSION`, and the device
extension inventory in `device-capability.json`. It starts the demo only if one
of the standard IL queries explicitly advertises SPIR-V; OpenCL 3.0 alone and
the presence of `cl_khr_il_program` alone are not substituted for that gate.

Attempt 01 reached `clCreateProgramWithIL`, which returned
`CL_INVALID_OPERATION` (`-59`) on the RTX 5090 / 575.57.08 stack. Its raw
records remain under `raw/spirv-opencl-575-01/`. A new diagnostic attempt must
use a new directory:

```bash
python3 run_spirv_opencl_preflight.py \
  --binary /home/yunwei37/workspace/gpu/bpftime-spirv-build-20260904/example/spirv/spirv_opencl_test \
  --source /home/yunwei37/workspace/gpu/bpftime-table1-575/vm/llvm-jit/example/spirv/spirv_opencl_test.cpp \
  --build-dir /home/yunwei37/workspace/gpu/bpftime-spirv-build-20260904 \
  --output-dir ./raw/spirv-opencl-575-02 \
  --execute-device
```

Exit code 2 with runner status `unsupported` is a complete, fail-closed
capability result: no demo process starts and no host SPIR-V generation or GPU
kernel execution occurs. It is not a successful device preflight.

The positive cell must emit and execute a valid `bpf_main` OpenCL kernel and
produce 142 from 100.  The runner then independently validates and disassembles
the retained SPIR-V module.  A copied module with a deliberately invalid magic
word must be rejected by `spirv-val`; the tampered module is never submitted to
OpenCL.  The analyzer reopens the raw logs and module and reproduces these
checks without importing the runner:

```bash
python3 analyze_spirv_opencl_preflight.py ./raw/spirv-opencl-575-02
```

The analyzer also replays retained attempt 01. It distinguishes validated
host-generated SPIR-V from OpenCL program creation and device-kernel execution;
none of these stages is inferred from another.

CPU-only offline tests do not acquire leases or enumerate/execute a GPU:

```bash
python3 -m unittest -v test_spirv_opencl_preflight.py
```
