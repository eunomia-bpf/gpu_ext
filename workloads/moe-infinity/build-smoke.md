# MoE-Infinity Blackwell build smoke

- Date: 2026-08-31
- MoE-Infinity commit: `b766f8f1f6379fac6cd23594713ba6f4c7650ad9`
- CUTLASS commit: `dc45f979ae336a235da1676b311f35efeb30149a`
- Python/PyTorch: Python 3.10.19, PyTorch 2.8.0+cu128
- Compiler/toolkit: `/usr/bin/g++` 13.3.0, nvcc 12.9
- Build flags: `MOE_ENABLE_SM120=1`, `MOE_ENABLE_SM90=0`, `NVTX_DISABLE=1`,
  `MAX_JOBS=4`, and the pinned CUTLASS checkout.
- Command: `CXX=/usr/bin/g++ CC=/usr/bin/gcc ... python setup.py build_ext --inplace`.
- Result: success. Six extension modules were linked. `cuobjdump --list-elf`
  confirms `sm_120` cubins in `_store`, `_engine`, `_paged_attn`, and
  `_marlin`, plus `sm_120a` cubins in `_v4_fp4`. `_kv_cache` contains no CUDA
  cubin.

The first attempt used `/usr/bin/c++` and failed. On this host that path is a
Python wrapper which reconstructs arguments through `os.system`, loses the
quotes in PyBind ABI macros, and fails to propagate the compiler exit code.
Selecting the real `/usr/bin/g++` fixed the build without an upstream source
change.

## Built artifacts

- `_store`
- `_engine`
- `_kv_cache`
- `_paged_attn`
- `_v4_fp4`
- `_marlin`

This is a compile/architecture smoke only. It does not establish runtime
correctness or performance, and it did not launch a GPU workload.
