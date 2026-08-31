# MoE-Infinity repaired Blackwell build record

- Date: 2026-08-31
- MoE-Infinity commit: `b766f8f1f6379fac6cd23594713ba6f4c7650ad9`
- CUTLASS commit: `dc45f979ae336a235da1676b311f35efeb30149a`
- Source changes: `instrumentation.patch` and `row-chunking.patch`
- Python/PyTorch: Python 3.12.3, PyTorch 2.13.0+cu129
- Compiler/toolkit: `/usr/bin/gcc` and `/usr/bin/g++` 13.3.0, CUDA 12.9
  compiler build 36037853
- Build flags: `MOE_ENABLE_SM120=1`, `MOE_ENABLE_SM90=0`, `NVTX_DISABLE=1`,
  `MAX_JOBS=4`, and the pinned CUTLASS checkout.
- Command: `CXX=/usr/bin/g++ CC=/usr/bin/gcc CUDA_HOME=/usr/local/cuda-12.9
  MOE_ENABLE_SM120=1 MOE_ENABLE_SM90=0 NVTX_DISABLE=1 MAX_JOBS=4
  CUTLASS_DIR=<pinned checkout> .venv/bin/python setup.py build_ext --inplace`.
- Result: success. Six extension modules were linked. `cuobjdump --list-elf`
  confirms `sm_120` cubins in `_store`, `_engine`, `_paged_attn`, and
  `_marlin`, plus `sm_120a` cubins in `_v4_fp4`. `_kv_cache` contains no CUDA
  cubin.

Python import resolution confirms that the active environment loads the
CPython 3.12 files below. Older CPython 3.10 files remain as historical build
residue and are neither imported nor admitted by the experiment runner.

## Built artifacts

- `_store.cpython-312-x86_64-linux-gnu.so`: 71,782,016 bytes
- `_engine.cpython-312-x86_64-linux-gnu.so`: 356,864 bytes
- `_kv_cache.cpython-312-x86_64-linux-gnu.so`: 17,253,632 bytes
- `_paged_attn.cpython-312-x86_64-linux-gnu.so`: 590,640 bytes
- `_v4_fp4.cpython-312-x86_64-linux-gnu.so`: 11,132,056 bytes
- `_marlin.cpython-312-x86_64-linux-gnu.so`: 11,950,088 bytes

The standalone numerical gate executed the repaired `MoEMLP::forward()` on the
RTX 5090 for 1, 256, 257, and 353 BF16 rows. Every result matched its
same-parameter reference within `rtol=1e-2`, `atol=1e-2`; observed maximum
absolute and relative error were both zero. This establishes the repaired row
path's numerical check, not full-model completion or performance. No repaired
120B preflight or timed block had been launched when this record was written.

The superseded compile-only build used Python 3.10.19 and PyTorch 2.8.0+cu128.
Its first attempt selected `/usr/bin/c++`, a Python wrapper on this host that
lost PyBind macro quoting. Selecting the real `/usr/bin/g++` fixed that
historical build. Those observations do not describe the active repaired
build above.
