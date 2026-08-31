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

## Built artifact hashes

- `_store`: `8e15796eab10dd97d38e09a601a8d438e9f29b52d2af412e673d729017895248`
- `_engine`: `85c8f18f14b7cf9284da7e5dedb0b17e19d72ac83a7c4a8ca654be07fcdd368c`
- `_kv_cache`: `bf2df3f667ba0cb97cbae658c296878d66c7250fb17044e997d1f48a3e82a30a`
- `_paged_attn`: `bc1980e4e94e30582e7a6b01b5c68316cf04d8af5c4df366be6743cd5a7fbdfe`
- `_v4_fp4`: `3a60955c40b32e9b7691f942a54488edab3c7482bd4d7123e6cda4862ef193cc`
- `_marlin`: `66e0659342340e786a69137707c186a9941b2b54301716cd6f7f0dda91f85d23`

This is a compile/architecture smoke only. It does not establish runtime
correctness or performance, and it did not launch a GPU workload.
