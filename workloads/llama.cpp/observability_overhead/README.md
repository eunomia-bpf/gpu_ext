# llama.cpp observability overhead

This directory contains a reproducible harness for the paper's device-side
observability overhead table. It measures llama.cpp prefill throughput with no
probe, then with the bpftime GPU observability examples attached to a selected
llama.cpp CUDA kernel.

The harness intentionally records the exact CUDA kernel symbol used for the
probe. The original paper table does not include that symbol, so this value is
part of the result provenance.

gpubpf's CUDA attach path patches PTX. The default runner therefore uses
`build-ptx-1b/bin/llama-bench`, whose CUDA library is built with
`CMAKE_CUDA_ARCHITECTURES=120-real;120-virtual` for RTX 5090. The llama model,
prompt size, GPU layer count, and benchmark command remain the same as the
regular build.

## Quick start

```bash
cd workloads/llama.cpp
uv run python observability_overhead/run_observability_overhead.py
```

If the CUDA-enabled bpftime build is not `bpftime/build`, select it explicitly:

```bash
uv run python observability_overhead/run_observability_overhead.py \
  --bpftime-build-dir ../../../bpftime/build-cuda-pr503
```

The wrapper script uses the same runner:

```bash
bash run_exp_observability_overhead.sh
```

Results are written to:

```text
workloads/llama.cpp/results/exp_observability_overhead/<timestamp>/
```

## Defaults

- Model: `models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
- llama-bench: `build-ptx-1b/bin/llama-bench`
- Workload: `llama-bench -p 512 -n 0 -r 1 -o json`
- Runs: 10 per configuration
- Target kernel: a llama.cpp RoPE forward CUDA kernel from `libggml-cuda.so`
- Tools: `kernelretsnoop`, `threadhist`, `launchlate`
- CUDA graphs are disabled with `GGML_CUDA_DISABLE_GRAPHS=1` because the
  current bpftime CUDA patch path can fall back on graph launches.
- `threadhist` uses `BPFTIME_THREADHIST_GPU_THREAD_COUNT=1048576` by default;
  the other tools keep `BPFTIME_MAP_GPU_THREAD_COUNT=8192`.
- `launchlate` defaults to the selected build's `libggml-cuda.so` and attaches
  to the exact mangled host launch stub for the selected kernel. Override with
  `LAUNCHLATE_UPROBE_BINARY` and `LAUNCHLATE_UPROBE_SYMBOL_HINT` only when the
  target build stores that stub in another library.

If `build-ptx-1b/bin/llama-bench` is missing, rebuild it with:

```bash
cmake -S llama.cpp -B build-ptx-1b \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_NO_VMM=ON \
  '-DCMAKE_CUDA_ARCHITECTURES=120-real;120-virtual'
cmake --build build-ptx-1b --target llama-bench -j"$(nproc)"
```

Override the target kernel when testing a different llama.cpp build:

```bash
uv run python observability_overhead/run_observability_overhead.py \
  --target-symbol '_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli'
```

Use `cuobjdump` to find candidate kernel symbols:

```bash
cuobjdump -symbols build/bin/libggml-cuda.so | rg 'STO_ENTRY.*rope|STO_ENTRY.*mul_mat'
```

Run a quick one-sample smoke test before the full 10-run experiment:

```bash
uv run python observability_overhead/run_observability_overhead.py \
  --runs 1 --pp 32 --timeout-s 300 --probe-startup-s 3
```
