# llama.cpp observability overhead

- Timestamp: `20260901_094736`
- Model: `/home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
- llama-bench: `/home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/build-ptx-1b/bin/llama-bench`
- Target kernel: `_Z9rope_normILb1ELb0Ef6__halfEvPKT1_PT2_iiiiiPKifff14rope_corr_dimsfPKfPKli`
- Workload: `llama-bench -p 32 -n 0`
- Runs per config: `1`
- PTX files in libggml-cuda: `120`
- CUDA graphs disabled: `True`

| Config | Runs | Prefill tok/s geomean | Overhead vs baseline | Max probe samples |
|---|---:|---:|---:|---:|
| baseline | 1 | 256.91 | - | 0 |
| launchlate | 0 | n/a | - | 0 |

Positive overhead means token/s degradation relative to the no-probe baseline.
A zero probe sample count means the selected CUDA kernel was not observed for that tool run.
