# Full-record warp-contiguous SoA: running

Root built the local Qwen candidate successfully in the isolated directory
/home/yunwei37/workspace/gpu/bpftime-auto-warp/example/gpu/full-record-soa-warp-20260909.3mlEOo.
Command: make -j2 LAYOUT=soa-warp CLANG=clang-18 CUDA_HOME=/usr/local/cuda-12.9.
The four exact build inputs are retained in source/; build.log records the
successful build. Binary size is 1545016 bytes; it remains local as kernelretsnoop.
The shared bpftime runtime and the old SoA binary are not rebuilt or replaced.

The hypothesis is that physical-warp-contiguous storage improves prefill
throughput versus the existing global-coordinate SoA indexing. The writer
uses block-major/thread-major indexing, keeping all ten u64 fields, each
thread's timestamp, full event count, 32 banks, per-slot capacity and final
whole-arena drain. It does not reduce events or average timestamps. This is
an opt-in layout change, not automatic warp aggregation.

The five-block runner rotates soa-warp / frozen soa / uninstrumented,
using the existing TinyLlama pp512 Table 1 throughput path. Root mechanically
adapted paired-aosoa.py by renaming the candidate arm and its descriptions;
the loop, measurements, frozen baseline binary and runtime are unchanged.
Both shared leases cover the complete run:

flock /tmp/gpubpf-revision-gpu0.lock flock /tmp/gpubpf-revision-struct-ops.lock python3 -B -u /home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/observability_overhead/revision-rq4/raw/full-record-soa-warp-20260909.cjoIog/paired-run.py

No old measurement is overwritten or resumed under a changed binary.
The final full-arena drain stays outside client prefill timing and is
reported separately. No clock comparison or new performance gate is added.
No result is claimed before the new measurements finish; no manuscript edit.

