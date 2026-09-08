# Record-preserving transposed transport: local implementation assignment

Session `ses_f7ea50ccdffeSGRcCn4AYmXR2k`, OpenCode / Qwen Next. Started
after the earlier device source task completed naturally; GLM and Qwen 27B
continue separate XSched source/build tasks, for three local sessions total.
The base runtime at `c4c83cd` is read-only and remains queued for measurement.
This is an independent patch assignment, not a completed optimization or result.

## Exact prompt

Implement an independent SOURCE PATCH for a record-preserving GPU ring transport optimization. This is one of three local OpenCode sessions; use Qwen Next. Root plans, reviews, commits/pushes and runs builds/experiments. Do not delegate, stop for silence, or set a short timeout.

Ownership: ONLY create workloads/llama.cpp/observability_overhead/revision-rq4/transport-soa-20260908/{transport-soa.patch,README.md} in gpu_ext. You may use a mktemp scratch directory with copies of the small relevant source files, edited using apply_patch. Do NOT edit or switch /home/yunwei37/workspace/gpu/bpftime-auto-warp: its built c4c83cd runtime and working source must remain frozen for a queued performance run. Do NOT create a worktree, edit other tasks, paper, driver, BPF example logic, benchmark runners or old data. Do not compile, run GPU code or reload anything: another task owns CPU build/GPU windows.

Read workspace/main/workload AGENTS.md. No hashes/checksums/fingerprints/digests. For patch creation use ordinary textual diff (no content-index headers), then apply_patch to save the patch file. Do not use shell file-writing tricks for code. Installed helper: /home/yunwei37/.codex/tmp/arg0/codex-arg029oUiU/apply_patch via bash; bash/read tools enabled. Native apply_patch may be unavailable. No new audit/gate framework or new tests campaign.

Base: bpftime-auto-warp c4c83cd, branch revision/automatic-warp-execution, read-only.
Relevant code:
- attach/nv_attach_impl/trampoline/default_trampoline.cu (_bpf_helper_ext_0025)
- runtime/src/bpf_map/gpu/nv_gpu_ringbuf_map.cpp/.hpp
- runtime/src/attach/bpf_attach_ctx_cuda.cpp and attach/nv_attach_impl/nv_attach_impl.hpp only if required.

Root's design:
Current mode2 keeps an entire 256-entry ring after EACH thread's 24-byte header. Adjacent lanes therefore store payload words far apart. Add opt-in BPFTIME_GPU_RINGBUF_TRANSPORT=3, map-owned just like mode2, with SAME logical per-thread ring capacity, exact record count and payload bytes, size handling, return values, drop categories, and encoded-publication handshake. Do NOT aggregate 32 events into one, sample, omit coordinates/timestamps, change BPF source, or claim whole-program leader execution.
Change INTERNAL layout ONLY for mode3:
  contiguous headers[thread_count], then word-transposed records.
  For slot index s and 64-bit record word w (word0 is original data_size),
  address = payload_base + ((s * record_words + w) * thread_count + tid) * 8.
  record_words = aligned(sizeof(uint64_t)+value_size)/8.
  Total allocation remains thread_count*(24+max_entries*record_stride)+error counters.
Adjacent threads with the same slot index now store adjacent 64-bit words. This is a memory-layout coalescing candidate, NOT measured traffic reduction or genuine multi-record publication batching.
Each lane reads its OWN data pointer and writes its own transposed words. Handle non-8-byte payload tails without overreading source, preserve data_size exactly, and never read another lane's thread-local stack via a shuffled pointer.
Mode3 uses each thread's existing encoded dirty/tail CAS/fence/release protocol unchanged; modes0/1/2 remain byte/layout compatible and unchanged. Out-of-range/invalid/full checks must happen before touching corresponding storage. Warp-only-output flag behavior is unchanged.
Host drain selects header location by map-owned mode. For mode3 reconstruct each original contiguous record into existing local_buffer (or minimal bounded scratch) before invoking the same callback in the same per-thread order. Publish head after callback as before. Stats must use the correct header location, with common error-counter offset and identical semantics. No GPU synchronization or periodic extra drain is added.

Deliver a valid unified patch applying to c4c83cd, plus concise README with touched files, exact layout formula, unchanged logical semantics, explicit limitations (unbuilt/unmeasured; no throughput claim, no automatic whole-program admission claim) and future command to apply after root freezes/measures mode2. A non-mutating git apply --check against the unchanged base is allowed; no source mutation there.
Prefer minimal changes in 3 files over new abstractions/modes beyond the requested mode3. No runtime rebuild or experiment yet. End with handoff and any actual implementation obstacle.
