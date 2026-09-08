# Transposed transport: parallel host/device ownership

After XSched's component build task completed naturally and was pushed as
`085ee4ec`, its local slot was assigned to host transport implementation.
Qwen Next was notified before dispatch: it now owns only the device patch;
Qwen 27B owns only the host patch. The common layout is fixed below.
This supersedes the original single-owner assignment in
`record-preserving-transport-soa-local-prompt-20260908.md`.
Session: `ses_f7e9d6af4ffeyntfT0hZqhYC5c` (direct Qwen 27B).

## Exact host prompt

Implement HOST-ONLY source patch for record-preserving transposed GPU ring transport, coordinated with Qwen Next's DEVICE-ONLY patch. Root plans, reviews, commits/pushes, builds and measures. Three local sessions maximum; the previous Qwen27 XSched build task completed naturally before this dispatch. Do not delegate, impose short timeouts, compile, run GPU, or reload anything.

Ownership ONLY: workloads/llama.cpp/observability_overhead/revision-rq4/transport-soa-20260908/host-transport-soa.patch and host-README.md. Qwen Next exclusively owns transport-soa.patch and README.md in the same directory, covering the device trampoline. Do not edit its files, existing tasks, BPF programs/examples, benchmark runners, paper, driver or worktrees. Target /home/yunwei37/workspace/gpu/bpftime-auto-warp is READ-ONLY at c4c83cd: its built runtime/source is frozen for queued Table1 measurement. Never modify it or its Git metadata.

Read AGENTS.md in workspace/main/revision-rq4. No file/content hashes/checksums/fingerprints/digests. Use ordinary textual diff without content-index headers to construct a unified source patch. You may copy the two relevant files into mktemp scratch and edit those copies with apply_patch, then save final patch using apply_patch. Installed helper via bash: /home/yunwei37/.codex/tmp/arg0/codex-arg029oUiU/apply_patch. Do not use shell file-writing tricks for code or patch files. bash/read enabled. Native apply_patch may be unavailable. Source reads, lightweight textual checks and non-mutating git apply --check are allowed. No extra verification campaign/gates.

Target files ONLY runtime/src/bpf_map/gpu/nv_gpu_ringbuf_map.cpp and .hpp; avoid touching other files unless you report a concrete necessity to root.
Root/Next shared ABI:
- existing transport modes0/1/2 remain unchanged.
- BPFTIME_GPU_RINGBUF_TRANSPORT=3 is opt-in under existing auto-warp environment convention, fixed at map construction; existing get_output_transport upload already propagates stored mode.
- All original per-thread 80-byte (or general value_size) records, capacities, return/drop semantics and callback ordering are retained. No sampling/one-event-per-warp substitution.
- mode3 headers are headers[N], each existing ringbuf_header is24 bytes. payload_base = data_buffer + N*24.
- record_stride = align8(8+value_size), record_words=record_stride/8.
- mode3 logical slot s, record word w (w0=data_size), thread tid maps to payload_base+((s*record_words+w)*N+tid)*8.
- errors remain at data_buffer+N*(24+max_entries*record_stride). Total allocation remains exactly old allocation.
- Each thread retains the encoded dirty publication protocol: dirty>>1=published tail, odd=producer busy; same head semantics, acquire/release rules. No new periodic drain/sync.

Implement host mode selection for3, correct header location in drain_data and get_stats, reconstruct each mode3 payload into bounded EXISTING local_buffer before invoking fn(payload,size). Preserve per-thread callback order, exact size and byte payload, head published after callback as before. Invalid size reports existing error. Modes0/1/2 must preserve existing behavior and layout. Sizes/offsets use checked existing allocation bounds and uint64 arithmetic. Unaligned source payload tails are device owner's job; host copies only size bytes for last word, not padding. Existing elem_lookup/update remain unsupported. Do not change the device trampoline; Next does that.

Prefer a minimal patch in these2 files, no new abstraction framework. README contains exact layout, unchanged semantics, dependency on Next device patch, base c4c83cd, source-only/unbuilt/unmeasured status, and apply commands root can use AFTER the queued mode2 run. No throughput or traffic-reduction claim. End with files changed and actual obstacles; no full-source self-audit loop.
