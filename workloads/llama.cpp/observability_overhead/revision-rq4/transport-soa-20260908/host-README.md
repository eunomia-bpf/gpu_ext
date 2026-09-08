# host-transport-soa.patch — host-only mode-3 host side (record-preserving transposed ringbuf transport)

Status: source-only, unbuilt, unmeasured. Base: `c4c83cd` of
`/home/yunwei37/workspace/gpu/bpftime-auto-warp`. The c4 runtime source
remains frozen for the queued Table1 (mode 2) measurement; HEAD
`88c2949` adds test-only changes (bpftime-verifier fixtures) and does not
touch the ringbuf transport.

## Requires the companion device patch

This host patch alone does not make mode 3 usable. The device trampoline
that writes the transposed layout is owned by Qwen Next and lives in
`transport-soa.patch` (device-only, this directory). Apply both before any
mode-3 build or run.

## Selection (opt-in)

`BPFTIME_GPU_RINGBUF_TRANSPORT=3` is opt-in under the existing auto-warp
environment convention (non-empty `BPFTIME_GPU_AUTO_WARP_EXECUTION` other
than "0") and is fixed at map construction. The stored level propagates to
the device map-info upload through the existing `get_output_transport()`.

## Mode 3 layout (total allocation unchanged)

- `record_stride = align8(8 + value_size)`, `record_words = record_stride / 8`
  (word 0 = payload size).
- `entry_size = 24 + max_entries * record_stride` and
  `data_size = entry_size * thread_count + sizeof(ringbuf_error_counters)`,
  exactly the existing values.
- Headers: `data_buffer + tid * 24` (existing 24-byte `ringbuf_header`).
- Payload base: `data_buffer + thread_count * 24`.
- Transposed word position: slot `s`, word `w`, thread `tid` is at
  `payload_base + ((s * record_words + w) * thread_count + tid) * 8`.
- Errors: `data_buffer + thread_count * entry_size` (unchanged offset).

## Unchanged semantics

- Modes 0/1/2 keep their existing layout and byte-for-byte behavior.
- Encoded dirty publication is retained: `dirty >> 1` = published tail,
  odd word = in-progress producer (skipped); same head, acquire/release
  rules, and per-thread callback order.
- Exact records preserved: every per-thread record (80-byte or general
  `value_size`), all capacities, and all return/drop semantics are
  retained; there is no sampling or one-event-per-warp substitution. Each
  mode-3 record is reassembled from the transposed words into the bounded
  existing `local_buffer` before `fn(payload, size)`; the host copies
  exactly `size` payload bytes (last word partial, never padding).
- Invalid size still returns `-EMSGSIZE`; head is published after the
  callback as before. `elem_lookup`/`elem_update` remain unsupported.
- Unaligned source payload tails (size not a multiple of 8) are handled by
  the device trampoline in the companion device patch.

## Files changed (host)

- `runtime/src/bpf_map/gpu/nv_gpu_ringbuf_map.cpp`
- `runtime/src/bpf_map/gpu/nv_gpu_ringbuf_map.hpp`

## Apply (root, after the queued mode 2 run)

    cd /home/yunwei37/workspace/gpu/bpftime-auto-warp
    git apply --check /home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/observability_overhead/revision-rq4/transport-soa-20260908/host-transport-soa.patch
    git apply /home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/observability_overhead/revision-rq4/transport-soa-20260908/host-transport-soa.patch
    git apply /home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/observability_overhead/revision-rq4/transport-soa-20260908/transport-soa.patch

The last command is the companion device patch; then build and measure.

## Runner exposure (source-only)

`run_table1_perf.py` now accepts `--auto-warp-transport 3`
(`AUTO_WARP_TRANSPORT_TRANSPOSED`); the default stays 1, the original
seven-arm mode and modes 0/1/2 are untouched. The runner only propagates
the chosen integer through the existing `BPFTIME_GPU_RINGBUF_TRANSPORT`
environment key, so mode 3 remains inert until the host and device patches
are applied, built, and measured.
