# Existing-evidence audit

The experiment was added only after checking the active revision checklist and
the current gpu_ext/bpftime device examples.

| Existing path | What it already proves | Missing for Reviewer A's raw-state question |
| --- | --- | --- |
| `gpu_ext/workloads/bpftime-device-smoke` | Exact per-thread return counts and a separate strict SIMT rejection | Values are counters in a per-thread array, not raw non-composable records. |
| `bpftime-table1-575/example/gpu/threadhist-gpu-kernel-shared-map` | Device coordinate helpers feeding a histogram | Coordinates are reduced into bins; the host cannot recover individual tuples. |
| `bpftime-table1-575/example/gpu/kernel_trace` | A GPU ring-buffer example with one sampled coordinate/timestamp | It is an unbounded demo loop with no CUDA truth join, multi-scale campaign, complete/drop accounting, overflow rejection, private-segment cleanup, or formal repetitions. |
| `bpftime-table1-575/example/gpu/gpu_shared_map` | Device write and host lookup for GPU array/hash maps | It documents last-writer-wins contention and does not validate a raw stream. |
| `gpu_ext/docs/revision-completion-checklist.md` | Tracks strict device counts, Table 1, policy ports, and trampoline scaling | It does not contain a raw cross-layer map experiment. |

The relevant implemented ABI is source-visible in
`bpftime-table1-575/runtime/include/bpftime_gpu_ringbuf.h`, which exports exact
committed/collected/pending and four drop categories, and in
`runtime/src/bpf_map/gpu/nv_gpu_ringbuf_map.cpp`, which drains every retained
fixed-size record.  The device helper in
`attach/nv_attach_impl/trampoline/default_trampoline.cu` maintains one bounded
ring per configured GPU thread and increments explicit full/out-of-range/
bad-size/other counters.  The independent aggregate uses
`runtime/src/bpf_map/gpu/nv_gpu_per_thread_array_map.cpp`.

Therefore the missing experiment can use the existing ABI without modifying
bpftime.  It must still avoid upgrading this concrete global/host-visible path
into a claim that automatic hierarchical placement or on-chip shards were
tested.

