# Expert marker smoke

Date: 2026-08-31

This is an implementation smoke, not a model experiment or performance sample.
It did not replace an NVIDIA module, attach struct_ops, or move GPU memory.

## Build

The marker-enabled llama.cpp branch built `llama-server` and `llama-cli` with
the existing CUDA Release configuration. `llama-cli --version` detected the
RTX 5090 and reported llama.cpp revision 7102. Dynamic symbol-table inspection
found global symbols for:

- `gpubpf_expert_tensor_layout`;
- `gpubpf_expert_route`; and
- `ggml_backend_sched_graph_compute_async`.

The new `extension/expert_buffering_trace` BPF object, skeleton, loader object,
and loader binary built through the repository Makefile. The C marker caller
built with `-Wall -Wextra -Werror`.

## Live uprobe smoke

The trace loader attached system-wide uprobes to the three symbols for eight
seconds. The marker caller then emitted one synthetic layout registration and
one route observation. The loader received:

```text
layout name=blk.7.ffn_gate_exps.weight base=1073741824
       total_bytes=564019200 per_expert_bytes=4406400
       n_experts=128 is_bias=0
route  graph=0 tensor_base=1073741824 expert_id=17
final  graphs=0 layouts=1 routes=1 dropped=0
```

`graph=0` is expected because the smoke calls only the two dedicated markers;
it does not invoke llama's graph-compute function with fake scheduler state.
The exact-model calibration must exercise the graph uprobe and requires
positive graph ordinals before route admission.

Disposition: **PASS** for marker symbol availability, BPF load/attach, argument
delivery, ring-buffer transport, and clean timed teardown. This does not yet
validate automatic tensor call sites, 36-layer calibration, or PMM decisions.

No file/content hashes, checksums, digests, or fingerprints were generated or
used.
