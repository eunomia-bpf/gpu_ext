# MoE-Infinity revision baseline

This directory stages the MoE research-system baseline named in revision R1.
Upstream dependencies and build products remain under ignored `deps/`; raw
experiment outputs will be kept under ignored `raw/`.

Current preparation pins EfficientMoE/MoE-Infinity commit
`b766f8f1f6379fac6cd23594713ba6f4c7650ad9`. Its source exposes an explicit
Blackwell build path:

```bash
MOE_ENABLE_SM120=1 MOE_ENABLE_SM90=0 CUTLASS_DIR=<cutlass> \
  pip install --no-build-isolation -e .
```

The compile-only historical smoke used the vLLM Python 3.10 environment.  The
paper-facing environment is now isolated under ignored `.venv/`: Python
3.12.3, PyTorch 2.13.0+cu129, Transformers 5.16.1, and the official
`sglang-kernel` 0.4.6.post1+cu129 wheel.  MoE-Infinity was rebuilt and
installed from the pinned source with CUDA 12.9; all six extensions import
with CUDA hidden, `uv pip check` accepts all 108 packages, and the CUDA
objects contain the expected sm_120/sm_120a cubins. Exact dependencies, source
revisions, build flags, filenames, sizes, and module versions are recorded in
`current-requirements.txt` and `artifacts-current.json` without checksum gates.

The exact `openai/gpt-oss-120b` MXFP4 snapshot and matching public GGUF are now
fully staged on workspace NVMe. A complete content audit verified all 15 HF
weight shards, seven tokenizer/config files, and the 63.4 GB GGUF. The admitted
MoE view contains only those 22 files and excludes `original/*` and `metal/*`.

Proposal 2 revision 3 in `plan.md` passed three independent review rounds. Its
offline implementation now includes:

- frozen public prompts, configuration/prompt schedules, and bootstrap indices;
- a side-effect-free MoE cache-counter endpoint and exact source patch;
- one combined host stride-prefetch + LFU struct_ops object with monotonic hook
  counters and an ownership-safe loader;
- a UVM Tools V2 monitor that counts actual `UvmEventTypeEviction` records;
- a fail-closed admission/command harness with exact model, binary, source,
  environment, GPU, storage, loaded-UVM-interface, and struct_ops checks.

Run the CPU-only checks and read-only admission with:

```bash
cc -std=c11 -Wall -Wextra -Werror -fsyntax-only \
  -I../../../gpu_ext-kernel-610/kernel-open/nvidia-uvm \
  -I../../../gpu_ext-kernel-610/kernel-open/common/inc test_uvm_tools_abi.c
.venv/bin/python -m unittest -v test_offline.py
.venv/bin/python run_moe_head_to_head.py admit \
  --output raw/admission.json
```

The compile-only ABI check passes against the pinned 610 headers: event 14,
72-byte V2 queue entries, 528-byte control area, ioctl layouts, and eviction
field offsets match the monitor. This does not establish runtime delivery.

After admission is green, the two authorized execution stages are:

```bash
.venv/bin/python run_moe_head_to_head.py preflight --attempt 1
.venv/bin/python run_moe_head_to_head.py run \
  --preflight raw/repaired-preflight/attempt-01 --output raw/full-run
```

`run` follows the frozen eight-attempt schedule, atomically retains invalid
attempts, and stops immediately after exactly five valid complete blocks. It
then reuses the frozen 10,000-by-5 index matrix for the preregistered paired
geometric-mean interval and the block-paired TTFT interval.

The recorded pre-run deviation in `plan.md` moves all four cells to the same
610.43.02 stack and identifies the workspace NVMe by filesystem UUID. Build
the pinned port in the sibling `gpu_ext-kernel-610` checkout; the existing
575 source checkout is not modified. The unrelated SGLang processes later
exited without intervention. Only the idle UVM module was replaced temporarily;
GDM continues to use the matching official core. Admission refuses stock UVM
because it requires live BTF proof of the exact custom `gpu_mem_ops` ABI and
policy kfuncs. The harness never signals foreign processes or clears unknown
struct_ops state.

Runtime update: full admission subsequently passed with the custom UVM, but
the first real preflight failed on MoE-Infinity's fixed 256-row expert buffer
when the frozen 512-token warm-up routed 353 rows to one expert. No request or
timing completed, and that original protocol remains closed; see
`runtime-preflight.md`.

At the author's direction, a new protocol now carries a disclosed repair in
`row-chunking.patch`. It keeps the same model, prompt, routing, and 256-row
workspace, but executes a large per-expert route in stable consecutive chunks
and concatenates the outputs in original row order. The active Python 3.12
repaired `_store` builds for sm_120, 36 offline tests pass, and the standalone
GPU numerical gate executes the actual repaired `MoEMLP::forward()` at
1/256/257/353 rows against a same-parameter reference. All four comparisons
passed at `rtol=1e-2`, `atol=1e-2`, with zero observed maximum absolute and
relative error. Independent re-review approved the repaired protocol after the
runner was tightened to accept only the three fixed, internally consistent
preflight attempt directories. The 120B preflight is now authorized; timing
remains unauthorized until one complete preflight passes every correctness and
engagement gate.
