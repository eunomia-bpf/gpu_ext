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
objects contain the expected sm_120/sm_120a cubins.  Exact dependency and
module hashes are frozen in `current-requirements.txt` and
`artifacts-current.json`.

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
  environment, GPU, storage, and struct_ops checks.

Run the CPU-only checks and read-only admission with:

```bash
cc -std=c11 -Wall -Wextra -Werror -fsyntax-only \
  -I../../../gpu_ext-kernel-610/kernel-open/nvidia-uvm \
  -I../../../gpu_ext-kernel-610/kernel-open/common/inc test_uvm_tools_abi.c
.venv/bin/python -m unittest -v test_offline.py
.venv/bin/python run_moe_head_to_head.py admit --full-hashes \
  --output raw/admission-full-hashes.json
```

The compile-only ABI check passes against the pinned 610 headers: event 14,
72-byte V2 queue entries, 528-byte control area, ioctl layouts, and eviction
field offsets match the monitor. This does not establish runtime delivery.

After admission is green, the two authorized execution stages are:

```bash
.venv/bin/python run_moe_head_to_head.py preflight \
  --output raw/correctness-preflight
.venv/bin/python run_moe_head_to_head.py run \
  --preflight raw/correctness-preflight --output raw/full-run
```

`run` follows the frozen eight-attempt schedule, atomically retains invalid
attempts, and stops immediately after exactly five valid complete blocks. It
then reuses the frozen 10,000-by-5 index matrix for the preregistered paired
geometric-mean interval and the block-paired TTFT interval.

The recorded pre-run deviation in `plan.md` moves all four cells to the same
610.43.02 stack and identifies the workspace NVMe by filesystem UUID. Build
the pinned port in the sibling `gpu_ext-kernel-610` checkout; the existing
575 source checkout is not modified. The custom 610 modules are not loaded.
Unrelated SGLang processes still own about 31 GiB on GPU 0. The harness never
signals them or clears unknown struct_ops state. No GPU correctness or
performance run is accepted until isolation and real preflight succeed.
