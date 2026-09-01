# DeepSpeed ZeRO-Inference availability decision

Date: 2026-08-31

Disposition: **unavailable for the exact GPT-OSS-120B MXFP4 comparison; no run
was started and no performance result exists.**

## Required boundary

The proposed fallback had to preserve the exact public GPT-OSS-120B MXFP4
model and use the official DeepSpeed ZeRO-Inference mechanism. It could not
silently dequantize the model, leave the dominant expert storage outside ZeRO,
or add a custom paging implementation and still count as that baseline.

DeepSpeed's official ZeRO-Inference documentation defines the path as ZeRO
stage 3 parameter offload to CPU or NVMe:

- <https://www.deepspeed.ai/2022/09/09/zero-inference.html>
- <https://huggingface.co/docs/transformers/main_classes/deepspeed>

## Source-backed incompatibility

The exact model config declares `model_type="gpt_oss"`, 36 layers, 128 local
experts, and `quant_method="mxfp4"`. In the pinned Transformers 5.16.1 source,
`integrations/mxfp4.py` converts each expert projection to a Triton tensor,
deletes `gate_up_proj` or `down_proj` from `module._parameters`, and attaches
the replacement as an ordinary attribute. The relevant installed-source lines
are 525--544 in:

```text
workloads/moe-infinity/.venv/lib/python3.12/site-packages/transformers/
  integrations/mxfp4.py
```

The same pinned quantizer also states that using full-precision kernels on the
GPU dequantizes the model to BF16 (`quantizer_mxfp4.py`, lines 180--203). That
would expand the representation and violate the exact-model gate.

ZeRO-3's parameter inventory cannot manage expert storage that is no longer an
`nn.Parameter`. It could at most offload the model's remaining ordinary
parameters while the dominant packed expert tensors stay outside the official
mechanism. Requiring those tensors in ZeRO's partition inventory is therefore
the correct engagement gate, but the official MXFP4 path cannot pass it.

## Decision and evidence boundary

- Do not install or launch DeepSpeed for this exact-model candidate: a 120B
  preflight cannot change the source-level ownership mismatch.
- Do not use a BF16-dequantized model, a different MoE model, Accelerate
  `device_map`, or custom expert paging as a substitute after observing this
  incompatibility.
- Do not report setup, source inspection, or dependency compatibility as a
  correctness or performance result.
- No GPU request, timing sample, output sample, or offload-engagement sample
  exists for DeepSpeed in this revision workspace.
- Preserve this named reason for the revision's artifact-availability record
  and continue with the promised runnable baselines and policy
  reimplementations on their own reviewed protocols.

No file/content hashes, checksums, digests, or fingerprints were generated or
used for this decision. Git revisions remain ordinary version bookkeeping.
