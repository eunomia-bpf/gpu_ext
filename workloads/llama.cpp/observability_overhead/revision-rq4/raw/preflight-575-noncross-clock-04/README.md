# RTX 5090 device observability preflight 04

## Verdict

This is a valid real preflight and a dependency gate, not a paper performance
result. The independent analyzer reports all five correctness arms true, one
complete timing block, and no rejected cells. It authorizes a fresh disjoint
pp512 full campaign for the same two tools.

The run uses bpftime runtime commit `478d10b` and gpu_ext harness commit
`5be7be1`. The runtime registers only requested hook targets for late-bootstrap
launch substitution; unrelated kernels continue through the application's
original CUDA functions.

## Commands

```sh
env CUDA_VISIBLE_DEVICES=0 \
  PATH=/usr/local/cuda-12.9/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64 \
  /usr/bin/python3 -B run_revision_rq4.py \
  --phase preflight --runs 1 --pp 32 \
  --output-dir raw/preflight-575-noncross-clock-04 \
  --tools kernelretsnoop threadhist \
  --bpftime-root /home/yunwei37/workspace/gpu/bpftime-table1-575 \
  --bpftime-build-dir /home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575 \
  --gpu-thread-count 22528

/usr/bin/python3 -B analyze_revision_rq4.py \
  raw/preflight-575-noncross-clock-04 \
  --output raw/preflight-575-noncross-clock-04/independent-audit.json
```

Both commands exited zero. Hardware admission recorded RTX 5090 and driver
575.57.08.

## Single-block observations

| Arm | Prefill token/s |
|---|---:|
| baseline | 7074.944 |
| gpubpf exit records | 131.231 |
| matched NVBit exit records | 132.662 |
| gpubpf exit-count histogram | 5464.883 |
| matched NVBit exit-count histogram | 6663.095 |

The exit-record paired difference is -0.020 percentage points under the
predeclared `NVBit overhead - gpubpf overhead` definition. The histogram
difference is -16.936 points. A one-block preflight has no uncertainty estimate
and supports no stable performance claim.

The next result gate is a new pp512, ten-block full campaign. Its gpubpf cells
must exit normally, contain no patched-kernel launch error, engage only the
selected `rope_norm` exit hook, and pass the existing exact event-count,
coordinate-multiplicity, losslessness, output, and cleanup checks. The retained
`full-575-noncross-clock-01` campaign remains invalid failure evidence.

Strict read-only OpenCode review with the configured Qwen model, corrected to
distinguish the selected `rope_norm` target from the unselected `mul_mat_q`,
returned PASS in session `ses_f9435cf39ffe5Q6NDrH3lffG8O` with these same
pp512 gates.
