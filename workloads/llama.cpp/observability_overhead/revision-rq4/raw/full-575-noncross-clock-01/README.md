# RTX 5090 device-side Table 1 pp512 full attempt 01

Status: **failed campaign; zero complete paired blocks**.

This directory is immutable failure evidence. It must not be resumed,
reclassified, or used for a paper-facing overhead comparison. The runner
exited 2, and the independent analyzer reports `complete=false`, zero of ten
required complete blocks, and twenty rejected gpubpf cells.

## What passed

- The independently audited pp32 preflight gate was complete.
- All five fresh correctness cells passed their exact-output and engagement
  gates.
- All ten baseline timing cells, ten NVBit exit-record cells, and ten NVBit
  histogram cells completed and passed their individual gates.
- Each NVBit exit-record cell produced exactly 23,068,672 nonzero-timestamp
  32-byte records over 44 launches, extent 2,048×256×1, and 524,288 unique
  coordinates, each with multiplicity 44.
- Cleanup removed every private segment and CUDA process. The retained safety
  records show no Xid, kernel panic, abnormal kernel/journal entry, surviving
  struct_ops object, or driver reset. A `passed=false` safety record reflects
  the cell error described below, not an OS/GPU crash.

These individually valid baseline/NVBit cells are diagnostic only because no
block contains both valid gpubpf and NVBit arms.

## Failure

Every pp512 gpubpf timing workload aborted before the selected `rope_norm`
probe executed. Both policies reproducibly report the same failure while the
agent launches a different recompiled kernel:

```text
Unable to launch patched kernel _Z9mul_mat_qIL9ggml_type12ELi128ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii: CUDA_ERROR_INVALID_VALUE
Patched CUfunction attrs: local=0B shared=0B const=0B regs=249 max_tpb=256
```

The client exits by `SIGABRT` (`-6`). Consequently every gpubpf
`kernelretsnoop` collector sees zero committed/collected events and exits 1,
and every gpubpf `threadhist` readback is complete but contains zero nonzero
coordinates and zero exit probes. This is not the earlier ring-capacity
failure: the pp512 ring is successfully allocated as 524,288×44 with a
32-byte value, but the workload never reaches the selected hook. Therefore
this run does not yet validate the repaired ring under pp512 traffic.

The repeated signature across two distinct BPF programs and all ten blocks
localizes the immediate failure to the pp512 late-bootstrap PTX replacement /
launch path, not to either policy's callback logic. Further source inspection
is required before assigning the exact launch-argument or resource cause.

## Recorded outcomes

| Configuration | Valid timing blocks | Attempts | Geometric mean token/s |
|---|---:|---:|---:|
| baseline | 10 | 10 | 38,045.27 |
| gpubpf exit record | 0 | 10 | n/a |
| NVBit exit record | 10 | 10 | 144.30 |
| gpubpf histogram | 0 | 10 | n/a |
| NVBit histogram | 10 | 10 | 34,141.82 |

No paired effect or confidence interval exists for either task.

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 \
  PATH=/usr/local/cuda-12.9/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64 \
  /usr/bin/python3 -B run_revision_rq4.py \
  --phase full --runs 10 --pp 512 \
  --preflight-dir "$PWD/raw/preflight-575-noncross-clock-03" \
  --output-dir "$PWD/raw/full-575-noncross-clock-01" \
  --tools kernelretsnoop threadhist \
  --bpftime-root /home/yunwei37/workspace/gpu/bpftime-table1-575 \
  --bpftime-build-dir /home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575 \
  --gpu-thread-count 22528
```

Strict read-only OpenCode session `ses_f944a46f4ffeiFOsSz2nqo5XoT`
cross-checked this failure classification against the structured results and
representative raw logs and returned `VERDICT: PASS`.
