# RTX 5090 device-side Table 1 preflight 03

Status: **complete preflight; not a paper-facing performance result**.

This fresh one-block pp32 campaign ran on the RTX 5090 with driver 575.57.08
and the predeclared two-tool matrix. All five correctness cells and all five
timing cells passed on their first attempt. The independent analyzer reports
`complete=true`, one valid complete block, and no rejected cell.

## Correctness and engagement

Both exit-record implementations produced the same 720,896-record correctness
trace with nonzero timestamps, 220 selected launches, a 32-byte record, extent
88×256×1, and 22,528 unique logical coordinates. Their exact multiplicities
were 1,024@220, 1,024@44, and 20,480@22. All invalid-coordinate,
segment-mismatch, framing/drop, dirty, pending, and second-drain counters were
zero, and both collector gates passed.

In the timed exit-record pair, each implementation produced exactly 1,441,792
records, 44 selected launches, extent 128×256×1, 32,768 unique coordinates,
and multiplicity 32,768@44. gpubpf allocated 32,768 rings with exactly 44
entries each and had zero loss. This directly fixes the retained
`preflight-575-noncross-clock-02` capacity failure.

Both histogram implementations also engaged. gpubpf read all 1,048,576
configured entries (8,388,608 bytes), with 22,528 nonzero coordinates and
720,896 total exit probes; NVBit reported the same nonzero and total counts.
Every cell reproduced the exact deterministic text output before timing, and
every safety record passed without Xid, abnormal kernel log, leftover BPF
object, or surviving CUDA process.

## One-block timing observation

| Configuration | Prefill throughput (token/s) | Overhead vs baseline |
|---|---:|---:|
| baseline | 7,052.77 | 0.00% |
| gpubpf exit record | 131.60 | 98.13% |
| NVBit exit record | 135.63 | 98.08% |
| gpubpf histogram | 4,773.51 | 32.32% |
| NVBit histogram | 6,677.75 | 5.32% |

For exit records, `NVBit overhead - gpubpf overhead` is -0.057 percentage
points in this single block: the two extremely heavy per-thread record paths
are nearly identical at preflight scale. For histograms it is -27.000
percentage points, so the current gpubpf implementation is visibly slower in
this block. A one-block preflight does not estimate uncertainty and cannot be
used as the final Table 1 claim; the disjoint pp512 ten-block campaign is the
paper evidence gate.

## Command

```bash
env CUDA_VISIBLE_DEVICES=0 \
  PATH=/usr/local/cuda-12.9/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64 \
  /usr/bin/python3 -B run_revision_rq4.py \
  --phase preflight --runs 1 --pp 32 \
  --output-dir "$PWD/raw/preflight-575-noncross-clock-03" \
  --tools kernelretsnoop threadhist \
  --bpftime-root /home/yunwei37/workspace/gpu/bpftime-table1-575 \
  --bpftime-build-dir /home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575 \
  --gpu-thread-count 22528
```

The independent audit was regenerated with:

```bash
python3 -B analyze_revision_rq4.py raw/preflight-575-noncross-clock-03 \
  --output raw/preflight-575-noncross-clock-03/independent-audit.json
```

Strict read-only OpenCode session `ses_f94790233ffeklfqpcYb0CC1Vz`
cross-checked this README against both structured result files and returned
`VERDICT: PASS` with no concrete inconsistency.
