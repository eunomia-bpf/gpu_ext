# RTX 5090 matched device observability result

## Verdict

Run status: valid. The independent analyzer reports `complete=true`, a matching
independently complete preflight, all five correctness configurations true,
10/10 complete randomized blocks, and no rejected or retried cell.

Tested hypothesis: mixed. gpubpf and the matched NVBit adapter have nearly the
same baseline-relative cost for per-logical-thread exit records, with gpubpf
slightly slower. gpubpf has lower overhead for the final per-logical-thread
exit-count histogram. This is supporting evidence for RQ4 and a mechanism/task
boundary, not a general gpubpf-over-NVBit result.

## Commands

```sh
env CUDA_VISIBLE_DEVICES=0 \
  PATH=/usr/local/cuda-12.9/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64 \
  /usr/bin/python3 -B run_revision_rq4.py \
  --phase full --runs 10 --pp 512 \
  --preflight-dir raw/preflight-575-noncross-clock-04 \
  --output-dir raw/full-575-noncross-clock-02 \
  --tools kernelretsnoop threadhist \
  --bpftime-root /home/yunwei37/workspace/gpu/bpftime-table1-575 \
  --bpftime-build-dir /home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575 \
  --gpu-thread-count 22528

/usr/bin/python3 -B analyze_revision_rq4.py \
  raw/full-575-noncross-clock-02 \
  --output raw/full-575-noncross-clock-02/independent-audit.json
```

Both commands exited zero. The run used an RTX 5090, driver 575.57.08,
TinyLlama-1.1B Q4_K_M, llama.cpp build 7102 (`26836b27`), pp=512, tg=0,
and ten seed-1797 randomized blocks. The frozen plan named build 7101, but the
preflight and every full-run arm consistently used the same build-7102 binary;
this bookkeeping deviation does not create an H2H mismatch.

## Results

| Arm | Prefill token/s, geometric mean | Mean overhead vs paired baseline |
|---|---:|---:|
| no-probe control | 38056.928 | 0% |
| gpubpf exit records | 128.374 | 99.663% |
| matched NVBit exit records | 144.302 | 99.621% |
| gpubpf exit-count histogram | 36531.772 | 4.007% |
| matched NVBit exit-count histogram | 34136.772 | 10.301% |

The predeclared paired effect is `NVBit overhead - gpubpf overhead`:

- Exit records: -0.04185 percentage points, 95% bootstrap interval
  [-0.04355, -0.04029]. gpubpf is slightly slower; the expected gpubpf
  advantage is contradicted for this task on this setup.
- Exit-count histogram: +6.29351 points, interval [6.12507, 6.47076]. gpubpf
  has lower overhead for this task on this setup.

## Validity and scope

Every exit-record arm in every block produced exactly 23,068,672 nonzero
32-byte `(global_x, global_y, global_z, timestamp)` records from 44 selected
`rope_norm` launches: 524,288 coordinates each appeared 44 times. All drop,
pending, dirty, mismatch, invalid-coordinate, and collector error fields were
zero. Both histogram arms reported 23,068,672 samples, 524,288 nonzero logical
threads, and 44 selected launches; gpubpf additionally passed complete
1,048,576-entry / 8-MiB readback. All processes exited zero, deterministic
outputs matched, and per-cell safety/cleanup records contain no Xid, throttling,
UVM reference, struct_ops, abnormal kernel log, or surviving compute process.

The comparison uses custom matched adapters built with NVBit v1.8 and the same
selected kernel and observable, while retaining each system's native transport
(gpubpf per-thread rings versus NVBit `ChannelDev`). Throughput covers the
instrumented benchmark, not collector shutdown after the benchmark exits. The
performance runtime has GPU verification disabled. Therefore this run does not
support claims about verifier cost, all device policies, all workloads or
hardware, stock NVBit tools, or the still-invalid cross-clock `launchlate`
comparison.

The earlier `full-575-noncross-clock-01` remains immutable invalid evidence of
the repaired late-bootstrap over-substitution bug. Strict read-only OpenCode
review with the configured Qwen model accepted the scoped interpretation in
session `ses_f940dbd46ffeg833UsHzvZutnV`.
