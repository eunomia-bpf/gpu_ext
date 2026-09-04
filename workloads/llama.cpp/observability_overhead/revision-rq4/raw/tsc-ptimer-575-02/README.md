# TSC/PTIMER control attempt 02

This is the corrected real RTX 5090 stock-driver control.  It pins CPU 23,
requests 16 public RM TSC/PTIMER pairs per batch, discards the two edge pairs,
and applies the source-backed shortest-of-three zipper bound.  All 210 interior
samples were accepted with zero rejection, regression, or migration and RM
cleanup completed.  The 7,073 ns median bracket exceeds the unchanged 1,500 ns
gate, so the result is a valid negative control and no dependent GPU identity or
performance cell ran.

Files:

- `records.jsonl`: 210 sample records and one summary (40,011 bytes)
- `stderr.log`: empty; the control emitted no diagnostic error (0 bytes)

Replay from the `revision-rq4` directory with:

```bash
python3 launch-clock-tsc/analyze_tsc_ptimer.py \
  raw/tsc-ptimer-575-02/records.jsonl
```

