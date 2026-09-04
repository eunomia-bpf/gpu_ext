# TSC/PTIMER control attempt 01

This retained real RTX 5090 control used the first conservative implementation,
which bounded each selected zipper interval by the entire span between adjacent
returned TSC midpoints.  It completed 210/210 samples with cleanup but reported
a 21,089 ns median precision bracket and failed the 1,500 ns gate.  This attempt
is superseded by attempt 02 because the NVIDIA source permits a tighter still
conservative shortest-of-three bound.  It contributes no launch-latency or
performance sample.

Files:

- `records.jsonl`: 210 sample records and one summary (40,221 bytes)
- `stderr.log`: empty; the control emitted no diagnostic error (0 bytes)

