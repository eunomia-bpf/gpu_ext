# RTX 5090 two-tool preflight 02: retained ring-full failure

Date: 2026-09-04 UTC

Status: **failed preflight; no timing value is eligible for Table 1**.

All five correctness cells passed. The timing cells for the baseline, both
`threadhist` implementations, and NVBit `kernelretsnoop` also completed their
local gates. The gpubpf `kernelretsnoop` timing cell failed its zero-loss
collector gate, so the preflight has zero complete paired blocks and the full
pp512 campaign remains barred.

The client itself exited zero. Its private loader exited one after reporting:

- 1,408,790 committed records versus 1,441,792 expected;
- exactly 33,002 full-ring drops;
- zero OOB, bad-size, or other drops;
- 32,768 unique coordinates and 44 selected launches;
- 32,689 coordinate-segment mismatches caused by the incomplete per-coordinate
  multiplicities; and
- zero dirty slots, pending records, or second-drain records.

This differs from retained preflight 01. The phase-specific layout repair
eliminated its undersized-slot OOB failure and allocated the intended 32,768
timing slots, but depth 16 did not buffer this burst until the concurrent
collector drained it. Any loss is disqualifying; the 33,002 records are not an
allowed tolerance and the local throughput is not reported as a result.

The independent analyzer exited two and records `complete=false`,
`valid_complete_blocks=0`, and exactly one rejected cell:
`gpubpf_kernelretsnoop`, block 1. The one-block `threadhist` point estimate is
preflight behavior only and is not a confidence interval or a formal result.

Every cell's before/after safety record is retained. After the failed cell the
GPU had no compute process, used 15 MiB, UVM had reference count zero, no
scheduler struct-ops object remained, and no Xid or abnormal kernel/journal
record was detected. The runner removed only its private shared-memory segment
after the owned client exited.

Do not retry this configuration unchanged and do not start the full campaign.
The next repair must guarantee zero loss at pp512 within the fixed 1,000 MiB
private segment without changing record semantics or weakening exact event and
launch equality against NVBit.
