# LMCache policy-ioctl GIL handoff ablation

The native decider executes Python with the GIL held. The BPF decider uses
`ctypes.CDLL`, which releases the GIL during ioctl, while the adapter's common
decision lock remains held. With 64 reader threads and repeated write
feedback, GIL reacquisition and lock contention may amplify a short syscall.
This is a hypothesis, not a conclusion from the previous adverse results.

One default-off constructor choice, `LMCACHE_GDS_IOCTL_KEEP_GIL=1`, uses
`ctypes.PyDLL` for the same libc ioctl. The ABI, BPF program, decision calls,
native policy and live-feedback executor remain unchanged. This does not
remove BPF calls or replace actual I/O. It may instead harm concurrency when
an ioctl is slow; the new comparison must retain that possibility.

Run five rotated blocks of four arms on the current polling executor:
FIFO, native, BPF with ordinary GIL release, and BPF with GIL retained. The
second BPF arm is a mechanism ablation, not an additional research baseline.
64 reads / 96 writes, 24 MiB objects, 2/4 ms spacing, 4096 MiB GPU pool and
fresh child processes match the prior workload. Use the existing runner's
`--single-cell` route, setting `--config`, `--block`, `--position`, and a
distinct `--cell-dir` for each arm. Set the environment variable explicitly
to zero or one per child; output directory names retain that choice.

Raw outputs:
`../raw/gds-mixed-gil-handoff-575-20260907-five-block/block-NN/position-NN-ARM/`.
Each contains `result.json` and the ordinary runner log. The run command is
the existing `current-venv/bin/python run_gds_mixed_backend.py` with
`--single-cell --policy-variant live-feedback --reads 64 --writes 96
--gds-buffer-size-mib 4096` and the per-cell arguments above. Native/FIFO and
BPF-release set `LMCACHE_GDS_IOCTL_KEEP_GIL=0`; BPF-keep sets it to one.

Compare within-block BPF-keep/BPF-release to test the handoff optimization,
BPF-keep/native for mechanism cost, and native/FIFO for policy benefit.
Record scheduled-arrival read p50/p99, write throughput, total bandwidth,
decisions and every failure. Report all paired effects, not a selected best
cell; p99 of 64 reads is the maximum, and five-pair ranges are not confidence
intervals. No additional preflight, clock, performance-threshold or retry
gate. Preserve all old data. Do not conflate this with the separately pending
event-driven executor or hardware NVMe/GPU P2P.

## Invocation correction

The initial 20 attempts used relative `--cell-dir` paths. LMCache returned
`Unable to detect fstype` before any I/O request, and all attempts exited 2.
Those raw records and logs remain in the originally named directory. The
corrected run uses absolute `--cell-dir` paths under the separate
`gds-mixed-gil-handoff-575-20260907-five-block-absolute` directory. No completed
performance sample is discarded or repeated; this corrects an actual
invocation error, not a performance/correctness threshold.
