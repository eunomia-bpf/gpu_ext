# RTX 5090 GPU-storage asynchronous policy experiment

Date: 2026-09-06  
GPU: NVIDIA GeForce RTX 5090  
Driver: 575.57.08  
Kernel: 6.15.11-061511-generic  
Storage: local NVMe ext4 (`/dev/nvme1n1p2`)  
Transport: CUDA 12.9 cuFile asynchronous API in compatibility mode

## What this cell measures

The executor submits a deterministic mix of 64 logical 24 MiB KV-storage
requests. Each request gets exactly one scheduling decision and is then split
into 16 MiB and 8 MiB cuFile operations. Five rotated blocks compare:

- `fifo`: every request is submitted in arrival order on one stream;
- `native`: the storage-aware policy runs directly in the executor;
- `bpf`: the identical policy runs through the live gpubpf UVM hook, while the
  same trusted executor owns file descriptors, GPU pointers, CUDA streams, and
  cuFile submission.

The policy maps each logical request to `SUBMIT_NOW`, `DEFER`, or `RECOMPUTE`.
Native and BPF made the same decisions in every run: 40 immediate submissions,
16 deferred requests, and 8 recomputations. Thus both policy arms issued 56
logical chunks (112 cuFile operations), versus FIFO's 64 chunks (128 cuFile
operations).

## Results

| Metric (median of 5) | FIFO | Native policy | gpubpf policy |
|---|---:|---:|---:|
| Elapsed time (s) | 1.028 | 0.822 | 1.048 |
| Physical-I/O throughput (GiB/s) | 1.459 | 1.597 | 1.253 |
| Urgent-read p50 (us) | 365,710 | 370,581 | 484,055 |
| Urgent-read p99 (us) | 379,518 | 508,950 | 690,294 |
| Decision cost per request (us) | 0.040 | 0.063 | 1.005 |

The storage path is noisy across blocks, so unpaired medians must not be read as
a stable native-versus-BPF throughput difference. The paired BPF/native elapsed
differences were -2.97%, -6.98%, +27.59%, -50.72%, and +87.86%; their median is
-2.97%. The stable mechanism-level observation is that the UVM ioctl plus BPF
decision adds about 0.94 us per 24 MiB logical request over the same native
decision. This is far below the per-request storage service time in this cell.

This cell demonstrates that gpubpf can control real asynchronous cuFile
submissions and reproduce the native policy's actions. It does not claim direct
NVMe-to-GPU P2P on this RTX 5090: cuFile used its compatibility path. It also
does not yet replace the existing five-arm LMCache end-to-end experiment; the
LMCache adapter remains the next integration step.

Raw records are in
`gds-control/raw/gds-policy-campaign-20260906.jsonl`; the generated median
summary is in
`gds-control/raw/gds-policy-campaign-20260906-summary.json`.
