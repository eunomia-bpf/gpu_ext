# RTX 5090 asynchronous GDS transport reference

Date: 2026-09-06

This is a performance-first transport reference for the executor that will be
shared by the FIFO, matched-native, and gpubpf storage-control arms. It is not
yet a result for a gpubpf policy.

## Host and workload

- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB
- driver: 575.57.08
- tool: CUDA 12.9 `gdsio` 1.12
- filesystem: local ext4 on `/dev/nvme0n1p1`
- GPU memory: `cudaMalloc`
- one worker, 1 GiB file, 24 MiB transfers, five-second requested duration
- no correctness or clock-precision gate was applied

## Results

| cuFile path | direction | throughput (GiB/s) | average latency (us) | operations |
|---|---:|---:|---:|---:|
| async stream (`-x 5`) | write | 3.975 | 5,869.8 | 819 |
| async stream (`-x 5`) | read | 1.934 | 12,053.8 | 355 |
| batch stream (`-x 7 -b -B 8`) | write | 4.090 | 5,685.9 | 1,000 |
| batch stream (`-x 7 -b -B 8`) | read | 1.917 | 12,131.1 | 1,000 |

The standalone batch path (`-x 6 -b -B 8`) returned `Error in IO Batch
Submit` on this host. The batch-stream path ran and is therefore the immediate
candidate for the independent background lane. At this 24 MiB size its raw
bandwidth is close to the ordinary async-stream path; the policy experiment
must test mixed urgent reads and background writes, where ordering rather than
single-flow bandwidth is the intended benefit.

## Commands

```text
gdsio -D <scratch> -d 0 -m 0 -w 1 -s 1G -i 24M -x 5 -I 1 -T 5
gdsio -D <scratch> -d 0 -m 0 -w 1 -s 1G -i 24M -x 5 -I 0 -T 5
gdsio -D <scratch> -d 0 -m 0 -w 1 -s 1G -i 24M -x 6 -b -B 8 -I 1 -T 5
gdsio -D <scratch> -d 0 -m 0 -w 1 -s 1G -i 24M -x 6 -b -B 8 -I 0 -T 5
gdsio -D <scratch> -d 0 -m 0 -w 1 -s 1G -i 24M -x 7 -b -B 8 -I 1 -T 5
gdsio -D <scratch> -d 0 -m 0 -w 1 -s 1G -i 24M -x 7 -b -B 8 -I 0 -T 5
```
