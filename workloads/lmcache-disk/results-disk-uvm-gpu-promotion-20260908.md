# Disk-backed UVM GPU promotion: five completed full-read runs

Completed 2026-09-08 at 12:11 PDT on RTX 5090 / CUDA 12.9 / NVIDIA
575.57.08 / Linux 6.15.11-061511-generic. Driver `dea1fefc`, client
`e07b4d69`. Five fresh processes complete the existing 256 MiB seven-stage
workload with `--gpu-promotion`. All five return zero. No manuscript edits.

| Stage | Earlier CPU-first median ms | GPU-promotion median ms |
|---|---:|---:|
| Initial GPU read | 0.336521 | 0.340561 |
| Register | 0.172915 | 0.170863 |
| First offload | 278.299030 | 286.335569 |
| CPU restore | 169.819063 | 192.396538 |
| Second release | 3.310861 | 3.114493 |
| First GPU restore | 170.601143 | 210.570854 |
| Repeated GPU read | 5.407625 | 0.250566 |

The columns are **separate sequential campaigns**, not randomized paired
samples. The earlier five runs are preserved in
[the CPU-first report](results-disk-uvm-restore-20260908.md), not repeated.
Repeated GPU read latency is 21.582x lower by the ratio of campaign medians
(new range 0.246044–0.262370 ms). This is not an application speedup.
First GPU restore is 23.429% higher by the same comparison, with new range
176.614956–227.996746 ms. The implementation improves subsequent GPU access
on this workload but does not show faster initial restoration; storage
variation and the added CPU-to-GPU migration are not isolated by these runs.
Initial versus repeated GPU read times also do not isolate caching effects.

## What changed

Default behavior remains CPU-first. For an explicitly enabled registered
range, the driver stages nonresident on-disk pages into CPU memory, then
uses its existing copy path to the faulting GPU instead of forcing CPU
residency. Root integrated the local Qwen implementation and its repair
moving the disk staging selection before the no-resident-source early return.
Root's bounded client glue issues ioctl 87 on the existing range-owning UVM
descriptor after registration, outside all timed restore stages. GPU reads
retain the same volatile source loads and full-buffer traversal.

The existing workload order is unchanged: initial GPU read, register,
offload, CPU restore, second release, GPU restore, repeated GPU read. Each
process uses a 268435456-byte local backing file opened with O_DIRECT;
actual recorded descriptor flags are `0140002`. The same backing path is
reused across fresh processes. O_DSYNC is off; no crash-durability or
cold-device-cache claim follows. Offloads report 65536 pages on disk and
zero pending/error pages. `bytes_expected` is the request size, not a
transport counter. Repeated log/raw-dump mode markers are not extra runs.

This remains an explicit, sealed read-only, same-virtual-address disk-UVM
primitive. Transport uses CPU staging, **not established NVMe-to-GPU P2P**.
It is not yet end-to-end LMCache KV integration, automatic pressure-triggered
offload, or a native/BPF storage-policy comparison. Existing LMCache policy
reports remain authoritative for those application-level comparisons.

## Reproduction and records

Raw path: `raw/disk-uvm-gpu-promotion-20260908.bG4MBx/`.
`run-five.sh` preserves the executed command, original module restoration
and exact owned-loader lifecycle. Its old PID values describe this run and
must be updated from the live process list before any future invocation.
The client command within each fresh process was:

```sh
gds-control/disk-uvm/disk_uvm_perf --gpu-promotion --size 256MiB \
  --backing-file /var/tmp/disk-uvm-gpu-promotion-20260908.njuwc4/backing.bin
```

Recompute the table from existing logs (no GPU execution):

```sh
python3 gds-control/disk-uvm/analyze_results.py \
  raw/disk-uvm-gpu-promotion-20260908.bG4MBx
```

`lifecycle.log` records five zero exits and
`DISK_RUN_EXIT=0 RESTORATION_OK=1`. Saved original UVM restored; GDS/KV loaders
3407414/3407415 both report `attached`. Post-run GPU is idle at 1 MiB.
The failed UVM-only build and successful full-module retry are both retained.
Large backing data and compiled modules/executables are excluded from Git.
