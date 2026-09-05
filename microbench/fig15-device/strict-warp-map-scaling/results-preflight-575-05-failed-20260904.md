# STRICT warp-map scaling preflight attempt 05: retained CUDA-init failure

Date: 2026-09-04
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08
Result directory: `raw/strict-warp-scaling-preflight-575-05`

This attempt is **invalid and contributes no sample**. The runner wrote the
environment inventory and the frozen 20-cell schedule and started the first
cell, `shape-32-block-01-order-01-noop`. That cell's loader constructed the
private shared-memory namespace and started the syscall server, which then
reported `Unable to initialize CUDA from syscall server side: 3`
(`NV_ERR_NO_MEMORY`) and aborted through `std::runtime_error`. No BPF program
load, verifier call, target transformation, patched-module load, attach, kernel
launch, or map readback occurred.

The retained directory contains `environment.txt`, `schedule.tsv`, and only the
first cell's `loader.log`. There is no `execution.tsv`, application log,
timing record, map-effect record, or analyzer output, so no arm, shape, or
block was sampled and nothing from this attempt enters any comparison.

Attempt 04 stopped at the same syscall-server CUDA-initialization error.
Attempt 01 stopped after its first cell and attempts 02 and 03 wrote only their
schedule; all four remain retained and unsampled. Attempt 06 restarted the
complete preflight design in fresh processes.
