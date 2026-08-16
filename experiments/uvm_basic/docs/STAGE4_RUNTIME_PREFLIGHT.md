# Stage 4 Runtime Preflight

Status: `READY_FOR_MANUAL_STAGE4_MODULE_SWITCH`

Timestamp: 2026-08-15 21:32:56 UTC.

## Environment

| Item | Observed | Result |
|---|---|---|
| Kernel | `6.15.11-gpuext-gpuext` | PASS |
| GPU | NVIDIA A30 | PASS |
| Driver | `575.57.08` | PASS |
| GPU memory | 24,576 MiB total, 24,165 MiB free | PASS |
| Active compute processes | none | PASS |
| Result disk free | 45,928,431,616 bytes (about 42.8 GiB) | PASS, greater than 32 GiB |
| Host `MemAvailable` | 261,054,242,816 bytes | PASS |
| Stage 4 CUDA/BPF binaries | present and executable | PASS |
| Stage 4 runners | present and executable | PASS |

For the largest planned reduced-capacity calibration point (8 GiB effective capacity at 1.10x), the preflight estimates:

- managed working set: 9,448,928,051 bytes;
- device reserve: 15,675,162,624 bytes;
- required host margin including 16 GiB: 42,303,959,859 bytes.

The available host memory exceeds that requirement. The runtime safety checker was corrected to include the reserve buffer in this calculation rather than checking only managed working set plus 16 GiB.

## Module Identity

| Module | Path | srcversion | SHA256 |
|---|---|---|---|
| Loaded distribution UVM | `/lib/modules/6.15.11-gpuext-gpuext/updates/dkms/nvidia-uvm.ko` | `182AB87276B2337B4B1A4CD` | `5802edbad2d4f9f0304707e1d2ce9a2344e45ef83f8f683714879af30b4e2be6` |
| Custom gpu_ext UVM | `/home/peng/workspace/gpu_ext_private/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko` | `5446825F901EFEAA48651FC` | `c785d8bcf2a953c238140c03ce7d19fc5953c94ac6d159c73e040839effe57a4` |

Both modules report driver version `575.57.08` and matching kernel vermagic:

```text
6.15.11-gpuext-gpuext SMP preempt mod_unload modversions
```

However, the loaded srcversion is the distribution module, not the custom module. Searches for `gpu_mem_ops`, `uvm_bpf_call_gpu_page_prefetch`, and `uvm_bpf_trace_gpu_page_prefetch_decision` returned no visible hook symbol.

## Stop Decision

At the end of preflight, Stage 4A through Stage 4F had not been executed. The experiment required an explicitly authorized UVM-only switch through `scripts/SAFE_STAGE4_COMMANDS.sh`; no module operation was performed by the nonprivileged preflight itself.

Kernel-log Xid inspection is `UNAVAILABLE_NONPRIVILEGED`; no claim of a kernel-log Xid count is made. `nvidia-smi` remained healthy and reported no running process.

Machine-readable evidence is saved in `results/stage4/runtime_preflight.json`; complete raw command output is in `results/stage4/runtime_environment.txt`.

## Subsequent Runtime Window

After explicit authorization, the reviewed UVM-only switch was performed, the custom srcversion and hook symbols were verified, and all nine Stage 4A calibration runs were executed. Calibration failed because no selected eviction occurred at 1.05x or 1.10x, so Stage 4B through Stage 4F remained gated. The distribution module was then restored. The final runtime state is recorded in `results/stage4/runtime_status.json` and [REDUCED_CAPACITY_CALIBRATION.md](REDUCED_CAPACITY_CALIBRATION.md).
