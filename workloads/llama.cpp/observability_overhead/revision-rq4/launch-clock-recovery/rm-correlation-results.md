# RM/PTIMER correlation canaries on RTX 5090

## Current verdict

The public 575 correlation control is reachable, but the conservative
public-data bracket does **not** pass the predeclared Phase-0 precision gate.
This is retained negative evidence, not a repaired `launchlate` result.

All runs used RTX 5090, NVIDIA 575.57.08, and Linux 6.15.11. The first two
canaries reached no timer sample. The third retained eight complete samples,
cleaned its private RM root, and observed no CPU-midpoint or PTIMER regression.

| Run | Outcome | Key evidence |
| --- | --- | --- |
| `raw/rm-correlation-575-01-canary` | Failed during device allocation | Root allocation succeeded; device allocation returned RM status `0x1b` (`NV_ERR_INSUFFICIENT_PERMISSIONS`); cleanup passed. |
| `raw/rm-correlation-575-02-root-canary` | Same failure under `sudo -n` | Shows that effective UID alone was not the missing condition; cleanup passed. |
| `raw/rm-correlation-575-03-access-canary` | Control path works; precision gate failed | 8/8 structurally valid samples, zero regressions/rejections, median outer ioctl width 14,087 ns and median conservative bracket 4,759 ns versus the predeclared <1,500 ns gate. |

The access failure was repaired by following the driver's own client/device
utility path: hold `/dev/nvidia0` open so `nv_is_gpu_accessible()` can match a
GPU fd, and set `NV0080_ALLOC_PARAMETERS::hClientShare` to the private root.
That repair is functional: the next canary allocated root/device/subdevice and
completed public control `0x20800406` eight times.

The remaining failure is precision, not access or sample accounting. The
outer xfer ioctl adds a userspace-to-kernel forwarding layer. A direct,
size-encoded RM control ioctl is the next safe diagnostic because the 32-byte
`NVOS54` payload fits the normal ioctl ABI. It must run as a fresh canary and
must retain the same `W/3`, 32 ns allowance, monotonicity, cleanup, and 1.5 us
gate. If that remains too wide, the public ABI path is rejected and the clean
fallback is the separately versioned driver control returning the selected
CPU endpoints.
