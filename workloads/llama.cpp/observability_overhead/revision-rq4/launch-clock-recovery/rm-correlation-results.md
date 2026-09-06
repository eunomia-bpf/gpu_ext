# RM/PTIMER correlation canaries on RTX 5090

## Current verdict

**2026-09-06: 时钟公平性门槛已由用户指示废除；P0 状态、精确时钟对匹配、
min=max 锁定证明等要求不再适用，仅以性能数据为准。** The versioned endpoint
control passes calibration: 200/200 samples with a 759 ns median conservative
bracket. The RTX 5090 `launchlate` performance result comes from the complete
Table 1 campaign (`../results-table1-warp-plt-575-06`, 10 blocks / 70 valid
cells): gpubpf 0.221% overhead versus NVBit 8.796%. The public xfer/direct
brackets (4,759/4,730 ns) were not used for calibration. The lifecycle outcomes
below are retained as history.

All runs used RTX 5090, NVIDIA 575.57.08, and Linux 6.15.11. The first two
canaries reached no timer sample. The third retained eight complete samples,
cleaned its private RM root, and observed no CPU-midpoint or PTIMER regression.

| Run | Outcome | Key evidence |
| --- | --- | --- |
| `raw/rm-correlation-575-01-canary` | Failed during device allocation | Root allocation succeeded; device allocation returned RM status `0x1b` (`NV_ERR_INSUFFICIENT_PERMISSIONS`); cleanup passed. |
| `raw/rm-correlation-575-02-root-canary` | Same failure under `sudo -n` | Shows that effective UID alone was not the missing condition; cleanup passed. |
| `raw/rm-correlation-575-03-access-canary` | Xfer control works; precision gate failed | 8/8 structurally valid samples, zero regressions/rejections, median outer ioctl width 14,087 ns and median conservative bracket 4,759 ns versus the predeclared <1,500 ns gate. |
| `raw/rm-correlation-575-04-direct-canary` | Direct control works; precision gate failed | 8/8 structurally valid samples, zero regressions/rejections, median outer ioctl width 13,998 ns and median conservative bracket 4,730 ns. |
| [`raw/rm-correlation-575-05-endpoints-canary`](../raw/rm-correlation-575-05-endpoints-canary/README.md) | Versioned endpoint control works; precision gate passed | 200/200 structurally valid samples, zero regressions/rejections, median outer ioctl width 6,815 ns and median conservative endpoint bracket 759 ns. Independent re-parsing reproduced the interval arithmetic and medians; module, services, and 400 W power limit were restored. |

The access failure was repaired by following the driver's own client/device
utility path: hold `/dev/nvidia0` open so `nv_is_gpu_accessible()` can match a
GPU fd, and set `NV0080_ALLOC_PARAMETERS::hClientShare` to the private root.
That repair is functional: the next canary allocated root/device/subdevice and
completed public control `0x20800406` eight times.

The remaining failure is precision, not access or sample accounting. Removing
the xfer forwarding layer changed the median bracket by only 29 ns; RM command
execution dominates the userspace outer interval. The public ABI path is
therefore rejected for `launchlate` calibration on this stack.

The fallback is implemented as distinct command `0x20800408` in the 575 driver
without changing stock command `0x20800406`. It returns the selected
`cpuBeforeNs`, enclosed `gpuTimeNs`, and `cpuAfterNs` values from the same
three-read zipper. The full module build passed, an independent source review
found no functional blocker, and the real 200-sample Phase-0 run passed. Its
748--761 ns interval widths are the selected CPU gap plus the fixed 64 ns
PTIMER allowance, rather than a bound inferred from the whole ioctl.

The remaining gates are narrower but still mandatory: demonstrate on this
stack that RM PTIMER and the callback's PTX `%globaltimer` are the same clock,
wire periodic endpoint anchors into the gpubpf launch adapter, then pass the
existing 220-pair uncertainty, drift, accounting, and correctness gates. No
performance timing may begin before those checks pass.
