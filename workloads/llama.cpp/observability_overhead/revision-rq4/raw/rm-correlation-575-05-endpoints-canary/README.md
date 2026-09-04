# Endpoint-v1 Phase-0 canary

This is a fresh calibration-only run on RTX 5090, NVIDIA 575.57.08, and Linux
6.15.11. The candidate core module was built from driver commit `86e7e0dd` and
exposed versioned control `0x20800408`; the stock `0x20800406` command remained
unchanged. The probe used the direct NVOS54 transport and requested 200
endpoint samples.

All 200 samples were structurally valid. There were zero rejected samples,
CPU-midpoint regressions, PTIMER regressions, control failures, output errors,
or cleanup errors. Independent re-parsing checked outer containment, both
offset endpoints, the 32 ns allowance on each side, every interval width, and
both medians. The selected-endpoint bracket was 748/759/761 ns min/median/max;
the userspace outer ioctl width was 6,766/6,815/11,360 ns. The unchanged
median-bracket gate is <1,500 ns, so this Phase-0 canary passes.

The full NVIDIA module subset was restored from the admitted 575 stage after
the canary. GDM and nvidia-persistenced are active, the driver responds at
575.57.08, and the 400 W power limit was restored. The first unprivileged
power/persistence commands were retained in `lifecycle.log`; they failed and
did not affect the endpoint sample gate. A subsequent privileged power-limit
restoration succeeded and is recorded in the same log.

This canary proves endpoint precision only. It does not yet prove PTIMER and
PTX `%globaltimer` identity, validate 220 launch pairs, or produce a
`launchlate` overhead result.
