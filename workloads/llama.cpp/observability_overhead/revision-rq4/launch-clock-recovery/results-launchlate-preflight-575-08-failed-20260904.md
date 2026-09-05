# Launch-latency preflight attempt 08: retained raw-path failure

Date: 2026-09-04
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08
Lifecycle directory: `raw/rm-correlation-575-08-endpoint-lifecycle`
Campaign directory: `raw/rm-correlation-575-08-endpoint-lifecycle/launchlate-preflight`

## Outcome

This attempt is **invalid and contributes no correctness or performance
result**. Both calibration controls passed, and all three correctness arms
passed. The single timing block also recorded successful process exits,
complete safety records, and valid launch engagement. Both launchlate arms
passed their three-anchor clock reconstruction, at-least-one-second held-out
validation, launch accounting, classification, and uncertainty gates.
The gpubpf arm accounted for and classified 220/220 correctness launches and
44/44 timing launches with zero uncertain samples; its held-out spans were
1,001,096,746 ns and 1,001,022,724 ns. NVBit likewise accounted for 220/220
and 44/44 selected launches with zero uncertain samples; its spans were
1,000,914,477 ns and 1,000,985,018 ns.

The independent analyzer nevertheless rejected the gpubpf timing cell. Its
stored benchmark and probe paths were under `launchlate_run_101/`, while the
runner wrote that cell's safety and telemetry under
`gpubpf_launchlate_run_101/`. A raw cell is valid only when its benchmark,
execution, probe, safety, and telemetry evidence share one config-specific
directory. The split makes raw closure unprovable even though the individual
records passed their semantic gates.

## Retention and retry boundary

The attempt-08 raw directory and `result.json` remain unchanged. No field is
reclassified or copied into a later attempt. The runner now uses
`gpubpf_<tool>_run_<id>/` for every artifact in each gpubpf timing cell; the
baseline and NVBit directory rules are unchanged. An offline independent-
analyzer regression recreates the attempt-08 split and requires rejection.

Attempt 09 may run only from a new lifecycle and campaign directory named in
`launchlate-attempt09-frozen-path.md`, after the existing module-lifecycle,
two-control, correctness, engagement, clock, safety, and cleanup gates pass.
