# Device-map preflight attempt 04: primed loader engagement

Date: 2026-09-04  
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08  
Result directory: `raw/map-tier-preflight-575-04`

All 8/8 frozen arms emitted the new syscall-server prime record and passed BPF
object parsing, correctness, exact map readback, target transformation/load/
attach evidence, detach, child cleanup, and private-segment reclamation.

The independent analyzer reports `run_status=valid_preflight` and
`tested_hypothesis=not_tested`. The two-launch timings are retained only to
prove execution; they are not a paper performance result and are not pooled
with any full campaign.
