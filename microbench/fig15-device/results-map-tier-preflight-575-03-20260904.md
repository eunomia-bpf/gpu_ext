# Device-map preflight attempt 03: post-cleanup-fix engagement

Date: 2026-09-04  
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08  
Result directory: `raw/map-tier-preflight-575-03`

All 8/8 frozen arms passed correctness, exact map readback, target
transformation/load/attach evidence, detach, child-process cleanup, and private
shared-memory reclamation after the caller-owned cleanup-state repair.

The independent analyzer reports `run_status=valid_preflight` and
`tested_hypothesis=not_tested`. Its two-launch descriptive medians are retained
only as execution evidence. They are not combined with any full campaign and
are not a paper performance result.
