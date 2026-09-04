# Device-map preflight attempt 02: all paths engaged

Date: 2026-09-04  
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08  
Result directory: `raw/map-tier-preflight-575-02`

All 8/8 frozen arm processes passed application correctness, exact map
readback, target transformation/load/attach evidence, detach, process cleanup,
and private-shared-memory reclamation. In particular, repaired `rpc_lookup`
and `rpc_update` produced all 32 distinct expected per-lane values.

The independent analyzer reports `run_status=valid_preflight` and
`tested_hypothesis=not_tested`. Its descriptive medians were 9.184/9.360
us per launch for device lookup/update, 10.608/27.120 us for direct host-mapped
lookup/update, and 33,706.558/33,704.754 us for serialized RPC lookup/update.
These two-launch preflight values establish execution only and are **not a
paper performance result**.

