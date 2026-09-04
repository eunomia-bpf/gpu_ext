# Launch-latency preflight attempt 01: retained endpoint failure

Date: 2026-09-04  
GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08  
Result directory: `raw/launchlate-575-preflight`

## Outcome

This attempt is **invalid and contributes no correctness or performance
result**. Admission passed, but the runner stopped during its calibration-only
clock-control gate, before launching any baseline, gpubpf, or NVBit workload
cell.

All 200 direct `endpoints-v1` control requests failed with
`control_error=-121` and `rm_status=86`; zero samples were accepted. The
dependent GPU-globaltimer identity test therefore did not run. The failure is
consistent with the running 575.57.08 core module predating the endpoint
command implementation used by this frozen harness. It is not evidence about
gpubpf or NVBit launch-latency overhead.

## Recovery boundary

The raw attempt is retained and will not be overwritten. A new attempt may run
only inside the dedicated module lifecycle wrapper: it loads the explicit
endpoint-capable candidate modules, validates the endpoint, runs the requested
experiment, and unconditionally restores the exact known-good module set and
system services. The retry must use a new result directory.

## Independent review

OpenCode with the local `spark-gateway/qwen3.8-27b-nvfp4-200k` model reviewed
the report and attached raw records in deny-all mode. Session
`ses_f92cabe3affe0raVfaXwz9Rrp6` returned `VERDICT: PASS`. It confirmed that
all performance fields are empty and that the report makes no performance
claim. It also requested retention of the per-request JSONL carrying the 200
error-code records; that file is included in the retained raw directory.
