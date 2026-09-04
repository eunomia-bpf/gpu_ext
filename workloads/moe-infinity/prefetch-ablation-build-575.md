# Predictive-prefetch ablation build readiness (RTX 5090 / driver 575)

Date: 2026-09-03

This record covers dependency build and ABI readiness only. It is not a GPU
experiment result and does not make a performance claim.

## Build

From `workloads/moe-infinity`:

```bash
taskset -c 8-15 .venv/bin/python -B build_paper_store.py
```

The command exited successfully after compiling four objects and linking the
editable-install extension in place at:

```text
deps/MoE-Infinity/moe_infinity/_store.cpython-312-x86_64-linux-gnu.so
```

Compiler warnings were non-fatal; the build completed with exit status 0.

## ABI gate

The harness's CPU-only runtime-inventory check passed against the rebuilt
extension. In particular, it found the five-argument predictive-prefetch
`_store` ABI required by `run_prefetch_ablation.py`. Dynamic-library inspection
reported no missing dependency.

## Next gate

Run the real four-arm preflight only after confirming that the active driver
stage is the protected 575 stage used by the accepted MoE campaign:

```text
/opt/gpubpf/modules/575.57.08/gpreempt-849ea75d-6.15.11
```

The preflight must pass correctness, exact-output, engagement, and resource
accounting gates before the full five-block campaign is eligible to run.
