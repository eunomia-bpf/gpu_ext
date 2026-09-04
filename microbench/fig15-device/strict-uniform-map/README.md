# STRICT uniform-map placement

This experiment is the strict-admitted complement to the sibling per-lane
map-tier run. Every GPU lane uses constant key 0 and the same value, so the
current SIMT verifier can admit the shared-map effect. It compares type 1503
device-global and type 1513 host-mapped arrays for update and lookup, with
native and strict no-op controls.

It is intentionally a different workload from the per-lane Full6 campaign.
Never pool their samples or attribute their ratio difference to the verifier.

Build and run from this directory:

```sh
make
python3 -m unittest -v test_strict_uniform_map.py
python3 run_strict_uniform_map.py --phase preflight \
  --output raw/strict-uniform-map-preflight-575-02
python3 analyze_strict_uniform_map.py \
  raw/strict-uniform-map-preflight-575-02
python3 run_strict_uniform_map.py --phase full \
  --output raw/strict-uniform-map-full-575-01
python3 analyze_strict_uniform_map.py \
  raw/strict-uniform-map-full-575-01
```

The live runner requires an RTX 5090 with driver 575.57.08, CUDA 12.9, the
verifier/CUDA/LLVM-enabled `build-table1-575-strict` runtime, and the existing
read-only GPU and struct-ops lease files. It refuses ambient injection and
never changes or deletes either lease.

The full result is
[`results-full-575-01-20260904.md`](results-full-575-01-20260904.md). The first
preflight is retained as invalid because its execution-record header was
malformed; no result is taken from it.
