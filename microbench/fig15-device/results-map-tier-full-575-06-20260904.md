# RTX 5090 device-map placement result

Date: 2026-09-04

GPU / driver: NVIDIA GeForce RTX 5090 / 575.57.08

Result directory: `raw/map-tier-full-575-06`

The preregistered full campaign is **valid** and its tested hypothesis is
**supported**. All 128 fresh arm processes completed in 16 balanced randomized
blocks without a failed-cell retry. The project analyzer and a separate
raw-only reconstruction agree on every reported statistic.

## Main result

| Operation | Device-resident median | Direct host-mapped median | Paired host/device ratio (97.5% CI) | Paired delta (97.5% CI) |
|---|---:|---:|---:|---:|
| Update | 3.846 us/launch | 36.185 us/launch | 9.4307x [9.3789, 9.4896] | +32.347 us [31.909, 32.834] |
| Lookup | 3.823 us/launch | 4.175 us/launch | 1.0904x [1.0797, 1.1113] | +0.346 us [0.310, 0.419] |

Both Bonferroni-adjusted co-primary intervals lie wholly above 1.0. Under the
frozen decision rule, keeping this 32-entry map on the GPU therefore reduces
the measured steady-state callback path for both operations. The effect is
strongly operation-dependent: direct host mapping is about 9.43x slower for
update but only 1.09x slower for lookup.

The serialized standard-array RPC arms take 33.843 ms/update and 33.831
ms/lookup, or about 8,844x and 8,991x the corresponding device-resident arms.
Those values diagnose this prototype's request/response protocol; they are not
a generic PCIe or map-placement cost. The no-op and native medians are 3.918
and 1.986 us/launch. Device update and lookup are respectively 0.044 us (95%
interval [-0.108, 0.011]) and 0.081 us ([-0.156, -0.004]) below no-op in the
paired descriptive medians; these tiny negative increments are not
interpretable as negative operation cost or a causal benefit from adding a map
operation.

## Validity evidence

- The frozen seed-1797 cyclic-plus-reverse schedule contains every arm once in
  each block and places every arm twice in each of the eight order positions.
- All 128 applications report RTX 5090, SM 120, warp size 32, eight warmups,
  64 timed launches, and 32/32 correct application outputs.
- All 112 attached cells contain one successful target transformation,
  patched-module load, attach, mode-specific ready record, and detach record.
  The transformed call is part of the statically loaded kernel body used by
  the application's exact launch loops; map readback separately validates the
  resulting lane/key values.
- All 3,072 expected map entries match exactly. The readback is intentionally
  idempotent and is not presented as an independent per-launch callback
  counter.
- Every loader and application process recorded by the campaign has exited;
  every private shared-memory segment identity is absent after cleanup.
- The loader prime occurs before BPF object parsing and before application
  launch, warmup, and CUDA-event timing. It removes the diagnosed one-time
  initialization interleaving and is not inside either primary timing path.

The runner did not persist a separate per-cell exit-status/cleanup manifest;
completion of the fail-fast 128-cell sequence and the independent post-run
process/segment audit supply that evidence for this campaign. A future runner
revision can make it durable without changing this result.

## Scope

This is an operation-matched placement result for one 32-thread block on the
current scalar-per-thread, GPU-verification-disabled runtime. It is not a pure
hardware map-access latency, end-to-end application result, verifier-overhead
measurement, grid-scaling result, or evidence of warp-leader aggregation.
It replaces the old undifferentiated “6000x CPU map” story with separate
direct-host-mapped and serialized-RPC comparisons.

Five earlier full campaigns were retained as invalid infrastructure attempts.
They contribute no performance sample. Attempt 06 began again at block 1 after
the loader's lazy-startup interleaving was fixed and does not reuse any earlier
prefix.

An isolated OpenCode request to the specified local
`spark-gateway/qwen3.8-27b-nvfp4-200k` model reached the model but timed out
after 300 seconds without a response. It supplies no review verdict and was
not treated as a pass; the independent raw-only review above was performed
separately.
