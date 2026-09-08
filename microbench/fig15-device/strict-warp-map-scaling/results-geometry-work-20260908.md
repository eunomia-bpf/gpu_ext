# RTX 5090 automatic-execution geometry/work measurements

Completed on 2026-09-08: 180 measurements, ten rotating native/off/on paired
blocks for each of six settings. All 180 applications and 120 attached loaders
exit zero. This is a CUDA-event microbenchmark, **not Table 1 prefill throughput**.
The unchanged `shared_update` BPF object is used in both attached arms.

Runtime: bpftime `e5e52e5`, `revision/automatic-warp-execution`, CUDA 12.9,
RTX 5090, driver 575.57.08. Extended benchmark source: main `d4c5e773`.
Every cell uses 128 threads/CTA, eight warmup launches and 128 timed launches.
Times below cover **all 128 timed launches**, in milliseconds.

| CTAs / arithmetic iterations | Native | Attached, automatic off | Attached, automatic on | Paired on/off change, median [95% CI] |
|---|---:|---:|---:|---:|
| 2 / 0 | 0.251984 | 0.332512 | 0.524096 | +57.516% [56.133%, 58.315%] |
| 4 / 0 | 0.230192 | 0.331568 | 0.524080 | +58.061% [57.226%, 58.750%] |
| 8 / 0 | 0.203184 | 0.333680 | 0.524288 | +57.099% [55.801%, 58.467%] |
| 1 / 8 | 0.213600 | 0.334752 | 0.387504 | +15.409% [14.553%, 16.788%] |
| 1 / 32 | 0.250800 | 0.523776 | 0.524256 | +0.180% [-0.146%, 0.432%] |
| 1 / 128 | 0.524528 | 0.530144 | 0.528976 | -0.151% [-0.504%, 0.584%] |

Native/off/on columns are cell-time medians. The effect column is the median
of ten within-block percentage changes, not a ratio of column medians.
Intervals bootstrap those ten paired changes 10,000 times, seed 1797, using
sorted bootstrap medians at indices 249 and 9749. All ten pairs are slower
with automatic execution for each CTA setting and for work=8; seven of ten
are slower at work=32, four of ten at work=128.

## Interpretation and separate invocation-count observation

Automatic execution is substantially slower in the low-work cases. Its
relative time difference becomes small as arithmetic increases; work=32 and
128 intervals include zero. These data do not establish a performance gain.
Nearly flat elapsed time across 2–8 CTAs does **not** establish that total
overhead is independent of block count: GPU parallelism and occupancy can
hide additional work. [Actual scalar callback counts](results-observed-counts-20260908.md)
are now measured in twelve separate diagnostic processes. They show 32x
fewer scalar calls with automatic execution, but calls still grow with CTA
count. `logical_lane_encounters` in the timing CSV is calculated from launch
geometry and is not that observation. The timing cells were not rerun.

## Raw records, units and failed prefix

All records are retained under
`raw/geometry-work-20260908.glLbTj/`. Completed campaigns are
`blocks-current-abi/` and `work-current-abi/`, with their runner logs.
Their original `cells.csv` files accidentally label microseconds as
`elapsed_ms`: the existing parser returns microseconds. Original files are
preserved unchanged. `cells-derived-ms.csv` reads milliseconds directly from
each original `FIG15_MEASUREMENT` application line. The runner serialization
is separately corrected for future runs; no measurements were repeated to
fix this reporting unit.

Reproduce the derived CSVs and table (from this directory):

```sh
python3 analyze_geometry_work_20260908.py raw/geometry-work-20260908.glLbTj
```

The earlier `blocks/` prefix contains a failed attempt, not part of the 180
completed cells. The embedded PTX template still allocated 40-byte map-info
entries although the source ABI had grown to 48 bytes. Its first attached
cell took 35,229.1016 ms and failed map readback. Regenerating the existing
trampoline template and rebuilding the runtime fixed this stale build
artifact (`e5e52e5`, pushed). The first template build could not find C++
headers; the successful invocation selected the installed GCC 13 headers.
Both build attempts, runtime build and failed prefix are retained. Historical
automatic-execution, Table 1 and P40 results are untouched.
