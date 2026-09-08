# Transport 2 vs 3 matched-pair runner

`run_transport_pairs.py` reproduces the `transport23-pairs` campaign as a
reusable CLI. It runs the original per-thread `kernelretsnoop` probe on
bpftime `886b4ca` with `BPFTIME_GPU_AUTO_WARP_EXECUTION=1` on both arms and
contrasts the two ringbuf transport layouts over five rotating same-block
matched pairs:

| label        | `BPFTIME_GPU_RINGBUF_TRANSPORT` | layout                        |
|--------------|---------------------------------|-------------------------------|
| `transport2` | `2`                             | encoded per-record tail (AoS) |
| `transport3` | `3`                             | transposed record words (SoA) |

Both arms keep the original per-thread object (80-byte payloads, 256 entries
per thread, 524288 allocated thread slots); only the header/payload memory
layout and host reassembly differ. No no-probe baseline and no NVBit replay
are produced; the same-block paired change is the metric. The prebuilt
original tool is reused, so no source preparation, compilation, or tool build
happens here.

## Graceful loader shutdown

By default the runner sets `wait_for_probe_exit`, so the private loader
receives SIGINT once after its CUDA client exits and is waited on without a
kill deadline. That lets the loader's final ring drain and post-processing
finish before the private SHM segment is removed. Process ownership and
private SHM cleanup remain in the existing helper. Disable with
`--no-wait-for-probe-exit`.

## Usage

```
python3 run_transport_pairs.py --dry-run
flock /tmp/gpubpf-revision-gpu0.lock flock /tmp/gpubpf-revision-struct-ops.lock python3 run_transport_pairs.py --output-dir raw/transport23-pairs-NEW
```

`--blocks N` sets the number of rotating pairs (default 5). The default
`--prebuilt-tool`, `--bpftime-root` and `--bpftime-build-dir` reproduce the
recorded `execute.py` invocation.

Existing measurement directories are never overwritten. Failed cells remain
recorded and result in a nonzero runner exit; owned-cleanup failures stop the
invocation rather than allowing another timing cell to overlap. Five pairs
rotate order 3/2, not exactly equal early/late counts. The completed batch is
published in `a9fb0a13`; this CLI's addition does not rerun it.

## Outputs

- `cells.json` / `summary.json`: one row per cell, appended after each cell
  and its collector finish. Numeric `throughput_tok_s` and the loader/collector
  status are retained separately. Loader exit and shared-memory cleanup are
  in each cell's `probe-execution.json`, not nested in parsed probe counters.
- `pair-XX/transportM/auto_warp_on_run_YY/`: helper-relative logs
  (`llama_bench.log`, `probe-execution.json`, `probe.log`, `agent.log`).

The internal helper label is `auto_warp_on` in BOTH arms; the outer
`transport2` / `transport3` labels identify the compared layouts. The
campaign is run under shared GPU locks; do not replay old raw results.
