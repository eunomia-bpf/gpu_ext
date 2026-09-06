# uvm-policy-mechanism workload

`uvm_fault_stream` is a short real-UVM workload: it allocates managed memory,
CPU-initializes one deterministic word per region, prints one readiness line,
optionally waits for a UVM monitor on stdin, then runs sparse GPU touch passes
that generate real GPU page faults. It doubles as a reusable controlled
pressure tenant.

Build (RTX 5090 / CUDA 12.9):

    nvcc -O3 -std=c++17 -Xlinker --build-id=none \
        uvm_fault_stream.cu -o uvm_fault_stream

## CLI

    ./uvm_fault_stream [--gib N] [--region-kib N] [--passes N] \
        [--pause-ms N] [--wait-for-monitor] --output PATH

- `--gib N` (default 8): managed allocation size.
- `--region-kib N` (default 64): sparse touch stride; one word per region.
- `--passes N` (default 1): run the same sparse GPU touch pass N times after
  the optional stdin release. Each pass is synchronized and timed with CUDA
  events; the allocation stays live for every pass.
- `--pause-ms N` (default 0): sleep in milliseconds between consecutive
  passes, not around the first or last pass.
- `--wait-for-monitor`: keep the `MONITOR_PID`/stdin protocol from the
  original experiment.

## Readiness line

After allocation and CPU initialization, but before the optional stdin wait,
the process prints exactly one line:

    READY pid=<pid> gib=<gib> regions=<regions> passes=<passes> pause_ms=<pause>

Consumers can start an external pressure/policy tenant after seeing this line
and before sending the stdin release.

## Output JSON

    {
      "bytes": ..., "region_bytes": ..., "regions": ...,
      "passes": N, "pause_ms": M,
      "kernel_ms": <final pass>,
      "kernel_ms_per_pass": [...],
      "kernel_ms_total": ..., "kernel_ms_median": ...,
      "kernel_ms_max": ..., "completed_passes": N,
      "mismatches": 0, "first_mismatch": null
    }

Mismatches are checked on the final pass. With `--passes 1` the output keeps
the legacy fields, and `kernel_ms` is that single pass.

## Small smoke

    ./uvm_fault_stream --gib 1 --region-kib 64 --passes 3 --pause-ms 200 \
        --output /tmp/uvm_pressure_smoke.json
