# Fig. 15 device-side evidence audit

## Scope and conclusion

This audit traces the active paper's Fig. 15(a,b) from the plotted PDF back to
the retained summaries, plotting code, bpftime benchmark, BPF programs, and
current CUDA/PTX runtime. It does not run a GPU.

The retained data are not sufficient to call either panel a reproduction.
Panel (a) has no pinned warp-aggregation implementation or official eGPU
artifact behind its two labels. Panel (b) mixes different map semantics,
iteration counts, and latency definitions. The plotting script also subtracts
the new baseline from both old and new series.

The defensible repair is split by mechanism:

- Stop the warp-aggregation/eGPU comparison until two distinct, claim-matched
  implementations are present and pinned. The current general runtime is a
  per-thread control, not the missing optimized arm.
- Replace the map panel with a prospective, operation-matched comparison of
  device-resident, directly host-mapped, and legacy host-RPC arrays. The
  adjacent `plan.md` freezes that experiment; the local harness implements it.

Fixed-work trampoline precision does not repair either panel: it times the
current per-thread return-only runtime and tests a different block-organization
hypothesis.

## Evidence path

### Retained figure inputs

- `docs/paper/img/results-raw/runtime/plot_microbench.py` reads only two
  aggregate Markdown tables and emits `microbench_comparison.pdf`.
- `docs/paper/img/results-raw/runtime/old/micro_vec_add_result.md` is one RTX
  5090 summary dated 2025-10-16. The same table entered bpftime in revision
  `495d42b` with the generic CUDA microbenchmark.
- `docs/paper/img/results-raw/runtime/micro_vec_add_result.md` is one RTX 5090
  summary dated 2025-12-03. Both Markdown files entered the paper together in
  paper revision `af02f1d`; the paper repository retains no associated JSON,
  per-iteration samples, probe log, correctness record, or invocation count.
- The newer table is not present in the current bpftime tree, and its source
  revision and build options are not recorded. The nearby 2025-12-03 bpftime
  change `2bc38dc` adds a kernel/GPU shared map; it does not add a warp-leader
  dispatch pass.

### What panel (a) actually plots

For the 32-element, one-block, 32-thread `tiny` workload, the script plots ten
rows shared by the old and new summaries: empty, entry, exit, entry+exit,
ring-buffer output, global timer, per-GPU-thread array, memtrace, GPU-array
update, and GPU-array lookup.

The source-native path is:

1. `benchmark/gpu/workload/vec_add.cu` launches `vectorAdd`, synchronizes after
   every launch, and reports a host-clock aggregate.
2. `benchmark/gpu/micro/micro_vec_add_config.json` selects a mode in
   `cuda_probe`.
3. `benchmark/gpu/micro/cuda_probe.c` enables one or two programs from
   `cuda_probe.bpf.c`.
4. The CUDA entry or return PTX pass inserts an ordinary `call`/`call.uni` to
   the compiled BPF function for every executing thread.

The current entry and return passes contain no lane election, ballot, shuffle,
or once-per-warp handler dispatch. The helper bridge's 32-lane loop serializes
host requests from already executing lanes; it is not warp-level aggregation
of one policy invocation. Thus:

- the current general runtime is a scalar per-thread implementation;
- the old table is reasonably traceable to the historical generic bpftime
  benchmark, but `eGPU-style` is not an independently pinned official eGPU
  artifact;
- the new table has no retained implementation identity and cannot be called
  the paper's described lane-local/warp-leader/broadcast prototype.

The operation labels also hide semantic differences. `empty` enables both an
entry and a return program, while `entry` and `exit` enable one. The GPU update
body performs lookup, load/add, and update. The GPU lookup body does not retain
the loaded value. Memtrace hooks a different synthetic attach point.

The nearby `benchmark/gpu/nvbit` directory does not supply the missing eGPU
arm. It is an NVBit prototype, its README states that launch callbacks are not
triggered for the Runtime-API workload, and both injected device timing bodies
are commented out. It therefore provides neither an eGPU implementation nor a
valid substitute series. This conclusion is specific to that historical
microbenchmark directory; the separately repaired Table 1 NVBit adapter tests
a different observability comparison and cannot identify Fig. 15's old/new
series.

### Baseline-subtraction error

`plot_microbench.py` obtains only the new `Baseline (tiny)` value, 5.15 us, and
subtracts it from both series. The old series' own baseline is 5.23 us. The
independent `audit_legacy_results.py` replay uses each series' own baseline.
For example, entry+exit changes from 1.02 us old overhead to 0.91 us new
overhead, only a 10.8% reduction. Therefore the text's unqualified 60--80%
range does not cover all ten plotted organizations even after repairing the
arithmetic. Correct subtraction still cannot turn two single historical
aggregates into a causal optimization comparison.

### What panel (b) actually plots

The two red bars are the absolute times of a standard `BPF_MAP_TYPE_ARRAY`
update and lookup on the three-iteration `minimal` workload. On a device-side
callback, an ordinary BPF array misses the CUDA fast paths and enters
`make_helper_call`: active lanes are serialized, a request is published through
mapped communication memory, and a host thread performs the map operation.
That path explains the roughly 33.7-ms values.

They are not measurements of the current direct host-backed GPU array
(`BPF_MAP_TYPE_GPU_ARRAY_HOST_MAP`, type 1513), whose data buffer is registered
host memory directly addressable by the GPU. The plotted GPU array (type 1503)
is allocated with `cuMemAlloc` and accessed through the device fast path.

The panel is not operation-matched:

- CPU update issues one standard-map update; GPU update issues a GPU lookup
  followed by a GPU update.
- CPU lookup dereferences and increments the returned pointer; GPU lookup has
  no externally checked result.
- CPU bars are absolute latency from three launches; GPU bars in the adjacent
  panel are baseline-subtracted overhead from 10,000 launches.
- No map is read back, so neither the helper semantics nor the number of
  callback executions is established.

The retained new summary yields descriptive absolute ratios of 4,875.8x for
the two differently defined update rows and 5,867.6x for the two differently
defined lookup rows. There is no single operation-matched 6000x estimate.

### Runner and correctness limitations

The historical `benchmark/gpu/run_cuda_bench.py`:

- runs each configuration once, in fixed order;
- retains only one mean parsed from each application;
- accepts target return code -11 as if it were successful;
- waits a fixed ten seconds instead of checking readiness;
- deletes its temporary probe log;
- records no PTX replacement, exact target, map readback, or invocation count.

The workload checks only printed `C[0]` and `C[1]`, does not compare them with
an oracle, ignores several CUDA return values, and uses host clock plus a device
synchronize for each launch. Impossible rows in the retained summaries (for
example, instrumented large ring-buffer cases faster than native) reinforce
why fail-closed raw replay is necessary.

## Evidence required for a valid replacement

Every prospective series needs:

- the exact source revision and build configuration;
- the same CUDA binary, hook site, launch geometry, warmups, timed launches,
  and device-event interval;
- randomized paired blocks from fresh processes;
- full CUDA output validation and failure on every nonzero return code;
- target-specific PTX replacement, module-load, and attach evidence;
- externally read map contents proving operation semantics;
- retained application, loader, and agent logs plus all timing observations;
- independent replay from those raw logs and uncertainty across blocks.

For panel (a), source inspection must additionally demonstrate exactly one
policy execution by a selected warp leader plus the claimed aggregation and
broadcast semantics. The comparator must be an official/pinned eGPU revision,
not the same current runtime relabeled after a date change.

## Stop and paper-decision boundary

Panel (a) is STOPPED. Do not run or cite a `warp aggregation versus eGPU`
number until both missing implementations pass the source and engagement
conditions above. If they are not recovered or implemented before the revision
deadline, remove panel (a), the 60--80% sentence, and once-per-warp performance
claims; describe the current general device path as per-thread and keep verifier
and trampoline-scaling evidence separate.

Panel (b) may proceed through the adjacent frozen plan because all three actual
map tiers exist in the selected runtime. If the direct host-mapped arm cannot
initialize or pass exact readback on RTX 5090, the run is incomplete rather
than evidence for device memory. If the prospective result does not reproduce
an order-of-magnitude gap, report the observed bounded comparison and delete
the 6000x statement.
