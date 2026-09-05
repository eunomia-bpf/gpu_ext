# Independent result review

## Scope and method

This review inspected `plan.md`, the target `warp_map_bench.cu`, the BPF source
`warp_map_probe.bpf.c`, the loader `warp_map_loader.c`, the runner, the
analyzer, and every raw file in `raw/strict-warp-scaling-full-575-01` (524
files) and `raw/strict-warp-scaling-preflight-575-06`. Arm counts, return
codes, STRICT admission records, map-effect rows, CUDA correctness records, and
detach markers were read directly from the raw logs, and the timing medians,
paired ratios, sign counts, and cross-shape factors were recomputed from the
`FIG15_MEASUREMENT` records. The analyzer was not imported for that
recomputation; its retained `analysis.md` and `analysis.tsv` were compared only
afterwards. The 18 offline tests pass. Both campaigns were additionally replayed
in an isolated copy outside `raw/`, where the analyzer reproduced the retained
outputs byte for byte; the interval endpoints come from that seeded percentile
bootstrap, whose seeds are now a deterministic function of frozen seed 1797 and
the fixed shape/comparison indices rather than of `PYTHONHASHSEED`.

## Failure bookkeeping

- Attempt 05 is invalid and contributes no sample. Its raw loader log shows the
  syscall server reporting `Unable to initialize CUDA from syscall server
  side: 3` (`NV_ERR_NO_MEMORY`) and aborting through `std::runtime_error`
  before any program load, verifier call, transformation, attach, launch, or
  map readback. Only `environment.txt`, `schedule.tsv`, and that one
  `loader.log` exist; there is no execution record or timing record.
- Attempt 04 stopped at the same error. Attempt 01 stopped after its first cell
  and attempts 02 and 03 wrote only their schedule. All four remain retained and
  unsampled.
- Preflight 06 is valid but execution-only: 20/20 processes, 15/15 target-PID
  `mode=STRICT` acceptances, and no effect row is emitted, which the test suite
  enforces.

## Completion and mechanism engagement

The full campaign is complete and internally balanced:

- Exactly 160 scheduled cells exist (five shapes x eight blocks x four arms),
  matching the frozen seed-1797 order, with no missing or extra directory. The
  40 `native` cells carry only the uninstrumented application record; the 120
  attached cells carry loader, agent, application, and execution records.
- All 160 processes return code zero — **zero nonzero codes** in the campaign —
  and all 160 emit an exact CUDA output record with zero mismatch over 8
  warmups and 128 measured launches.
- All 120 attached cells bind exactly one target-PID `mode=STRICT` acceptance
  and one verifier timing record to the recorded PID, with instruction counts
  2 / 14 / 23 for no-op / shared-key / warp-key programs, and exactly one
  verified map descriptor per cell at type 1503, key size 4, value size 8, and
  64 entries. No reject, skip, or verifier-unavailable record occurs.
- Each attached cell shows one `matched=1` transformation, one patched PTX
  module load, one successful attach, one loader ready marker, and one detach
  marker (120 detach markers). The 120 target PIDs and 120 private
  `fig15_warp_*` names are unique, and no process or shared-memory segment
  survived.
- The independent map oracle accepts every cell: 40 no-op readbacks with
  **count 0**, 40 shared-key readbacks with **count 1** at key 0, and 40
  warp-key readbacks whose keys are all in range, whose values all equal
  `magic XOR key` against a single constant magic (512 key/value pairs, one
  magic), and whose final distinct key counts are 4, 4, 8, 16, and 32 at shapes
  32, 128, 256, 512, and 1024 — at least the requested simultaneous warp
  coverage in every cell.

## Independent recomputation

Medians from CUDA-event milliseconds divided by 128 launches, within-block
paired log-ratios, and log-space change from 32 to 1024 threads:

| Shape | native | noop | shared_update | warp_update |
|---|---:|---:|---:|---:|
| 32 | 1.842500 | 3.899625 | 3.880500 | 3.845250 |
| 128 | 1.785875 | 3.924375 | 3.847125 | 3.873875 |
| 256 | 1.842500 | 3.884500 | 3.876500 | 3.880000 |
| 512 | 1.941250 | 3.871500 | 3.866875 | 3.861625 |
| 1024 | 1.844500 | 3.891500 | 3.873500 | 3.857375 |

| Comparison | shape/role | ratio | positive signs |
|---|---|---:|---:|
| shared / noop | 32 / 128 / 256 / 512 / 1024 | 0.9964 / 0.9848 / 0.9945 / 1.0033 / 0.9912 | 3, 2, 4, 4, 3 of 8 |
| warp / noop | 32 / 128 / 256 / 512 / 1024 | 0.9904 / 0.9919 / 0.9925 / 0.9949 / 0.9917 | 2, 3, 4, 4, 2 of 8 |
| warp / shared | 32 / 128 / 256 / 512 / 1024 | 0.9936 / 1.0058 / 0.9988 / 0.9975 / 0.9962 | 1, 8, 4, 3, 2 of 8 |
| warp / shared | 32 -> 1024 change | 1.0062 | — |
| shared / noop | 32 -> 1024 change | 1.0102 | — |
| warp / noop | 32 -> 1024 change | 1.0060 | — |

Every point estimate and sign count reproduces the retained analysis output
without using its implementation; only the bootstrap interval endpoints are
taken from it, after confirming seed stability.

## Interpretation boundary

The run validly demonstrates that the current STRICT verifier, CUDA JIT,
attach path, and device-resident map accept and execute a per-warp-keyed update
at one, four, eight, sixteen, and thirty-two warps per block, and that the
accepted program really reaches the GPU and writes the planned keys. It finds
**no** cost-growth advantage for warp-uniform keys: all three cross-shape
factors contain no change, and the only per-shape interval excluding one in the
predicted direction is the `warp_update/shared_update` contrast at 32 threads
(0.9936 [0.9848, 0.9971]), which is the shape with a single active warp and
therefore no inter-warp contention for the hypothesis to relieve. The frozen
directional hypothesis is contradicted.

Scope that must travel with any use of this result:

- **Single block per launch.** Launches are `<<<1, threads>>>`; nothing here
  addresses multi-block or grid behavior.
- **Scalar per-thread callback.** One `call.uni __bpftime_cuda__kernel_trace,
  ();` per thread, with all 32 lanes of a warp still issuing the update. This
  is not warp aggregation, not once-per-warp dispatch, not warp-leader
  execution, and not evidence of constant per-warp trampoline cost.
- **Helper 510 reads the physical PTX `%warpid`.** The final distinct key counts
  4 / 4 / 8 / 16 / 32 by shape are SM-local hardware warp slots observed across
  136 launches of a process, not logical CTA warp IDs; the oracle demands at
  least the requested simultaneous coverage and intentionally not the exact
  slot set.
- The idempotent final readback proves a map effect, not invocation
  cardinality. The timed interval holds the whole trampoline, callback, and
  update path, so this is an end-to-end callback comparison rather than a
  contention decomposition. STRICT verification precedes the timed interval and
  is an admission gate, not a metric.
- No pooling across shapes, no independent-run test, and no pooling with the
  verifier-disabled per-lane placement campaign or the strict-uniform
  constant-key campaign.

## Judgments

- Run status: **valid**.
- Tested hypothesis: **contradicted** — no detected warp-key scaling advantage.
- Research value: **negative result / mechanism boundary**.
- Paper impact: **none from this campaign.** Keep it as repository evidence and
  as an explicit answer to the map-granularity and thread-count-scaling
  questions; do not relabel it as a paper-positive result, and do not use it to
  reinstate the retired warp-aggregation wording. The accompanying
  no-op/no-contention findings remain the only supported statements.
- Next paper decision: state that under STRICT per-lane callback semantics,
  changing the device-map key granularity from one shared key to one physical
  `%warpid` key per warp shows no detected cross-shape scaling advantage from 1
  to 32 warps per block, and that reducing callback work requires per-warp
  execution, which the current callback semantics do not provide.

No blocking defect was found in strict-admission evidence, callback engagement,
map-effect correctness, schedule balance, cleanup, timing attribution, or the
paired inference.
