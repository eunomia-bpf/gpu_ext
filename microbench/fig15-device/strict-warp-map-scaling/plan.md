# STRICT warp-uniform map-sharding experiment plan

Date: 2026-09-04

## Paper question and hypothesis

Reviewer D asks whether device trampolines scale with large thread and block
counts, while Reviewer A asks whether the GPU map organization supports useful
non-aggregate state.  This experiment isolates one mechanism question:

> Under STRICT admission and the current scalar-per-thread callback semantics,
> does assigning each resident warp a uniform device-map key reduce the growth
> in callback cost relative to making every warp overwrite one shared key as
> active warps per block increase?

The directional hypothesis is that a warp-uniform key reduces inter-warp
write contention at larger block sizes.  It does not remove the 32 identical
per-lane overwrites within each warp, and it does not test a warp-leader
execution optimization.

## Frozen system and workload

- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0.
- Driver: 575.57.08; CUDA toolkit: 12.9.
- Runtime: the existing verifier-enabled bpftime/gpubpf build with
  `BPFTIME_VERIFIER_LEVEL=STRICT` and targeted late bootstrap.
- Target: a purpose-built CUDA kernel with exactly one explicit
  `call.uni __bpftime_cuda__kernel_trace` per thread and one block per launch.
- Shapes: 32, 128, 256, 512, and 1024 threads, corresponding to 1, 4, 8,
  16, and 32 active warps.
- Every process uses a private shared-memory namespace and takes the existing
  GPU and structural-operation lock files read-only with non-blocking exclusive
  locks.

## Arms and intervention

The four arms are run as fresh processes in a deterministic randomized order
inside every shape/block pair:

1. `native`: target kernel without injection (timing floor and correctness
   control).
2. `noop`: STRICT-admitted empty callback (trampoline/JIT control).
3. `shared_update`: STRICT-admitted device-array update using constant key 0
   and a constant value (strong mechanism baseline).
4. `warp_update`: otherwise identical update using helper 510 (`%warpid`) as
   the key and a value derived only from that warp-uniform key (intervention).

Both map arms use one 64-entry type-1503 device-resident shared array.  The
same target kernel, hook point, callback count, value width, update helper,
system visibility fence, build, and launch schedule are held fixed.  Only the
key/value computation and resulting address differ.

## Repetitions and schedule

- Preflight: one complete four-arm block at each shape, one warmup and four
  measured launches.
- Full: eight complete four-arm blocks at each shape, eight warmups and 128
  measured launches.
- Frozen seed: 1797.  Arm order is independently shuffled for every
  shape/block pair before any GPU result is observed.
- Primary unit of replication: one fresh process.  The full campaign therefore
  contains 160 process runs and 40 complete four-arm comparison blocks.
- One-hour campaign deadline; any partial campaign remains labeled partial and
  is not promoted to a result.

## Admission, engagement, and correctness gates

Every attached process must prove all of the following before its timing is
admitted:

- the target PID has exactly one `mode=STRICT` acceptance record and one
  verifier timing record for the selected program;
- all map descriptors report type 1503, key size 4, value size 8, and 64
  entries; no reject, skip, or unavailable marker occurs;
- exactly one target stub is transformed, the patched PTX module loads, the
  selected program attaches, and the private namespace is removed;
- the CUDA output oracle covers every launched thread with zero mismatch;
- `shared_update` reads back only key 0 with the fixed magic value;
- `warp_update` reads back at least the requested number of nonzero warp keys,
  every nonzero key is in range, and every value equals `magic XOR key`;
- `noop` has no nonzero map entry.

The warp-key oracle intentionally allows more than the requested number of
keys across repeated launches because `%warpid` is a hardware SM-local warp
identifier and different launches may occupy different warp slots.  Requiring
at least the simultaneous warp count is the invariant supplied by a
single-block launch.

## Metrics and analysis

The primary metric is CUDA-event microseconds per launch.  At each shape, use
complete same-block pairs to report medians and paired ratios:

- `shared_update / noop` and `warp_update / noop` (incremental scaling);
- `warp_update / shared_update` (primary intervention effect).

Report percentile-bootstrap 95% intervals over the eight paired blocks using
a fixed analysis seed, plus the sign count for `warp_update < shared_update`.
The cross-shape secondary result is the change in the two incremental ratios
from 1 to 32 warps.  No pooling across shapes and no independent-run test is
allowed.

## Claim boundary and stopping rules

- A lower warp-sharded time supports only reduced contention for this
  device-array overwrite callback; it is not evidence that the generic
  trampoline runs once per warp.
- A tie or regression is retained and reported.  No arm, shape, or complete
  block may be removed after timing is observed.
- Because every lane still executes the callback, this test cannot establish
  a constant per-warp trampoline overhead.  It complements, rather than
  replaces, the existing fixed-work no-op scaling result.
- Preflight can stop on any admission, engagement, cleanup, CUDA correctness,
  or map-effect failure.  Full results are published only after all 160 runs
  and an independent raw-record audit pass.

## Artifact layout

- Harness and tests: this directory.
- Immutable run directories: `raw/<campaign-name>/`.
- Human-readable result: `results-<campaign-name>-20260904.md`.
- Independent audit: `independent-review.md`.

