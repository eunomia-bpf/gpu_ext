# CPU safety rejection matrix plan

Date: 2026-09-04

## Question and admission

This supporting experiment asks whether the current executable artifacts expose
a reviewer-readable matrix that separates three enforcement layers: the base
program verifier, the GPU SIMT pass, and driver-owned transition validation.
It directly addresses Reviewers B/F's request for concrete rejected policies;
it does not measure performance or establish full-stack deployment.

The previous Phase A already covers base memory bounds, termination, a
lane-varying branch, map key/value uniformity, a non-uniform atomic target, and
the prohibited `bpf_gpu_membar` helper. Re-running those cases is only a
regression gate. The new evidence is four previously unreported SIMT side-effect
pairs: direct stores into shared-map values, a varying helper writing through a
shared-map pointer, lane-varying map-update flags, and a lane-varying host-bridge
payload. Five representative transition pairs make the operation-specific
fallback/preserve result explicit while calling the production-shared driver
header rather than a duplicated validator.

## Frozen matrix and interpretation

Every unsafe program has a matched accepted control differing only in the
relevant provenance, helper, request, or snapshot field.

| Layer | Unsafe case | Matched control | Required result |
| --- | --- | --- | --- |
| Base verifier | stack write at `fp-520`; helper-dependent backward loop | same write at `fp-8`; constant-bounded loop | unsafe rejected; control accepted |
| SIMT pass | lane-derived branch/map side effect/atomic address | warp-ID-derived or shared-map control | unsafe rejected; control accepted |
| SIMT pass, new | lane-derived direct shared store, helper output, update flags, or host-bridge payload | same instruction shape with warp-uniform data/helper | unsafe rejected with the named SIMT diagnostic; control accepted |
| Transition validation | below-minimum timeslice, stale phase, conflicting request, invalid action 99, wrong PMM owner | minimum legal value, current phase, idempotent repeat, BYPASS, matching owner | unsafe request preserves/native-routes as specified; control applies |

The hypothesis is supported only if all selected pairs and both existing
regression suites pass. A rejection without its positive control, or a
positive control rejected by an unrelated base-verifier error, invalidates that
pair. Results remain CPU-only: no policy is attached and no GPU or driver state
is changed.

## Command and boundaries

Run from the `gpu_ext` root:

```text
BPFTIME_ROOT=../bpftime-r5 \
  extension/revision-safety-rejection-matrix/run_cpu_matrix.sh \
  docs/experiment/revision-safety/rejection-matrix-cpu-575-01/execution
```

The runner links the public `verify_gpu_program` entry point from the existing
verifier build and includes the production `nv-gpu-transition-validator.h`.
Raw stdout and the effective-capability line are retained under
`execution/raw/`; build products use a fresh temporary directory and are not
experiment evidence.

Linux host-policy kfunc/BTF admission is deliberately excluded from this
CPU-only slice. It requires a privileged `BPF_PROG_LOAD` against the loaded
driver's BTF; compiling an object would not execute Linux's verifier. The
current process has no effective capabilities. Existing privileged load-only
evidence remains separate and is not relabeled as a result of this run.
