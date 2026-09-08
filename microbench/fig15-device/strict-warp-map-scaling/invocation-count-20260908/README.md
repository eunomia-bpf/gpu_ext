# Opt-in actual probe-call invocation counter (strict-warp-map-scaling)

Diagnostic-only counter for the *actual* number of probe handler invocations
executed by the automatic-warp `shared_update` benchmark. Separate
diagnostic runs; ordinary performance timings remain uninstrumented.

## Artifacts

| file | scope | status |
|---|---|---|
| `invocation-count.patch` | 6-file counter patch (ptxpass + runtime plumbing) | integrated in bpftime-auto-warp `7cfba18` |
| `invocation-count-readback.patch` | incremental readback fix (single hunk in `nv_attach_impl.cpp`) | applies on `7cfba18`, shipped in `e61bdb3` |

Apply order on a clean `7cfba18^` tree: `invocation-count.patch`, then
`invocation-count-readback.patch` (`git apply` or `patch -p1` from the
bpftime-auto-warp repo root).

## Enabling

`BPFTIME_GPU_WARP_HOOK_CALL_COUNT` set to any non-empty value other than
`"0"` (same convention as `BPFTIME_GPU_AUTO_WARP_EXECUTION`). When unset:
no counter PTX, no diagnostic stream synchronization or readback. The host
still checks the opt-in flag; byte-identical host execution is not claimed.

## Semantics

- Counter global: `.global .u64 bpftime_warp_hook_call_count;` emitted into
  the patched application PTX (ptxpass layer). No BPF atomics, no change to
  the BPF object, to `default_trampoline.cu`, or to native/warp-policy
  eligibility.
- auto-warp ON: one predicated `red.global.add.u64` under the elected-leader
  predicate per call site → counts per-warp leader invocations.
- auto-warp OFF (scalar): one predicated add per call site under the
  original call-site predicate (unconditional at kernel-entry sites) →
  counts per executing thread.

## Readback

`ptxpass::warp_hook_call_count_env_enabled()` gates a readback inside
`nv_attach_impl::record_patched_launch_event`: synchronize the launched
stream, `cuMemcpyDtoH` the module counter, log the cumulative value. The
module-load log (`warp hook call count global present`) only confirms
presence, not a value.

Log markers in the application/agent log:

- `warp hook call count global present` — at module load (env on).
- `warp hook call count: N (cumulative actual probe-call invocations ...)`
  — after each patched launch event; duplicate cumulative observations
  occur (runtime + driver bookkeeping). **Use the final value, never sum.**
- `Unable to read back warp hook call count` — warn on DtoH failure.
- Native (uninstrumented) runs: no agent log; count is not applicable,
  not a zero.

## Observed diagnostics (12 cells, final cumulative values)

| CTAs | off (per thread) | on (per leader) | ratio |
|---|---|---|---|
| 1 | 256 | 8 | 32 |
| 2 | 512 | 16 | 32 |
| 4 | 1024 | 32 | 32 |
| 8 | 2048 | 64 | 32 |

Ratio 32 = warp width; off = threads × launches, on = warps × launches,
consistent with 2 launches and 128 threads/CTA. Raw logs are under
`../raw/observed-counts-live-20260908.4UfFEw/`; the earlier non-live directory
retains the failed destructor-only readback attempt. See the
[completed report](../results-observed-counts-20260908.md) for all six settings.

## Scope notes

- The counter covers launches since module load; these diagnostics use zero
  warmup and two launches per process.
- The 180 performance timings and disk-5 runs completed/pushed separately;
  this directory holds count observation only.
