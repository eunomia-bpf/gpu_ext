# Callback-delivery handoff (20260909)

Source-only candidate delivered:
`workloads/xsched/level2/tool/xsched_guard_tool_callback.cu`.
No build, install, GPU action, or commit was performed. Root owns build and
run. The original `xsched_guard_tool.cu` and the isolated HAL source belong to
the GLM session and were not edited here.

## Derivation base

The candidate is a copy of the committed `xsched_guard_tool.cu` file version
last updated in `c6153b3b`
(the pre-repair version, byte content of 2026-09-08 21:33), with the single
placement change below.

Concurrent observation, for root's coordination: at 2026-09-09 07:07 a GLM
session edited `xsched_guard_tool.cu` in the working tree (uncommitted, shows
as ` M` against `c6153b3b`) and applied the same callback-side delivery using
`deliver_val`/`deliver`. The two sources now diverge (see
"Remaining concerns"); root decides which file is the build input.

## Evidence and what it does and does not show

XG5 actual runtime (failure repair, not performance results): 201 host
preparations/callbacks, zero disarms. The host published distinct slots, but
BE1's retained device marks kept slot 1/kernel_idx 1 through entry ordinal
200 (then permanently resume type 2); BE4 kept slot 3; three BE workers had
missing/repeated work; LC finished.

What this shows: later launches consumed a stale early per-launch value at
runtime. It does not by itself prove which API ordering caused that. The
official NVBit example (`mem_trace.cu` `enter_kernel_launch`) instruments
first and calls `nvbit_set_at_launch` inside the callback afterwards. The
XG5 evidence is consistent with the hypothesis that the host-side
pre-instrumentation set is what later launches stop consuming; that this
placement is the cause, and that the reorder is the fix, remain hypothesis
until the post-repair run.

## Exact differences vs the derivation base

1. `xg_host_prepare` behavior is unchanged: stores the TLS `armed_val`
   marker, bumps `xg3_set_cnt` with the existing log line, and still issues
   the outer `nvbit_set_at_launch(ctx, f, ctx_dev)` as a retained fallback.
   Only a comment was added there.
2. `nvbit_at_cuda_event` (pre-launch, target-filtered path only):
   - Captures `uint64_t armed = armed_val;`. If 0: bumps
     `xg3_disarm_cnt` with the existing log line; its former in-place zero
     set is deferred to the trailing call. If nonzero: consumes the one-shot TLS marker
     (`armed_val = 0`), as before.
   - After the instrumentation block (`guarded_launches++` and
     `insert_entry_call` on first target launch), adds one trailing
     `nvbit_set_at_launch(ctx, f, armed);`. This delivers the saved
     nonzero value for managed launches after instrumentation, and applies
     the one-shot disarm (`armed == 0`) after instrumentation as well.
   - Net effect vs base: the armed-value set moves from host-side-only to
     callback-side-after-instrumentation (outer set retained as fallback);
     the disarm set moves from before to after instrumentation.
3. The other executable logic is unchanged: target filtering, `xg3` counters
   (set/cb/disarm), `xg_host_prepare` ABI and symbol, native/BPF
   `decision_mode`, `insert_entry_call` argument list
   (`guard_pred`, `launch_val64` offset 0, const `decision_mode`),
   `nvbit_at_init`/`nvbit_at_term`, includes, state variables.
   No new state protocol, no new diagnostics, no policy or SASS changes.

## Remaining concerns

- Causality unproven. The reorder matches the official sample ordering and
  is consistent with the XG5 stale-value evidence, but the run is required to
  show whether later launches now consume the distinct host slots. The
  single post-repair diagnostic (probe unchanged) is for root; no completed
  baseline is repeated.
- Outer fallback not proven. For launches on threads where the NVBit
  pre-callback is not delivered (the original design notes name the
  scheduler thread that performs the Reactivate replay), only the outer
  host-side set applies, and that is exactly the pre-instrumentation
  ordering the XG5 run shows failing to update. Whether the outer set
  delivers correctly for such launches is unknown, and the candidate adds
  no diagnostics for that case.
- Divergence from the GLM working-tree edit. Same core ordering (set after
  instrumentation inside the callback, outer prepare retained), but the GLM
  version keeps the disarm before instrumentation and guards delivery with a
  `deliver` flag, whereas the candidate applies the disarm after
  instrumentation through the single trailing call. Only one of the two
  sources should be the build input.
- Build note for root: the `level2-build/Makefile` `TOOL_O` rule hardcodes
  `$(SRC)/tool/xsched_guard_tool.cu`. Building the candidate means the same
  nvcc object flags pointed at the candidate file, then the existing
  `TOOL_SO` link step with the carrier object. No Makefile edit was made
  here (not an owned file).
