# Independent review: POD preflight 575-04

2026-09-03. **PASS: no observed blocker to starting the frozen full sweep.**
This is admission evidence, not a performance result or full-shape reproduction.
Evidence: [`raw/preflight-575-04/`](raw/preflight-575-04/).

The reviewer read the complete plan and numerical-protocol-v2 rules, then ran
`run_study.validate_preflight` against a fresh inventory from the current
`preparation()` paths. All **19 runtime files** match the saved inventory;
manifest, completed order, five arms and protocol
`pod-fp16-upstream-match-v2` pass. No frozen file was modified.

Independent saved-record checks, on CPU17:

- All five arms completed the Llama-3-8B attention shape, decode batch 32,
  with ten warmups and three retained timed observations. Both complete outputs
  match the official FP16 FA reference: `max_abs_vs_official=0` in every arm,
  with the unchanged hard `atol=1e-3, rtol=1e-5` gate.
- Three POD diagnostic records each contain **3,328 actual CTA contexts**.
  An independent calculation checked per-SM tickets, global atomic claims,
  exhaustion fallback and exactly-once coverage of **2,048 prefill + 1,280 decode
  slots**. Inline/CUDA use engine 1; BPF uses engine 2 throughout, with 338 real
  exhaustion fallbacks. Nondeterministic SM assignments are not required to match.
- Both adapter arms record **14 bridge launches** (one diagnostic, ten warmups,
  three timed), 81,920 requested/verified dynamic-shared-memory bytes, and one
  prepared function. BPF decisions execute inside the real device operator;
  this is not host-JIT selection. The comparison includes the launch adapter.
- Raw GPU telemetry recomputes to the saved summaries for all five arms;
  pre/post safety checks pass on driver 575.57.08. Injected targets enter through
  `taskset → env → Python`; the wrapper environment has no `LD_PRELOAD`.
  The BPF loader records exactly one `READY kernels=6` and one orderly `CLOSED`;
  private-segment removal is recorded and that exact segment is absent.

V2 deliberately separates FP16-arm agreement from cross-precision
characterization. Every arm retains complete FP32 scans: prefill checks
33,554,432 elements with one reported excess (maximum 0.0013279914855957031);
decode checks 131,072 elements with no excess. Every prefill excess has its
complete diagnostic metadata and saved Q/K/V/actual/FP32 arrays with matching
recorded sizes. **This does not claim full-FP32 agreement.** Earlier v1 failures
and the interrupted preflight-03 remain unchanged and excluded.

The review reran no inference and reconstructed no unsaved output tensors.
It reused the existing safety/report validators and additionally checked saved
CTA arithmetic, launch environments, diagnostic files and cleanup evidence.
No GPU work, build, threshold adjustment or active-full data inspection was
performed. The five-block, ten-shape, five-arm full sweep remains the required
performance evidence; these three-sample preflight timings must not enter it.
