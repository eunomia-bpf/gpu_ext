# Fig. 15 device-side evidence repair

This directory separates two device-side claims that the current paper figure
combines:

- `legacy-evidence-audit.md` traces the retained figure inputs and freezes the
  STOP/delete boundary for the unavailable warp-aggregation/eGPU comparison.
- `warp-source-gap.md` exhaustively classifies the bounded local source and Git
  history search for that missing implementation and its near misses.
- `warp-source-gap-review.md` records the independent deny-all Qwen review.
- `plan.md` defines one prospective RTX 5090 experiment for the runnable map
  placement question: device-resident versus direct host-mapped arrays, with
  the old serialized host-RPC path as a diagnostic.

The historical arithmetic can be replayed without a GPU:

```sh
python3 audit_legacy_results.py
python3 -m unittest -v test_audit_legacy_results.py
```

The map-tier harness is intentionally the current scalar per-thread runtime.
It must not be described as the missing warp-aggregation prototype, and the
fixed-work trampoline experiment must not be substituted for either Fig. 15
claim.
