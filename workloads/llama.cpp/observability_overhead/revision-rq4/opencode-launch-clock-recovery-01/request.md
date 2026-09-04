# OpenCode launch-clock recovery review request

- Date: 2026-09-03 (America/Vancouver)
- OpenCode: 1.18.27
- Model: `opencode/ling-3.0-flash-fin-free`
- Session: `ses_f9561deceffesNAiWUqRJj02S8`
- Mode: `--pure`, read-only prompt review
- Configuration:

```json
{
  "snapshot": false,
  "share": "disabled",
  "permission": {"*": "deny"},
  "tools": {
    "write": false,
    "edit": false,
    "bash": false,
    "webfetch": false,
    "task": false
  }
}
```

The reviewer received the frozen classification and drift semantics, the two
preflight-575-07 launchlate summaries, the proposed one-second NVBit minimum
anchor span, the CUPTI 12.9 documentation findings, and the NVIDIA RM timer
correlation findings. It was asked to distinguish actual drift from a
conservative bound, reject result-dependent threshold changes, and identify
the remaining gpubpf blocker.

Two follow-ups corrected reviewer mistakes rather than changing the requested
verdict:

1. `%globaltimer` endpoint-change direction is not proof that a calibration
   bracket missed the device timestamp; the raw logs do not retain enough
   per-sample final intervals to split gpubpf uncertainty causes.
2. 5,092 ns over one second is 5,092 ppb, not 5 ppb; the CUPTI diagnostic has
   not been executed and cannot establish a documented clock-domain contract.

