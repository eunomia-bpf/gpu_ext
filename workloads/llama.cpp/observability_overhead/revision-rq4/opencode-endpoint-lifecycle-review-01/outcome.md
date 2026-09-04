# OpenCode endpoint lifecycle review outcome

OpenCode session `ses_f92c138f9ffelZHG0KwYQI9chQ` received the deny-all
request and all four attached implementation files using
`spark-gateway/qwen3.8-27b-nvfp4-200k`. The initial review and one short-verdict
continuation produced no model text. OpenCode logged `AI_APICallError` for both
streams and retried them; the two client processes were stopped after more than
five minutes so they would not remain as background work. Therefore there is
**no Qwen verdict**, and this record must not be represented as a model PASS.

The bounded local review performed while waiting added or confirmed the
following concrete gates:

- a requested preflight/full child must pass independent raw analysis before
  wrapper completion, and runs before rollback while endpoint-v1 is loaded;
- the child argv is fixed, uses no shell, receives exactly the two already-held
  read-only lease descriptors, and cannot accept an arbitrary callback;
- child argv, return code, stdout, stderr, timeout/interrupt state, and analyzer
  result are retained; SIGINT/SIGTERM terminates the owned child process group
  before rollback;
- partial label mutation is marked before it begins, post-removal failures
  always attempt exact `849ea75d` restoration, and failed recovery additionally
  removes scheduling labels and stops admitted GPU services;
- `complete=true` is published only after the fixed probe, requested child,
  exact module/service/label/node/boot/400 W/safety restoration, and both lease
  closes succeed.

This is an honest failed model-review attempt plus local CPU-only validation,
not a substitute for the pending live module lifecycle and GPU campaign.
