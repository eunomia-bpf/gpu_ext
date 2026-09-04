# Device-trampoline preflight attempt 03

Status: **native arm accepted; counter arm rejected; no performance result**.

The per-arm telemetry lifecycle worked: the native arm passed correctness,
telemetry validation, and the post-arm zero-UVM safety gate, then was safely
checkpointed. Its one measured cell checked all 1,048,576 outputs with zero
mismatches.

The next scheduled counter arm also completed application correctness and
cleanly detached both links, but its full map readback failed the independent
engagement oracle. The target map had the expected preflight shape: slots
0--65,535 each counted three warmup-plus-measured callbacks, with all remaining
slots zero. The nominal marker map unexpectedly had that same target-shaped
readback instead of 32 slots counted once. The runner therefore rejected an
ambiguous/misrouted attachment rather than treating its timing as trampoline
evidence.

The final safety check recorded UVM reference count zero, an idle 15 MiB GPU,
no compute process, no kernel anomaly, and no attached struct_ops. This attempt
does not establish attached overhead and must not be pooled with a later valid
preflight.
