# Table 1 runtime strict-device validation on RTX 5090

Two fresh positive/negative pairs passed on 2026-09-04 using driver 575.57.08,
Linux 6.15.11, and the verifier-enabled Table 1 runtime at bpftime revision
`b266cf2`. This validates the ported strict path before testing the actual
Table 1 observability objects; it is not a performance result.

| Pair | Positive | Negative | Safety and cleanup |
| --- | --- | --- | --- |
| [01](raw/575-table1-port-strict-01) | STRICT admitted 13 instructions and the real type-1502 map; 4,096 threads each executed eight callbacks, totaling 32,768 | Lane-varying branch rejected at instruction 1 before policy-entry insertion/bootstrap; later counter snapshot remained entirely zero | Native/instrumented numerical checks passed; private segment removed; no owned survivor, Xid, abnormal kernel record, UVM reference, or struct_ops residue |
| [02](raw/575-table1-port-strict-02) | Same complete result in a fresh process/private segment | Same explicit rejection and all-zero post-rejection snapshot | Same complete result |

The runtime cache records `ENABLE_EBPF_VERIFIER=ON`, CUDA attachment ON, and
LLVM JIT ON. Its full 459-step build passed. Direct CPU gates passed: the GPU
verifier 23 cases/137 assertions, strict-reject cleanup 1/4, strict-counter
fixture 1/6, late-attach source invariants 7/7, and the ring-buffer source/PTX
protocol. A separate verifier-OFF build passed its four-mode configuration
matrix (2 cases/22 assertions): explicit STRICT fails closed, WARNING reports
that verification is unavailable and continues, and NO_VERIFY/default retain
their intended behavior.

The negative records `policy_entry_created=0`, not “no generic hook.” Frida and
CUDA interception are installed before policy verification; strict rejection
prevents the policy entry and late bootstrap, and the zero counter proves the
rejected callback did not execute. A failed detach during rejection cleanup now
preserves remaining attach tracking instead of resetting it. This is scoped to
the strict-reject path; older epoch/session teardown paths are not claimed
fixed.

OpenCode/Qwen reviewed the port and narrow cleanup in deny-all mode. The final
runtime review passed in session `ses_f93b7add6ffeYVutF7pzqK2Ag6`; the
verifier-OFF review session was `ses_f93ba37adffenstjfE4x9JSE7a`.

The next gate is A0: STRICT admission and full pp=32 correctness/engagement for
the actual compact `kernelretsnoop` and `threadhist` objects. Existing Table 1
performance remains verifier-OFF evidence until a separately paired
STRICT/NO_VERIFY campaign completes.
