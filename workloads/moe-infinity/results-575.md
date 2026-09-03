# Retained generic-policy diagnostic, not a MoE-Infinity algorithm reproduction

Stopped on user instruction after **one complete paired block**. The BPF cell
uses host-stride prefetch plus sampled LFU, not MoE-Infinity's activation-aware
algorithm. The requested original-algorithm reproduction remains unfinished;
do not continue the superseded five-block campaign.

The four configurations are a deployment H2H, not an equal-output pure-policy
comparison or a reproduction of the authors' original-hardware paper numbers.
BPF/UVM is the clean same-engine comparison: all eight output goldens match.
MoE/UVM matches **0/8**, and N-CMoE32/UVM matches **5/8**.

| Configuration | Output tokens/s | Median first-nonempty-text TTFT (s) | Median E2E (s) |
|---|---:|---:|---:|
| MoE-Infinity, cache fraction 0.75 | 11.6373 | 1.979 | 5.478 |
| gpubpf host-stride + sampled LFU | 1.62860 | 23.428 | 34.057 |
| llama UVM | 5.93908 | 5.951 | 11.074 |
| llama N-CMoE32 | 8.95986 | 4.276 | 7.121 |

These are single-block descriptive values, not five-block estimates or
confidence intervals. Each cell requested eight fixed 512-input/64-output
prompts after one excluded warm-up. Throughput includes prefill and response
delivery, but excludes server startup/warm-up. TTFT counts the first nonempty
text SSE, including whitespace; it is not necessarily the first model token.
The first block's 4 cell results, 32 raw streams, 2,048 output tokens, policy/
engine counters, telemetry, and cleanup records passed the offline raw audit.

Exact-output matches against UVM, by one-based frozen prompt number:

| Configuration | Matching prompts | Different prompts |
|---|---|---|
| gpubpf | 1, 2, 3, 4, 5, 6, 7, 8 | none |
| N-CMoE32 | 1, 2, 3, 4, 6 | 5, 7, 8 |
| MoE-Infinity | none | 1, 2, 3, 4, 5, 6, 7, 8 |

All configurations independently passed their exact repeated-output checks.
Raw text is retained in the preflight evidence and is not duplicated here.
The BPF command is frozen at `-t 2 -n 2 -m 128`; it replaces the default
prefetch decision with a host-stride-selected two-page window and sampled
frequency-driven reordering. Its result is the joint effect of that concrete
policy and mechanism, not a measurement of generic BPF execution overhead.
No completed-eviction count is claimed from eviction-prepare callbacks.

## User-directed stop and safety handoff

During attempt 2's UVM cell, the owner sent SIGINT to runner PID 174219. The
runner stopped its own server; that server exited with SIGABRT during thread
join/teardown. Compute processes disappeared, memory returned to 2 MiB,
UVM refcount became zero, and struct_ops was empty. Nevertheless GPU
utilization remained 100% for over 60 seconds, so the post-cleanup gate failed.
No new Xid was observed. This state was **not** declared clean or safe for a
new CUDA run. The temporary lease holder released both locks to the root
coordinator, which recovered the idle device by ordinary module unload/load,
without force or reboot. At 2026-09-03 00:36:42 UTC the new `e3bb2938` driver
stage reported 2 MiB, 0% utilization, 10.49 W, UVM refcount zero, and no compute
clients. That recovery is not retroactively a passed cleanup for attempt 2.

The coordinator's wider journal inspection also found `NVRM: Going over RM
unhandled interrupt threshold` (IRQ 217) at 00:20:46 UTC, during the old BPF
cell. The original cell checks did not classify that warning. Consequently
the old campaign must not be described as having no kernel anomalies; these
observations alone do not establish the warning's cause.

Evidence: [stop record](raw/head-to-head-575-lossless/timing/user-directed-stop.json),
[completed block](raw/head-to-head-575-lossless/timing/attempt-01/block.json),
[interrupted attempt](raw/head-to-head-575-lossless/timing/attempt-02/block.json),
and [repair/failure history](continuation-575.md).

The CUDA-12.9 compiler selection and lossless SSE transport repair are useful
correctness fixes, not implementation of the missing activation-aware policy.
