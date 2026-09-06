# results-uvm-kv-pressure-canary-575-20260906
Method: one minutes-apart canary, not repeated H2H.
| run | TTFT (ms) | req/s | output tok/s | success/failure |
| --- | --- | --- | --- | --- |
| native -05 | 116.2761 | 1.7030 | 27.2475 | 8/0 |
| scoped gpubpf -06 | 113.6902 | 1.6944 | 27.1111 | 8/0 |
Delta BPF/native: output -0.5006%, req/s -0.505%, TTFT -2.223% (lower).
Scope: old unscoped -04 tracked the pressure tenant; the scoped policy saw 48 KV ranges / 1056 MiB, initial 528 chunks, engine PID only, 528/531 warm, later 426 tracked, 483 releases.
Framing: front half of the storage-tier async state machine - LMCache durable disk backing/restore plus gpubpf GPU/UVM residency. Full GPU-resident -> durable/evictable -> restore-pending -> GPU-resident still needs the 575 discard/restore actuator and disk-backed repeated comparison.
