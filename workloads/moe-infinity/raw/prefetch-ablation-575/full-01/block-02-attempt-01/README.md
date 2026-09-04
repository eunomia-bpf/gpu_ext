# Block 02 attempt 01: retained startup-timeout failure

Date: 2026-09-03

Status: **failed attempt; excluded from performance analysis**.

The first scheduled cell, `bpf-prefetch-on`, completed all six requests and
its local gates passed. The next cell, `bpf-prefetch-off`, reached
`TOPO: 75 stages, 36 sparse` during cold model construction but never emitted
the topology-complete or HTTP-ready messages. The unchanged outer readiness
gate expired after its predeclared 900 seconds.

The runner then performed owned cleanup. The stalled server required the final
SIGKILL stage and therefore records `server_exit_code=-9` and a cleanup error;
the block is invalid by construction. Post-cleanup evidence reports no compute
process, 15 MiB GPU memory use, UVM reference count zero, no struct-ops map or
link, no new RM warning, no Xid, and no kernel/journal anomaly.

Read-only inspection localizes the wait to model/topology initialization but
does not identify whether the exact cause was a topology mutex, host allocation,
or CUDA runtime/driver wait. Five neighboring cold starts using the same
binary, store, and runtime completed in about one minute, and the prefetch
toggle is configured after model construction, so this single event is not
evidence of a deterministic prefetch-off or BPF-policy failure.

The complete block 01 was independently re-audited and remains the only valid
block at this checkpoint. The successful first cell in this failed attempt is
not reused or pooled. An unchanged `--resume` must retain this directory, create
`block-02-attempt-02`, and rerun all four arms so that block pairing remains
intact. If the same initialization stop repeats, it becomes a reproducible
blocker rather than grounds for additional unchanged retries.
