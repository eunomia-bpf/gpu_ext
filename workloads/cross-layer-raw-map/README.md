# Cross-layer raw-record map experiment

This harness addresses Reviewer A's question about whether gpubpf's cross-layer
state path is limited to composable reductions.  It uses two existing bpftime
GPU map ABIs in one device-side BPF return handler:

- a per-GPU-thread ring buffer carries every raw
  `(sequence, block xyz, thread xyz)` tuple to a host reader; and
- a per-GPU-thread array independently maintains callback, sequence, block, and
  thread-coordinate aggregates.

The finite CUDA target writes the same tuple from CUDA built-ins into a device
truth array.  The host validates every native and instrumented CUDA tuple,
every retained BPF tuple, every aggregate shard, and the ring-buffer accounting.
Thus the positive cells test non-composable raw readback rather than inferring
it from a counter.

Machine-readable target/probe stdout and runtime diagnostics are retained in
separate `*.log` and `*.stderr.log` files. This prevents asynchronous agent
shutdown messages from corrupting a JSON record while preserving both streams.
The runner records a private shared-memory segment identity only while the
exact inode is demonstrably open or mapped by its live probe. Cleanup still
refuses to unlink a missing, unknown, or replaced segment. BPF-object open
failures report the object path plus both libbpf and `errno` diagnostics.

## Frozen protocol

Each randomized block contains three fresh-process cells:

| Cell | Geometry | Launches | Required disposition |
| --- | ---: | ---: | --- |
| `small` | 256 threads (2 x 128) | 3 | all 768 tuples and zero drops |
| `large` | 2,048 threads (16 x 128) | 3 | all 6,144 tuples and zero drops |
| `overflow_negative` | 256 threads (2 x 128) | 6 | retain four tuples/thread, report exactly 512 full drops, reject the stream as incomplete evidence |

The ring capacity is deliberately fixed at four records per GPU thread.  The
probe does not drain until the finite target exits.  This makes the negative
case deterministic and prevents concurrent polling from hiding overflow.  A
drop is never treated as a successful raw-stream observation: the negative
cell passes only when the validator labels it
`rejected_incomplete_raw_stream` and accounts for every omitted tuple.

Preflight is one complete randomized block.  The formal campaign is five
complete randomized blocks (15 cells) and will not start without a passed,
protocol-compatible preflight.  Each cell runs a native CUDA truth process, a
new private bpftime syscall-server/probe process, and a separate instrumented
CUDA truth process.  Exact process groups and the exact owned `/dev/shm`
segment are cleaned; the runner retains the shared GPU/struct-ops leases and
the repository's driver, UVM, service, telemetry, and kernel-log safety gates.
The campaign performs no within-run retries. Infrastructure failures retain a
failed manifest and require a new output directory, so completed cells cannot
be silently selected or replaced.

## Build and inspect the plan

```bash
make
make test-offline
python3 run_raw_map.py dry-run --plan-mode preflight \
  --output raw/preflight-575-01
python3 run_raw_map.py dry-run --plan-mode full \
  --output raw/full-575-01 --preflight raw/preflight-575-01
```

`dry-run` does not inspect build artifacts, acquire leases, create its output,
launch a process, or touch the GPU.  A real run uses new directories:

```bash
python3 run_raw_map.py preflight --output raw/preflight-575-01 \
  --runtime-build ../../../bpftime-table1-575/build-table1-575
python3 run_raw_map.py full --output raw/full-575-01 \
  --preflight raw/preflight-575-01 \
  --runtime-build ../../../bpftime-table1-575/build-table1-575
```

The checked 575 Table-1 runtime currently has device verification disabled.
The manifest records that configuration, so these cells cannot be cited as
strict-verifier evidence.  Existing strict positive/negative device tests
cover that separate question.

## Claim boundary

A successful formal run shows that the current GPU-to-host ABI can carry and
exactly recover bounded raw records that cannot be reconstructed from the
aggregate control alone.  It does not measure latency or bandwidth, exercise
an on-chip/shared-memory shard, prove transparent automatic map placement, or
show that arbitrary unbounded data structures fit the ABI.  Ring capacity is
finite; its explicit failure mode is detected loss, not lossless streaming.
