# Prediction-protected executor: rebuilt three-mode correctness evidence

All three enhanced canaries completed successfully on **2026-09-03**, with the
last arm exiting at **02:15:33 UTC**. These are correctness and engagement
observations, **not comparative performance samples**. The BPF arm uses the
same-snapshot native shadow oracle in this preflight; timed cells disable it.

The shared executor now preserves the current positive prediction set against
speculative eviction, permits demand to bypass that protection, and rejects
unissued stale-epoch work. Its three selectors are unchanged. Source and gates
were committed in `96da503`, `35c9c9f`, and `7ee9da8`. The complete replayable
patch is tracked at the workload root; each arm's `admission.json` retains its
exact admitted runtime inventory.

## Build and finite tests

- `/usr/bin/g++-13 -std=c++17 -O2 -pthread` compiled the actual production-helper
  test, which passed all five scenario groups: protection and demand progress,
  stale claim, replacement between selection and eviction, stale-before-copy,
  already-issued completion, queue replacement/drain and full-width identities.
  See [build-helper.log](build-helper.log). Its generated executable remains
  local and is not needed in Git.
- `taskset -c 8-15 .venv/bin/python -B build_paper_store.py` rebuilt four affected
  C++ objects, linked and installed the new store, and exited zero. The build ran
  from 02:10:15 to 02:11:07 UTC. The installed
  `_store.cpython-312-x86_64-linux-gnu.so` is **72,853,336 bytes**. See
  [build-store.log](build-store.log).
- Each arm independently ran the real BF16 expert numerical checks for rows
  1, 256, 257 and 353, plus four accumulation arrival orders at 353 rows.
  Every maximum absolute and relative error was zero.

The coordinator's declared driver stage was
`/opt/gpubpf/modules/575.57.08/gpreempt-849ea75d-6.15.11`, with Linux 6.15.11,
NVIDIA 575.57.08 and RTX 5090. GPU and struct_ops leases serialized the runs.

## Three requests per arm, including raw SSE

Each fresh server processed prompt 0 and prompt 1 as nonstream requests, then
prompt 1 again through SSE. Every request used 512 input and 64 output tokens;
all outputs matched the frozen same-frontend goldens exactly. Each SSE contains
64 token frames followed by DONE, with `length` termination. The engine and
independent exported metric both counted exactly 64 output tokens during SSE.

| Retained observation | `native-off` | `paper-native` | `paper-bpf` |
| --- | ---: | ---: | ---: |
| Final server exit code | 0 | 0 | 0 |
| Protected-resident eligibility skips | 0 | 3,009,364 | 3,098,326 |
| Copy starts / completed prefetches | 0 / 0 | 3,990 / 3,990 | 4,052 / 4,052 |
| Prefetch first-use hits | 0 | 2,406 | 2,442 |
| Unused-prefetch evictions | 0 | 1,578 | 1,602 |
| Unused prefetched residents at drain | 0 | 6 | 8 |
| Stale unissued tasks rejected | 0 | 0 | 2 |
| Protected candidates after drain | 0 | 0 | 0 |
| BPF rank / match / eviction calls | 0 / 0 / 0 | 0 / 0 / 0 | 6,912 / 4,608 / 12,623 |
| Selector shadow mismatches | 0 | 0 | 0 |

All native-off activation state and counters remained zero. Both paper modes
ended at prediction epoch 13,830. An eligibility skip counts one protected
resident considered during a victim snapshot, not one prevented transfer; do
not reinterpret millions of skips as millions of completed prefetches.

The small native/BPF transfer-count differences reflect asynchronous execution;
selector parity is checked against the same snapshot, not inferred from aggregate
counts. The completed/hit/wasted/still-resident counters conserve exactly in each
arm. These observations neither establish a throughput advantage nor compare the
new three-request workload directly against an old eight-request timing window.

## Independent retained-artifact audit

After all server and telemetry writers closed, a separate CPU-only audit reread
each raw response, SSE file, result, telemetry file and final JIT log. It checked
the frozen output texts, 65-frame SSE lifecycle, independent 64-token accounting,
all new activation requirements, recomputed GPU telemetry, final JIT counters,
zero JIT errors and clean retained safety snapshots. All three arms passed.
Every SSE is 11,019 bytes with no NUL content.

Owned raw files and their directories were explicitly synchronized after this
audit. In particular, a producer `passed` flag was not taken as a substitute for
the actual retained observations. The earlier interrupted campaign and corrupted
native responses were not repaired, overwritten or reused.

The final cleanup left no compute processes, 2 MiB GPU allocation, 0% GPU
utilization, UVM reference count zero, and no struct_ops maps or links. No new
RM unhandled-interrupt warning was observed. The coordinator subsequently
started the independent five-block, three-mode timed campaign in
`../timing-849ea75d-02-postboot`; its completion must be established separately.
