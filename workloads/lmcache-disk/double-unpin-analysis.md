# Double-unpin warning in the LMCache V3 CPU arm

Scope: established evidence from retained raw cells only. The
double-release explanation below is explicitly a source-based diagnosis
(inference from LMCache v0.5.4 source), not established runtime causality;
no root-cause attribution is recorded here.

## Observed warning

LMCache v0.5.4 logs, from `lmcache.v1.memory_management`
(`memory_management.py:819`):

    Pin count of MemoryObj <offset>is negative: -1.Double unpin occurred
    somewhere.Setting pin count back to 0 as a hack but please find the bug.

It is a `LMCache WARNING`; the affected cell continues and completes its
requests.

## Counts in retained raw cells

Line matches of the warning text in each arm's retained `server.log`:

| Cell                                   | Arm            | Double-unpin warnings |
| -------------------------------------- | -------------- | --------------------- |
| `raw/storage-575-v3-correctness-02`    | `lmcache_cpu`  | 48                    |
| `raw/storage-575-v3-correctness-02`    | `lmcache_disk` | 0                     |
| `raw/storage-575-v3-correctness-02`    | `recompute`    | 0                     |
| `raw/storage-575-v3-diagnostic-01`     | `lmcache_cpu`  | 6                     |
| `raw/storage-575-v3-diagnostic-01`     | `native_prefix`| 0                     |

## Established occurrence pattern

In both CPU cells the warnings occur only on warm cache-hit requests,
immediately after the request-scoped `Retrieved 1536 out of 1536 required
tokens` line:

- correctness-02: 8 warm requests x 6 chunks of 256 tokens = 48 warnings.
  Each cluster carries six consecutive chunk offsets and matches the
  request's retrieved size of 0.1406 GB.
- diagnostic-01: 1 warm request x 6 chunks = 6 warnings.

The observed call sequence is: the synchronous lookup pins the retrieved
chunks' MemoryObj, `retrieve` releases the pin, and the connector adapter's
`lookup_unpin` then performs a second release, driving the pin count to -1
and triggering the warning. LMCache resets the count to 0 in place. That
double-release reading is a source-based diagnosis of the sequence; it is
inference, not established runtime causality.

All eight warm requests in correctness-02 still reported
`Retrieved 1536 out of 1536 required tokens`, so the warning coexists with
complete retrieval.

## Relation to the warm output mismatch

This warning is separate from the cached-prefill output mismatch:

- In correctness-02, `lmcache_cpu` and `lmcache_disk` produce identical
  text for all eight cold and all eight warm requests. The divergent warm
  outputs are only in the `recompute` arm (five of eight prefixes: 0, 1, 2,
  5, 6).
- The disk arm shows the same warm outputs as the CPU arm while emitting
  zero double-unpin warnings.
- In diagnostic-01 the six warnings co-occurred with a warm request whose
  generated token (ID 2303, text `"  \n"`) and top-20 logprobs matched the
  native-prefix arm exactly.

No conclusion is recorded here about the root cause of either the warm
output divergence or the pin-count bookkeeping defect; both remain open.
