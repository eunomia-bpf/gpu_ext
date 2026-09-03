# OpenCode consultation and implementation reconciliation

Session: `ses_f98a48122ffe6XCxV9HoThZwo5`, configured default model (no
model override), `snapshot:false`. Initial permissions allowed only
read/glob/grep/list and denied all other tools. The attached complete task is
`opencode-prompt.md`; the final response is preserved verbatim in
`opencode-final.md`. Advisory statements there are not measurements.

The first invocation misparsed its `--file` argument before model work;
`opencode-events.jsonl` and `opencode-stderr.log` preserve that failure.
The corrected invocation started around 13:00 UTC on September 3, 2026,
terminal 86337. Root explicitly requested implementation proceed in parallel
with the consultation; the core was implemented/tested before its final
review. Its read-only event stream `opencode-events-02.jsonl` records actual
reads of the new patch, ring header/tests and CPU/build logs, as well as the
original sources. Thus this was an independent proposal/review, not a claim
that OpenCode authored the applied patch.

After repeated extra reads, root requested a bounded final response. The
owned CLI was interrupted (terminal 86337, exit **130**); the same session was
resumed with `snapshot:false`, **all tools denied**, asking for at most 600
words using already-read evidence. Terminal 8969 exited **0**. Its complete
`opencode-events-03.jsonl` contains one nonblank final text and zero tool calls.
Both invocation stderr logs are retained. No GPU or build was delegated.

## Accepted and corrected advice

- Accept the capacity/done invariant, successful-query-only event retirement,
  final drain, unchanged same-stream dependencies and HP admission lock. Real
  depth-2 LC protection still needs measurement. The exception during old
  executor shutdown remains a pre-existing limitation, not silently fixed.
- Its requested `kernel_unstarted=0`/consolidation case is already represented
  in exhaustive C/JIT inputs at both bounds; the original policy also has an
  explicit native/JIT no-consolidation assertion. This is input/parity coverage,
  not proof of a GPU race test. No additional concurrency guarantee is claimed.
- Do **not** adopt its suggestion to rebind every `HERE` path and copy the
  cubin/old clients: that would conflate immutable trace/model locations and
  the actually executed private binaries. The private source patch preserves
  original trace/cubin paths and separately inventories the private client,
  BPF bytecode and executed source. No cubin copy or nvcc rebuild is needed.
- Its phrase about overlap “iff bound 2” is too strong: bound 2 permits but
  does not guarantee overlapping host issues. The runner records actual peak
  and overlap. Unexercised depth-2 cells are retained; a preflight that does not
  exercise depth 2 cannot admit the causal full comparison.
- Preserve its key scope corrections: the paper says **1.3% slowdown**, not
  microseconds; host outstanding records are not device-queue occupancy, and
  the old 19–20% loss is not already attributable to this fence.

The private runner's nine synthetic CPU tests pass. They include both bounds
and C/JIT evidence, event/drain/CTA corruption, exact matrix and config,
all-offered request arithmetic, preflight runtime/profile/exposure admission,
and formal completeness. The first saved runner test attempt exposed a patch
serialization error (two trailing context lines were missing); the corrected
patch passes application checking and tests. Both `runner-tests-01.log` and
`runner-tests-02.log` remain, with no GPU result implied.
