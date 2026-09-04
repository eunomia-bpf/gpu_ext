# Stale-state host-uBPF consumer preparation

Date: 2026-09-04

## Outcome and scope

The bounded stale-state decision model now executes through a real host-uBPF
JIT and matches the native model over malformed inputs and a long deterministic
phase stream. This is **dependency evidence only**. It is not a real preflight,
GPU/UVM engagement, a stale-state performance result, or evidence that native
and BPF mechanisms have equal cost.

The preparation removes one implementation unknown from the frozen live plan:
the BPF consumer exists, compiles to BPF bytecode, loads and compiles through
uBPF, fails closed at its bounded context ABI, and has no native fallback. Live
execution remains blocked on the driver-owned timestamped snapshot, the matched
native consumer, and common driver decision/effect diagnostics described in
[`README.md`](README.md).

## Executed CPU path

The retained source was tested with:

```text
taskset -c 18 make -j2 test-offline \
  BPFTIME_ROOT=/home/yunwei37/workspace/gpu/bpftime-r5 \
  BPFTIME_BUILD=/home/yunwei37/workspace/gpu/bpftime-r5/build-r5-v2
```

The clean bpftime source was at ordinary Git revision `ea9907d1df4b` on
`revision/r5-safety-evidence`. The BPF path used clang 18.1.3 and llvm-objcopy
18.1.3; the host wrapper used g++ 13.3.0. These are build identities, not
file/content integrity evidence.

The command exited 0 with:

- 12 boundary/ABI JIT calls, covering both legal actions, zero sequence,
  invalid phases, zero and reversed timestamps, a future publication, torn or
  reserved inputs, an undersized context, and a null wrapper input;
- 306,012 total actual JIT calls and zero wrapper contract errors;
- 102,000 native/JIT comparisons at each of 0, 100, and 1,000 ms publication
  delay, with exact action and full context equality in every comparison;
- dense plus sparse decisions equal to 102,000 for every delay; and
- 13/13 existing offline protocol tests passing.

The deterministic stream's wrong-phase counts were 0, 5,162, and 51,139 at
0, 100, and 1,000 ms. They only confirm that the CPU fixture exercises the
intended freshness contrast. They are synthetic diagnostics and must not be
reported as measured workload behavior.

## Review

Read-only OpenCode plan review session `ses_f94faad6bffeYejEMeFZVKnia1`
returned **READY WITH CONDITIONS**. Its conditions were integrated: the JIT
path is part of `make test-offline`, compiler entry points are version-pinned,
load/execute failure has no native fallback, every prior model boundary is
covered, and counters close exactly.

Fresh read-only result review session `ses_f94f41458ffeqwZbvaneF7a7bJ`
returned **READY**. It accepted the real bytecode-load/JIT path, ABI guards,
deterministic stream, exact comparison, and scope language. One sentence in the
review referred broadly to callback/snapshot/record closure; this CPU run has
no driver callbacks or live records, so only the JIT call and contract-error
counters close here.

A same-session follow-up reviewed a post-review safety correction: the wrapper
now handles a size mismatch before any typed context dereference, and the test
uses an aligned constructed context with a deliberately short declared length.
The complete suite was rerun unchanged. The reviewer confirmed that this fixes
the potential out-of-bounds read and retains **READY**.

Both sessions used model `opencode/ling-3.0-flash-fin-free`, disabled snapshots
and sharing, and denied write, edit, shell, web-fetch, and task tools. The plan
session initially printed proposed shell calls as text; the calls did not run,
and the same session completed its review from supplied context.

## Five-part verdict

- **Run status:** valid for the CPU-only dependency scope.
- **Tested hypothesis:** supported; the actual host-uBPF JIT preserves the
  bounded native decision model for all tested inputs.
- **Research value:** dependency-only; it enables a faithful future BPF arm but
  adds no standalone paper result.
- **Paper impact:** no result-number change. It strengthens readiness and keeps
  the stale-state discussion explicitly prospective.
- **Next paper decision:** do not claim the stale-state question answered. Add
  the shared driver snapshot/native consumer/diagnostics, then run one excluded
  seven-cell real preflight before the 21-cell campaign.
