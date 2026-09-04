Perform a strict read-only implementation review. Do not invoke any tool,
shell, edit, web request, or subagent. Inspect every attached file directly.

The frozen plan has already passed plan review. Decide whether this exact
implementation is ready for a later CPU-only real preflight, not whether a
preflight or timing result exists now.

Audit:

1. the C++ constructors and structural checks against the frozen linear and
   warp-uniform-diamond formulas and the real `verify_gpu_program` API;
2. that `--describe` truly performs no verifier call/timing and a timed process
   performs exactly one API call with construction outside the interval;
3. isolated Release build provenance and separation from the strict runtime
   build/DSOs;
4. exact schedule, fresh-process execution, affinity/environment gates,
   timeout/no-retry behavior, incremental failure preservation, and absence of
   GPU work;
5. independent analyzer replay, raw evidence binding, block bootstrap,
   Theil--Sen exponent, paired ratio, noise veto, and complete interpretation
   rule; and
6. tests, especially whether malformed or selectively missing evidence can be
   accepted or cause an analyzer crash.

Report only concrete blockers and important non-blocking limitations with file
locations. End with exactly one line:

VERDICT: PASS

or

VERDICT: FAIL
