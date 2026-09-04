You are reviewing a frozen-candidate systems experiment plan. Work read-only:
do not invoke any tool, shell, edit, web request, or subagent. Inspect every
attached file directly.

Question: Is `plan.md` scientifically valid and executable as a bounded
measurement of the current `verify_gpu_program` API's one-time admission
latency versus instruction count and CFG density?

Check especially:

1. whether both constructed program families can be legal under PREVAIL and
   the GPU uniformity/SIMT checks;
2. whether the claimed source bounds (API allocation check, 65,536 VM default,
   signed 16-bit branch displacement) are stated precisely;
3. whether the matched family construction actually isolates CFG density at
   fixed instruction count;
4. whether fresh-process timing, randomized blocks, metrics, bootstrap, timeout,
   and no-retry rules are enough to interpret the result without silently
   selecting data;
5. whether expected/contradictory/mixed outcomes and claim exclusions avoid
   overclaiming; and
6. whether any defect would invalidate the result, as distinct from optional
   polish or a request for a broader experiment.

Return concise findings with file locations. End with exactly one line:

VERDICT: PASS

or

VERDICT: FAIL
