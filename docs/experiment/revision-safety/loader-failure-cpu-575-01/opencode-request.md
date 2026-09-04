# OpenCode read-only review request

You are an independent systems reviewer. Do not edit files, run commands, use
tools, or invent results. Review the attached plan, harness, implementation
source excerpts, raw summary, environment, and representative logs.

Check these questions:

1. Does the harness call the real interposed `BPF_PROG_LOAD` path rather than
   calling a verifier test function?
2. Is the invalid/control program pair meaningful, and do the three fresh
   repetitions support the claimed outcomes?
3. Does equality between the post-rejection valid program ID and the fresh
   valid-only ID, together with the loader logs, support the narrowly worded
   claim that STRICT rejection did not allocate a program slot?
4. Are STRICT, WARNING, NO_VERIFY, and the unset default interpreted honestly,
   especially the fact that only explicit STRICT is fail-closed here?
5. Are CUDA/GPU, Linux-kernel-verifier, driver-transition, full-stack-safety,
   and performance exclusions explicit and sufficient?
6. Does the cleanup target only the per-cell private shared-memory object and
   run on failure as well as success?
7. Are the result gates vulnerable to a false pass from missing or unrelated
   output?

Give one concise verdict: PASS, PASS WITH REQUIRED FIXES, or FAIL. Separate
blocking findings from optional improvements. State whether the evidence can
support only a loader failure-handling boundary or a broader safety claim.
