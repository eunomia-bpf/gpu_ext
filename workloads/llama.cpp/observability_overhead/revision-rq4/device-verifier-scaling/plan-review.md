# Device-verifier scaling plan review

- Date: 2026-09-04
- Reviewer: OpenCode 1.18.27 with
  `spark-gateway/qwen3.8-27b-nvfp4-200k`
- Session: `ses_f92d8c29bffeWFJt4ytk8ZJiwH`
- Mode: `opencode run --pure`; snapshots and sharing disabled; all OpenCode
  permissions denied; write, edit, shell, web, and delegation tools disabled
- Evidence: `raw/opencode-plan-review-events.jsonl` and
  `raw/opencode-plan-review-followup-events.jsonl`
- Final verdict: **PASS**

The first response was truncated while writing finding 5, before the required
verdict. A single same-session follow-up completed findings 5--6 and returned
the required `VERDICT: PASS`; it did not restart or broaden the review.

## Findings

1. Both program families can be legal under the attached implementation.
   Helper 510 is modeled as uniform with no out-parameters or prohibited effect;
   the linear family terminates directly; and each `JEQ +1` diamond is forward,
   rejoins, and branches on a uniform value.
2. The stated bounds match source: the public API has only its vector-capacity
   guard, the execution VM's default is 65,536 instructions, and branch
   displacement is signed 16-bit. The plan correctly does not claim to measure
   65,536 instructions.
3. The common prefix, helper, exit, exact length, build, CPU, schedule, and
   repetitions make the paired linear/diamonds comparison a valid isolation of
   branch/join density over the tested constructors.
4. The fixed schedule, fresh processes, fixed warmup count, timeout, no-retry
   rule, complete-block bootstrap, strict cardinality, and independent replay
   prevent silent row selection.
5. The predeclared exponent rule is internally complete, and the plan labels
   1.25 as an empirical threshold rather than a complexity proof. The claim
   exclusions prevent promotion to soundness, GPU execution, full attach cost,
   portability, or the runtime maximum.
6. No scientific or executability blocker remains. The reviewer noted only
   that the exact prefix and ALU opcode will live in the probe; the frozen
   structural checks and independent reconstruction make this optional plan
   detail rather than a validity defect.

