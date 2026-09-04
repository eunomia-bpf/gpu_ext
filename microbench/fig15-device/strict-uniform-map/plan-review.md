# Plan review

## OpenCode / Qwen attempt

- Date: 2026-09-04
- OpenCode: 1.18.27, `--pure --variant minimal`
- Model: `spark-gateway/qwen3.8-27b-nvfp4-200k`
- Controls: CPU 20; GPU hidden; snapshots and sharing disabled; all
  permissions denied; write, edit, shell, web, and delegation tools disabled
- Inputs: `plan.md` and `opencode-plan-review-request.md`
- Bound: one 180-second attempt; no retry

The process reached the external timeout with no event or text. It supplied no
review and no verdict, so it is not counted as a plan-review pass. Execution is
not gated on this unavailable advisory result. The real preflight must instead
demonstrate all six paths end to end, including target-PID STRICT acceptance,
positive verifier timing, exact map descriptors, map-effect readback, one
target transformation/module load/attach, and cleanup. Any failed condition
stops that campaign.
