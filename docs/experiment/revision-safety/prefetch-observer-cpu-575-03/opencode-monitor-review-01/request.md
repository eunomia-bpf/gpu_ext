You are an independent read-only systems reviewer. Do not call tools, edit
files, run commands, launch GPU work, or use the network. The complete relevant
files are attached directly.

Audit only the compute-process lifecycle gate and cleanup behavior in the
attached `run_safety.py`, its synthetic tests, monitor, and README. Answer:

1. Can a sample taken before the target is actually released satisfy the
   post-release gate?
2. Are empty-before-start, target-only-at-pause/ready/after-release,
   empty-after-exit, foreign-PID rejection, and the one-second maximum gap all
   reconciled against saved timestamps?
3. Once loader attachment IDs are known, does every failure path on which the
   loader group has exited check that its tracing and policy links disappeared?
4. Can SIGINT/SIGTERM interrupt child registration or any cleanup stage, or is
   cancellation queued and raised only after owned cleanup and record writing?
5. Give `READY` or `REQUIRED FIXES`, separating blockers from limitations. The
   accepted limitation is that periodic sampling cannot see a foreign process
   whose complete lifetime falls between samples; do not treat the gate as
   continuous proof.

Do not infer measurements and keep the response concise.
