# OpenCode final monitor review

Verdict: **READY**.

1. A pre-release sample cannot satisfy the post-release gate. The consumer
   requires `query_started_mono_ns` to be strictly later than the saved gate;
   release is saved only after the target pipe is written, flushed, and closed.
   Final validation rebinds the marker to the exact stored query and PID set.
2. Pretarget-empty, pause, ready, post-release target-only, and post-detach
   empty samples are exact-matched by query start and finish. All lifecycle
   markers are positive and ordered. Query duration plus idle, start, and
   finish cadence are each bounded to one second; foreign PIDs and monitor
   errors reject the run.
3. Once attachment IDs are known and the loader group has exited, tracing and
   nonzero policy link disappearance is checked before monitor, stream, and
   post-safety cleanup. Later failures cannot skip it. The only `not_attempted`
   cases explicitly retain a surviving target or loader.
4. SIGINT/SIGTERM handlers only append to a queue. Body checkpoints initiate
   cancellation, while cleanup contains no interrupt checkpoint. Cancellation
   is propagated only after owned cleanup attempts and `execution.json`.

No blocker was found. The accepted limitation remains: bounded sampled
exclusivity cannot see a foreign GPU client whose complete lifetime falls
between adjacent queries.
