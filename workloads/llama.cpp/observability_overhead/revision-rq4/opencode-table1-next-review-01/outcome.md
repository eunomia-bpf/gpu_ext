# OpenCode RTX 5090 Table 1 next-experiment review

This is an advisory design review, not a measurement. All attempts used new
owned CLI processes pinned to CPU 18 with tools, edits, sharing, snapshots and
updates disabled.

The initial command exited 1 because `--file` consumed the following positional
message; its exact error is in `stderr.log`. The first corrected Qwen 27B call
tried to emit a denied Bash tool call and stopped without a final review. The
second corrected call explicitly required a no-tool answer from the attached
request; it exited 0. Its raw stream is [`retry2-events.jsonl`](retry2-events.jsonl)
and its complete response is [`final-review.md`](final-review.md).

The response ranks lossless BPF exit collection first and launch-target/clock
calibration second. It requires exact per-launch and aggregate event closure,
zero drop/pending/error records, a real CUDA launch target, a disclosed common
clock mapping or interval, symmetric boundaries, and exact workload output
before timing. This agrees with the independent source audit. Because the model
had only the request, its suggested ring/perf-buffer details are generic and
must not override the actual per-thread GPU ring implementation. No RTX 5090
timing claim follows from this review.
