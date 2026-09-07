# Next storage-policy experiment: overlapping reads and writes

The completed end-to-end five-arm campaign measures immediate-submit policy
overhead. The policy-input ablation adds fixed pressure/slack/cost inputs, but
its cold-store-then-warm-read sequence does not establish benefit under
read/write contention. Demand reads in that path always submit immediately.

## Selected implementation

`../run_gds_mixed_backend.py` was implemented through local OpenCode/Qwen
27B. Its scheduled-arrival five-block comparison is complete; see
`../results-575-gds-mixed-scheduled-20260907.md`. It drives the installed
LMCache 0.5.4 GdsBackend and the committed
admission adapter, with real cuFile/O_DIRECT operations on 24 MiB objects.
Already-stored demand-read objects and fresh background-write objects are
distinct. Requests overlap and preserve their offered arrival times.

| Arm | Decision implementation | Inputs |
| --- | --- | --- |
| FIFO | Immediate submission | Same offered workload |
| Native | Existing native policy | Controlled pressure 801 permille; slack 10 ms |
| BPF | Same policy through UVM command 82 | Identical inputs to native |

Use bounded GPU buffers, fresh output directories, one first block followed
by five rotated blocks. Build/setup and population are outside the measured
steady interval; all steady-interval writes must complete before its end.

## Measurements and interpretation

Record offer, admission/submission and completion times, bytes, failures and
available decision counts. Report demand-read offer-to-completion p50/p99,
write completion throughput, total bandwidth and makespan. Compare native
against FIFO for policy behavior, and BPF against native for mechanism cost.
Deferral may improve urgent reads, worsen them, or reduce aggregate bandwidth;
the outcome is empirical.

The pressure value is a controlled policy input, not measured live HBM
pressure. Current adapter code delays individual writes; it does **not**
enforce the returned batch target or implement write coalescing/priority
ordering. The previous draft's coalescing claims were unsupported and are
not part of this experiment. There is no modeled busy-loop substitute for
real KV recomputation; recomputation remains outside this selected workload.
The transport remains cuFile compatibility until direct NVMe/GPU DMA is
actually demonstrated.

Prior executor decision medians (native 0.063 us, BPF 1.005 us) motivate
the comparison, but do not predetermine its application-level result.
Collect every attempted cell without correctness, clock, preflight or retry
gates, and retain the preceding campaigns separately.

## Active follow-up: live pending-demand feedback

The completed fixed-delay experiment lowers BPF/FIFO paired read p99 by
12.440% at the median, but worsens read p50 and reduces paired write throughput
by 1.673%. The next implementation responds to actually outstanding demand
reads instead of deferring every write based on a constant pressure input.
Local OpenCode/Qwen Next session `ses_f866bda8dffeljd81GZlstX8rx` attempted
this opt-in variant but exited with CLI status 1 and APIError HTTP 524 before
producing source. It was not stopped for silence or by a root-imposed timeout.
The Qwen 27B takeover in existing LMCache session
`ses_f86e7cf67ffeiTI9l4PQ5lbDWu` also ended with APIError HTTP 524 before
source changes. Its last completed request recorded 97279 input tokens;
this is context-size evidence, not proof of the error's cause. A fresh
Qwen 27B session, `ses_f86352254ffeAZaYGUr1nqjZnI`, has implemented the first
two-file step: matching native/BPF flagged-write decisions. The root built
`gds_policy.bpf.o` with the existing Makefile and Python syntax compilation
passed. `HINT_LIVE_DEMAND` selects the new branch, with a maximum 1 ms
individual deferral; unflagged requests retain the old decisions. This is
source/build evidence, not feedback-policy performance. After the completed
device-tool campaign, the root replaced the earlier owned loader with this
new object; the existing loader printed `attached` on the unchanged 575
driver. No new workload ran for that loading step. That session subsequently exited 0
after reading the integration files, without producing provider/executor or
runner edits. A fresh Qwen 27B session, `ses_f861098a9ffe4pASzKSgl4K6Lx`,
now owns only the backend-adapter file: live pending-read state and the
re-evaluating write executor. A parallel Qwen Next session titled
`gds-live-feedback-runner-step` owns only `run_gds_mixed_backend.py` and adds
the opt-in selection and feedback-record serialization. The shared API is
`LiveDemandRequestProvider(Telemetry(...))` and `adapter.feedback_records`.
No completed
decision step or experiment is repeated. Those parts remain unfinished.
Both earlier API-failed
sessions exited themselves; neither was stopped for silence.
The earlier runner task remains complete and is not rerun. No feedback-policy
result is claimed yet. The live-feedback runner session
`ses_f860b1ef2ffeKrPO1Jhf06vsPb` subsequently exited with status 1 and
`APIError` (`<none>` message), without changing the runner. No HTTP status
was reported for this failure. A fresh local GLM Next session titled
`gds-live-feedback-runner-glm` now owns that same single file; the failed
Qwen Next process is gone, not a duplicate active worker. Qwen 27B completed
the adapter integration and exited normally with its final handoff. Provider
definitions are in `5f2a4ef0` (main `3e09b9e9`); live read accounting and the
deferred-write executor are in `bc6f70ff` (main `f22a085f`). The executor
refreshes pending demand and elapsed budget before every wait, including its
first scheduled execution, and the fixed-delay path still waits only once.
Demand counters close after actual reads, including errors during batched
admission. Python syntax compilation passed. Runner selection and actual
live-feedback performance remain unfinished; this is not a measured result.
GLM owns that remaining runner step. The measured device-array implementation
and its workers are complete.

Native and BPF will consume the same live pending-read count, explicitly
identified by a caller-hint flag in the existing 136-byte command-82 ABI.
Only safe background writes with pending demand and remaining deferral budget
are deferred, in increments of at most 1 ms, with a fresh decision after each
wait. Otherwise they submit immediately. Demand reads always submit. The
default cumulative write-deferral budget is 10 ms, not a process timeout.
The executor and buffer/Future ownership must be shared by native and BPF.
The existing fixed-delay mode remains the default and old records are untouched.
This is demand feedback, not HBM telemetry, recomputation, coalescing or P2P.
