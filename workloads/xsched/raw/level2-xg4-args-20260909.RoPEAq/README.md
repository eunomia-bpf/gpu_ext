# XSched Level-2: host argument observation completed

2026-09-09 PDT, RTX 5090 / NVIDIA 575.57.08. This is one diagnostic of the
already broken tool-actuated native_port path, not performance evidence.
The local model added read-only ParamCnt/ParamData accessors and a bounded
XG4 log at the two host actuator publication sites. It does not change
the priority policy, delivered argument values, or device instrumentation.
The exact two modified source files are retained in source/.

The isolated HAL builds and installs successfully. run.sh then runs the
existing six-process workload once (2 LC, 4 BE; four queues each;
50 tasks, 9511106 iterations, 340 blocks, 256 threads). Both shared leases
cover build and execution. No driver change or recovery is involved.

| Worker | Exit | Logged argument publications | Host field mismatches |
| --- | ---: | ---: | ---: |
| BE1 | 2 | 64 | 0 |
| BE2 | -2 | 50 | 0 |
| BE3 | 2 | 64 | 0 |
| BE4 | 2 | 64 | 0 |

The 242 observed publications all have five parameters and reps=9511106;
task % 50 equals command_id-1, published command_id matches the command,
and published launch type matches the actuator type. These observations
include later commands, not only the initial command. The log is capped
at 64 per process, so it does not cover every launch.

BE1, BE3 and BE4 nevertheless report zero completed blocks for stream-0
commands 2–5. Their stream-0 command-1 counts are 680, 340 and 680,
respectively (expected 340). The existing runner reports BE1's output
failure and interrupts the remaining BE2 worker; BE2 has no completed
output result, rather than being a successful or independently failed
correctness case. Both LC workers exit zero; the cell/runner exit is 1.

The data narrow the failure beyond these observed host-side argument
copies and published fields. They do **not** prove that NVBit delivers
those values correctly to device execution, nor that the guardian/return
path is correct. Thus this does not justify the earlier proposed shortcut
"host task correct => device delivery clean". Device argument delivery,
guardian state and return/restore behavior remain open; no new performance
win or complete Level-2 reproduction is claimed.

host-argument-observations.json contains all observed fields and existing
failure-window lines; original worker JSON, failure.json and run.log
remain. The previous candidates and all prior failed cells are untouched.
The already built workload from blocks-done-20260909.ykPiLQ is reused,
and only the isolated tool HAL is rebuilt (not the native-blob HAL or
shared bpftime runtime). No manuscript is edited.
