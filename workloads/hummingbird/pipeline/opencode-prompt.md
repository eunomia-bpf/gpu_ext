Draft a small concrete patch proposal; do not edit files or run commands.
Use the configured default model, snapshot=false, and only read/glob/grep/list.
Do not compute or record file hashes/checksums/digests. Do not read credentials,
active raw data, or unrelated repositories. Return your complete final report
with proposed code snippets/diff and specific risks, not only a plan summary.

Task: prepare a Hummingbird completion-fence ablation in
/home/yunwei37/workspace/gpu/gpu_ext/workloads/hummingbird/pipeline/.
Read the parent directory's idle_policy.c/h, idle_executor.cpp/h,
hummingbird_client.cpp, Makefile, run_study.py, test_idle_policy.cpp,
test_study.py, plan.md, plan-review.md and results-575-20260903.md.
The old published source/build/raw must remain unchanged: suggest a patch
applied to private copied sources plus a small private build/runner wrapper.
Reuse the original cubin, DISB/executor libraries, real VGG/ResNet models,
profile, arrivals, numerical checks, GPU leases, telemetry and result metrics.
No GPU, nvcc, dependency rebuild, service change, Git write or new framework.

The completed 50-cell result is negative: idle C/BPF lose about 19–20% BE
goodput versus fixed GPreempt. The per-piece completion fence is a possible
cause, NOT a measured cause. The paper says 1.3% slowdown, NOT 1.3 us.
Root verified paper v2 p8: issue the next kernel near completion after
duration minus launch_overhead; its device queue bound is at most one.
Two outstanding host event records are not proof of two queued hardware
kernels, and no microsecond preemption guarantee may be inferred.

Prepare two explicitly fixed bounds: default 1 retains the old behavior;
opt-in 2 allows one next launch at the predicted tick while the preceding
completion is pending. Native C and actual ubpf JIT receive the same new
outstanding/bound observations and make the allow/wait decision themselves;
never native-prefilter the BPF admission. Keep HP admission lock/recheck, stop
new LP launch as soon as HP publishes pending, preserve same-stream ordering
across original kernels, split-grid exact-once and no-op copies. Track each
issued launch with a ring event, reuse only after a successful query retires
it; record completion and bounded outstanding evidence, fully drain every
request before reporting completion. This is an ablation of the extra fence,
not automatically a complete original Hummingbird reproduction.

Identify the smallest files/changes, supply a concrete safe event-ring patch,
and targeted CPU tests for bound 1/2, underprediction/full ring, HP stop,
retirement/event reuse, cross-kernel ordering and drain. Include actual-JIT
parity tests. Propose four same-frontend cells C/BPF x bound 1/2 using the
existing frozen two arrivals, five paired blocks and 60 s measurement;
short real preflight is later root-only. Do not tune depth/profile on outcome.
Explain how the wrapper can reuse run_study.py's safety and measurements while
auditing the new bound and private binaries without rewriting the framework.
Review code growth and unauthorized scope changes. State remaining real-GPU
admission requirements explicitly; CPU checks are not experimental results.
