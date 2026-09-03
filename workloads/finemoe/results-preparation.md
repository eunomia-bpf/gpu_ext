# FineMoE preparation: retained first-load failure

No FineMoE numerical correctness or performance result is established yet.

`raw/golden-v1/stage/worker.log` records the original full BF16 Qwen checkpoint
failing while loading its seventh of eight shards. The worker exited 1 before
any generation: **0 of 73 requests and 0 of 9 repeat checks completed**.
The failing allocation requested 20 MiB with 14.62 MiB free; PyTorch reported
25.77 GiB allocated and 4.97 GiB reserved but unallocated. That large unused
reservation is a fragmentation/reservation-pressure clue, not proof of the
precise cause or proof that the full 26.67 GiB checkpoint cannot fit.

The controller's `stage/result.json` preserves the failed status and confirms
clean teardown: no cleanup error, GPU back to 15 MiB with no compute process,
UVM references 0, empty struct-ops state, and no Xid/kernel abnormality. Its
44 telemetry samples show a 32,095 MiB peak and no disallowed throttling.
`campaign.json`, `campaign-failure.json`, launch/environment, raw stdout, and
telemetry remain in the original directory; nothing is relabeled as a valid cell.

The approved next attempt uses the canonical PyTorch environment variable
`PYTORCH_ALLOC_CONF=expandable_segments:True`, uniformly for golden, history,
preflight, and every formal arm. The controller removes an inherited legacy
`PYTORCH_CUDA_ALLOC_CONF` and records only its removed name, not its value.
PyTorch documents this allocator option as experimental; it may reduce some
allocation fragmentation but does not guarantee this workload will fit.
See [PyTorch CUDA environment variables](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)
and [CUDA memory management](https://docs.pytorch.org/docs/main/notes/cuda.html).

The retry will use fresh `raw/golden-v2`. Checkpoint revision, BF16 precision,
the full 24-layer/60-expert model, frozen MT-Bench 64/8/1 cohort, 0.5 offload pool
budget, and repeat-derived numerical-tolerance rule remain unchanged. No extension
rebuild or GPU retry was performed as part of this environment-only repair.

## Second load: memory hurdle passed, native SIGILL remains

After the allocator-only change, `raw/golden-v2/stage/worker.log` confirms all
eight shards loaded. Peak sampled GPU memory was 27,925 MiB, below v1's peak.
The process then terminated with return code −4 (`SIGILL`) before any completed
request: again 0/73 requests and 0/9 repeats. There is no Python traceback, so
the specific native instruction/library is not established by this log.
This is a different failure from v1's OOM, not a numerical result or evidence
that a particular GPU policy caused the failure.

The v2 controller again recorded clean teardown: GPU 15 MiB, no compute process,
UVM references 0, empty struct-ops state, no Xid/kernel abnormality, and no cleanup
error. All 50 telemetry samples passed the existing throttling gate.

The next step is a single same-model native-stack diagnostic in fresh
`raw/golden-sigill-gdb-01`, using `compare.py --mode golden --native-backtrace`.
GDB runs in batch with init files, auto-load and debuginfod disabled; raw stdout
includes `bt 30` and eight instructions at the stopped PC, followed by explicit
inferior termination. Existing leases, telemetry and owned-process cleanup remain
active. Only this diagnostic enables Python faulthandler. Its campaign is marked
`diagnostic=true` and is rejected as a golden/history/preflight reference even
if execution unexpectedly succeeds. A normal no-debugger golden must succeed
before later stages; no model, precision, cohort, budget or tolerance changes
are part of this diagnostic.
