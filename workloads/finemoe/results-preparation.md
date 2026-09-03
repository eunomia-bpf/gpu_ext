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
