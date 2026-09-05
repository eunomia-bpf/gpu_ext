# Endpoint-v1 bounded module lifecycle

`run_endpoint_module_lifecycle.py` is the fail-closed wrapper for the one
launch-clock recovery probe that needs the endpoint-v1 core module. Its default
mode is CPU-only admission; no module or cluster state changes without the
explicit `--execute` flag.

The wrapper admits only the fixed prebuilt directory
`/opt/gpubpf/modules/575.57.08/launchlate-endpoint-86e7e0dd-575-02` and the
exact known-good rollback stage `gpreempt-849ea75d-6.15.11`; it has no source
checkout dependency. Before any live-state mutation, it validates all four
candidate and rollback modules using ordinary file inventory and size,
module version, vermagic, dependencies, parameter names, and BTF interfaces.
It additionally requires both the versioned endpoint-v1 and stock correlation
implementation symbols in the candidate `nvidia.ko`. It then copies the
admitted candidate into a fresh stage and repeats the same artifact validation.
It does not require the unrelated scheduler-init diagnostic ABI.

Execution takes both existing experiment leases, snapshots the boot, module
parameters and runtime interfaces, services, device nodes, k3s node readiness
and labels, and safety counters. It removes only the two admitted scheduling
labels, stops the admitted display/persistence services, proves there are no
device holders, and uses plain `rmmod` plus explicit `insmod` in dependency
order. After the probe, or after any exception or interrupt once removal has
started, it removes the candidate and restores the exact four rollback modules,
parameters, 400 W limit, nodes, services, labels, boot identity, and safety
state. The result cannot be marked complete until rollback succeeds and both
leases close.

The probe gate requires exactly 200 valid, ordered endpoint-v1 samples and one
passing summary, including successful cleanup and no CPU/PTIMER regressions.
The candidate is never left installed intentionally. A rollback failure
withholds service and label restoration so workloads cannot be rescheduled onto
an unvalidated stack; manual recovery is then required.

The optional child is not an arbitrary callback. `--child-mode preflight` runs
only the fixed launchlate preflight; `--child-mode preflight-full` runs that
preflight and, only after its independent analyzer passes, the fixed ten-block
full campaign. The wrapper supplies the exact runner, model defaults, bpftime
source/build, tool selection, GPU-thread count, result subdirectories, and both
already-held read-only lease descriptors. It uses neither a shell nor an
ambient injection environment. Each child argv, return code, stdout, stderr,
and independent analyzer result is retained in `lifecycle.json`; wrapper
completion requires every requested child gate before rollback.

Attempt 12 requests one fixed pair, 2392 MHz SM / 14001 MHz memory, and admits
only P0 observations whose memory clock is exactly 14001 MHz and whose SM clock
is in the supported, explicitly enumerated set `{2392, 2400}`. Both pairs must
appear in the device's supported-clock query. This is a finite enumerated-bin
set, not a tolerance or range: every other SM value, memory value, or P-state is
rejected. The wrapper checks this state before the probe and before and after
each child. Clock resets are attempted in reverse order before module rollback
on every exit path.

The child does not inherit the invoking shell's executable search path. Its
PATH is fixed to the CUDA 12.9, standard local, and system binary directories,
and admission resolves `cuobjdump`, `nvcc`, `git`, `make`, `nvidia-smi`,
`patch`, and `taskset` to their exact allowlisted locations. This specifically
prevents a repeat of attempt 06, where the endpoint probe passed but the child
stopped before any workload because `cuobjdump` was outside the inherited
minimal sudo PATH.

The only admitted live invocation for the next fresh attempt is:

```bash
sudo -n python3 /home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/observability_overhead/revision-rq4/launch-clock-recovery/run_endpoint_module_lifecycle.py \
  --candidate-dir /opt/gpubpf/modules/575.57.08/launchlate-endpoint-86e7e0dd-575-02 \
  --stage /opt/gpubpf/modules/575.57.08/launchlate-endpoint-stage-575-12 \
  --output /home/yunwei37/workspace/gpu/gpu_ext/workloads/llama.cpp/observability_overhead/revision-rq4/raw/rm-correlation-575-12-endpoint-lifecycle \
  --child-mode preflight-full \
  --execute
```

Run it only after the current GPU experiment has released both shared leases.
This wrapper does not authorize any later launchlate timing campaign.
