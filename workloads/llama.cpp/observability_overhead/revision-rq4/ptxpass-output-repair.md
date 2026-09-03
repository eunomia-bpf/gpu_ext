# PTX-pass output repair — 2026-09-03

The remaining stdout line in [histogram diagnostic 01](targeted-diagnostic-results.md)
comes from `ptxpass_core::log_transform_stats`. The plugin has its own default
logging registry, so configuring the agent's logger did not redirect this
call. The core header already describes transformation statistics as stderr
diagnostics. The function now writes them explicitly to stderr using its
existing stdio dependency; application stdout and the output oracle are untouched.

The private runtime rebuilt the core, all three statically linked PTX plugins
and the agent: nine build steps, exit zero. The new kretprobe library is
133,517,464 bytes; the agent is 242,955,944 bytes. The controlled
[runtime patch](runtime-575/runtime-575.patch) includes the function change and
the narrow unit-test expectation update; reverse application checking passes.
Unit testing is OFF in this build, so the updated unit case and full suite
were **not** run or claimed to pass.

[test_ptxpass_output.py](runtime-575/test_ptxpass_output.py) loads the actual
rebuilt kretprobe plugin in a fresh CPU process and transforms a real minimal
PTX kernel using the actual LLVM path. Before repair, the successful transform
emitted 123 stdout bytes including diagnostics. After repair, both the subagent
and root runs preserve exactly the 28-byte application sentinel on stdout,
with the diagnostic and test report on stderr. The output still contains the
injected call and is 230 bytes; the plugin returns success and `modified=true`.
No CUDA context or GPU workload is initialized by this check.

The [raw build and before/after/root-after streams](../../../../docs/experiment/revision-safety/table1-ptxpass-output-575-01/)
are retained, including the old failed stdout check. Reproduce with a fresh
output directory:

```sh
taskset -c 17 python3 -B runtime-575/test_ptxpass_output.py \
  --build-dir /home/yunwei37/workspace/gpu/bpftime-table1-575/build-table1-575 \
  --output /tmp/table1-ptxpass-cpu-new
```

The subsequent [three-arm histogram diagnostic 02](targeted-diagnostic-results.md)
also passes exact application stdout and aggregate-count checks on the GPU.
Neither the CPU regression nor that untimed diagnostic is a performance run.
The independent
[host-stub investigation](../../../../docs/experiment/revision-table1-host-stub-diagnosis-20260903.md)
also finds a possible optimized-away launch hook; it is source diagnosis, not
a repaired launch-latency path.
