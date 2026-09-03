# Table 1 controlled 575 runtime

Base bpftime source: `d6316fa`; LLVM-JIT submodule: `f66cafa`.
Apply `runtime-575.patch` to that base and the sibling
`../gpubpf-observability.patch` to the example tools. The former records only
the selected CUDA attachment repairs, actual-thread-count metadata fix,
and early/plugin diagnostic-output routing repairs with a narrow unit-test update;
the latter includes complete histogram readback. No verifier rule is changed.

`preparation.json` is the preparer's original record, including its 23-test
checkpoint, before root's build and subsequent CLI-output repair. It is not
the final experiment result. The main runner now passes 24 CPU tests.
Root built `bpftime-agent`, `bpftime-syscall-server` and the associated PTX
passes/compiler in the separate `bpftime-table1-575/build-table1-575` tree:
Debug, CUDA attachment ON, LLVM JIT ON, verifier OFF, unit testing OFF,
CUDA 12.9 and LLVM 15. The build completed 108/108 after initializing the
pinned nested uBPF dependency; the first failed build remains retained.
See the [build logs](../../../../../docs/experiment/revision-safety/table1-runtime-build-575-01/).

Use both runtime-root and build-directory flags in the [plan](../plan.md),
with CUDA 12.9's bin directory in the sudo coordinator PATH. Preserve the
compiled absolute paths to the PTX tools. Do not overwrite the independently
verified R5 runtime or the dirty main bpftime worktree. This performance build
does not establish strict verifier enforcement or a completed Table 1 result.

The subsequent [bootstrap-output repair](../bootstrap-output-repair.md) adds
`runtime/agent/bootstrap_logger.hpp`, initializes logging before CUDA registration
and shared-memory setup, and sends only the extraction child's diagnostics to
stderr. The actual header passes three CPU output-routing cases and the private
agent was rebuilt. A failed intermediate build is retained explicitly. No new
GPU result is implied; exact application output, coverage and clock correlation
remain required before the next performance campaign.

The remaining [PTX-pass output repair](../ptxpass-output-repair.md) rebuilds all
three plugins and the agent, then verifies the actual plugin's stdout/stderr
in a CPU process. It includes the prior failed check. Unit testing remains
OFF; the updated in-tree unit case was not executed by this build.
