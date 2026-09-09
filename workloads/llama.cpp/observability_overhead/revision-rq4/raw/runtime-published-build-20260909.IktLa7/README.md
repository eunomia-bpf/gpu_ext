# Table 1: fresh bpftime runtime build

2026-09-09, 11:50:45–11:53:47 PDT: root's build completed with exit zero,
including source/dependency downloads and compilation (about 182 seconds).
It built both runtime libraries in a new directory from public bpftime
revision `eef8a51abaf2ca1f0cdca9f2425af3bd535da1b7`. The old measured
checkout and its build directory were neither used for binaries nor modified.

## Reproduce the build

From the gpu_ext repository root, with the host build dependencies available:

```sh
artifact_runtime_out=$(mktemp -d /tmp/gpubpf-table1-runtime.XXXXXX)
bash scripts/artifact/build_table1_runtime.sh \
  --output-dir "$artifact_runtime_out" --jobs 2
```

The [published entrypoint](../../../../../../scripts/artifact/build_table1_runtime.sh)
accepts explicit source revision/repository, CUDA, LLVM and compiler paths.
The output directory must be empty or new; nothing is deleted. It checks out
the requested source and its Git submodules, configures Debug with CUDA
attach, LLVM/uBPF JIT and the userspace verifier, then builds only
`bpftime-agent` and `bpftime-syscall-server`. Git commit IDs are recorded
as source versions; there is no separate content-integrity procedure.

[build.sh](build.sh) records root's exact invocation, wrapped by both shared
leases. [runner.log](runner.log) records its start/end and commands;
[build.log](build.log) retains the complete download/configure/compiler output.
[build-report.json](build-report.json) records the actual source/submodule
commits, resolved compiler/configuration paths and resulting libraries.
The local-model script used for this run is also captured here, based on
main `21b85e58`; it matches the reviewed entrypoint at build completion.

| Built library | Size, bytes |
| --- | ---: |
| `runtime/agent/libbpftime-agent.so` | 277,369,912 |
| `runtime/syscall-server/libbpftime-syscall-server.so` | 272,480,568 |

The fresh source checkout remained clean after compilation. Large binaries,
downloaded dependencies and build caches stay outside Git; only the small
recipe, logs, source capture and report are published.

## Scope

This closes the observed runtime-library build gap on the existing host,
using CUDA 12.9, LLVM 15 and the installed compilers/development packages.
It does not establish installation on a fresh OS, rebuilding llama.cpp or
obtaining its model, or executing probes/benchmarks with these new libraries.
It does not establish that the new binaries are identical to historically
measured binaries. No GPU performance cell was run, and no Table-1/P40 or
optimization result was replaced. See the
[runtime map](../../../../../../docs/artifact/table1-runtime.md) for the
separate measured cohorts and remaining execution dependencies.
