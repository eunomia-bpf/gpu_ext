# Reproducing the CPU-only deployment audit

This directory separates what can be run without a GPU from source-only and
future-GPU claims. Read [`RESULTS.md`](RESULTS.md) for the conclusions.

## Inputs used for the retained run

- bpftime source: `/home/yunwei37/workspace/gpu/bpftime`
- bpftime CPU build: `/home/yunwei37/workspace/gpu/bpftime/build`
- modified 575 source: `/home/yunwei37/workspace/gpu/gpu_ext-kernel-575`
- NVIDIA reference release: official tag `575.57.08`

The reference archive URL is recorded in `raw/revisions.tsv`. The retained
download was 18,948,158 bytes. Exact local refs and ordinary Git revisions are
also recorded there.

## Commands

The scripts require Bash, CMake, a C compiler, `rg`, `strace`, `readelf`,
`nm`, and `modinfo`. They do not invoke a GPU, load a kernel module, use
`sudo`, or run Git commands.

Fetch and extract the exact upstream source release into a temporary directory:

```bash
audit_tmp=$(mktemp -d)
curl -fL \
  https://github.com/NVIDIA/open-gpu-kernel-modules/archive/refs/tags/575.57.08.tar.gz \
  -o "$audit_tmp/open-gpu-kernel-modules-575.57.08.tar.gz"
tar -xf "$audit_tmp/open-gpu-kernel-modules-575.57.08.tar.gz" -C "$audit_tmp"
```

From the `gpu_ext` repository root, run:

```bash
docs/experiment/revision-deployment-575/run_cpu_build.sh \
  /home/yunwei37/workspace/gpu/bpftime \
  /home/yunwei37/workspace/gpu/bpftime/build

docs/experiment/revision-deployment-575/audit_sources.sh \
  /home/yunwei37/workspace/gpu/bpftime \
  /home/yunwei37/workspace/gpu/bpftime/build \
  /home/yunwei37/workspace/gpu/gpu_ext-kernel-575 \
  "$audit_tmp/open-gpu-kernel-modules-575.57.08"

docs/experiment/revision-deployment-575/run_cpu_lifecycle.sh \
  /home/yunwei37/workspace/gpu/bpftime \
  /home/yunwei37/workspace/gpu/bpftime/build 5

docs/experiment/revision-deployment-575/run_ptrace_diagnostic.sh \
  /home/yunwei37/workspace/gpu/bpftime \
  /home/yunwei37/workspace/gpu/bpftime/build
```

Every script exits nonzero on a missing prerequisite or failed semantic or
lifecycle gate. `run_cpu_lifecycle.sh` also rejects inherited `LD_PRELOAD` or
`BPFTIME_USED`, a build without an explicit CUDA-attach-off setting, a CUDA
dependency in the agent, and an unexpected cleanup path.

## Retained evidence

- `raw/cpu-build.log`, `raw/bpftime-artifacts.tsv`, `raw/agent-ldd.txt`, and
  `raw/agent-entry-symbols.txt`: fresh CPU build and artifact inspection.
- `raw/lifecycle.tsv` and `raw/lifecycle-summary.tsv`: five matched repetitions
  of preload and Frida-backed attach.
- `raw/attach-ptrace-syscalls.txt` and `raw/ptrace-diagnostic.tsv`: independent
  confirmation that the injector issued ptrace operations.
- `raw/semantic-checks.tsv` and `raw/source-inventory.tsv`: PTX/SASS and route
  source audit.
- `raw/open-module-delta.tsv` and `raw/open-module-delta-summary.tsv`: explicit
  production-source comparison against the official 575.57.08 release.
- `raw/module-artifacts.tsv` and `raw/module-symbols.txt`: existing 575 module
  identity and symbol inspection; this is not a fresh driver rebuild or load.
- `future-gpu-protocol.md`: predeclared gates for evidence that cannot be
  collected on this CPU-only host.

