# Deployment and portability evidence audit

## Bottom line

This audit supports two generic, CPU-only deployment routes: process-start
`LD_PRELOAD` and running-process injection through bpftime's Frida-backed CLI.
It also confirms that the 575.57.08 open-module tree contains built scheduler
and UVM extension symbols. It does **not** establish GPU hook correctness,
first-device-event latency, a product SASS backend, or an AMD/Intel port.

The paper-facing decisions are therefore:

| Claim | Evidence status | Required disposition |
|---|---|---|
| PTX implementation exists | Source-audited; not executed here | May describe as the current implementation, while citing GPU evidence from a separate GPU run only |
| Working SASS prototype | Unsupported | Remove. The discovered NVBit code is a benchmark/example, not an end-to-end gpubpf backend |
| Running-process attach uses ptrace | Runnable on the opt-in CPU target | State precisely: the CLI delegates injection to Frida; Frida issued ptrace syscalls in this diagnostic |
| `LD_PRELOAD` deployment route | Runnable on the CPU target | May state that the generic agent can initialize at process start; do not imply device-hook validation here |
| 273 ms one-time attach latency | Unsupported by retained samples | Remove the number until the frozen GPU protocol is run |
| Approximately 100 LOC in open modules | Contradicted for the present full production delta | Remove or replace with a precisely defined and reproducible narrower boundary |
| AMD/Intel portability | Discussion only | Present as an architecture direction, not a completed port |

## Runnable CPU evidence

The bpftime tree was freshly built with
`BPFTIME_ENABLE_CUDA_ATTACH:BOOL=OFF`. The CLI, agent, and syscall-server
targets built successfully. The agent's dynamic dependencies contain no CUDA
library. The link step emitted a pre-existing executable-stack warning from an
assembly object; the build nevertheless completed. See
[`raw/cpu-build.log`](raw/cpu-build.log),
[`raw/bpftime-artifacts.tsv`](raw/bpftime-artifacts.tsv), and
[`raw/agent-ldd.txt`](raw/agent-ldd.txt).

Five matched repetitions were run in the fixed order preload then attach. Each
repetition used a fresh shared-memory name and required all of the following:
the target remained alive, no agent mapping existed before activation, an
agent or Frida mapping appeared afterward, `BPFTIME_USED=1` became visible in
the target, the per-target IPC socket appeared, and the command succeeded.
All ten repetitions passed.

| Route | Repetitions | Ready mean (ms) | Min--max (ms) | Failures |
|---|---:|---:|---:|---:|
| Process-start preload | 5 | 6.890 | 6.523--7.385 | 0 |
| Frida-backed running-process attach | 5 | 14.637 | 13.226--15.561 | 0 |

These small measurements are lifecycle checks, not performance results. The
ready timestamp is the target's first observation of `BPFTIME_USED`; no CUDA
library, GPU program, GPU hook, or first device event is involved. The raw
samples are in [`raw/lifecycle.tsv`](raw/lifecycle.tsv) and the derived values
are in [`raw/lifecycle-summary.tsv`](raw/lifecycle-summary.tsv).

The host used Yama `ptrace_scope=1`. The synthetic target explicitly called
`PR_SET_PTRACER_ANY`; this is recorded in
[`raw/run-environment.tsv`](raw/run-environment.tsv). A separate `strace`
diagnostic observed 20 ptrace-family operations, including one
`PTRACE_SEIZE`, and a final detach. The target stayed alive, had zero matching
mappings before injection and five afterward, and exposed the IPC endpoint.
This verifies an actual ptrace lifecycle inside Frida for this opted-in target,
without `sudo`. It does not show that an unmodified process under a restrictive
ptrace policy is attachable. See
[`raw/ptrace-diagnostic.tsv`](raw/ptrace-diagnostic.tsv) and
[`raw/attach-ptrace-syscalls.txt`](raw/attach-ptrace-syscalls.txt).

## Source audit: PTX versus SASS

The product attach implementation under `attach/nv_attach_impl` contains the
PTX compiler and eBPF-to-PTX transformation path. It contains no NVBit or SASS
implementation marker in C, C++, CUDA, or header sources.

The NVBit implementation examined in this audit is under
`benchmark/gpu/nvbit`. Its host example obtains SASS instructions and inserts
calls before and after instructions. However, both injected device timing
functions have their clock reads and writes commented out, so their active
bodies are no-ops. More importantly, this directory neither consumes a gpubpf
eBPF program nor connects to the product attach implementation. It is useful
as an NVBit API experiment or baseline, but not evidence of a working SASS
gpubpf prototype. The fail-closed checks are recorded in
[`raw/semantic-checks.tsv`](raw/semantic-checks.tsv).

## Open-module boundary audit

The local 575 tree was compared with NVIDIA's official 575.57.08 tag source.
The downloaded release archive was 18,948,158 bytes. Tests, generated build
outputs, and NVIDIA binary objects were excluded from the production-source
boundary.

Within the audited production boundary, nine new source/header files contain
2,017 physical lines. Nine existing upstream production files differ by 539
added and 49 deleted lines. The combined file-level delta is therefore 2,556
added and 49 deleted physical lines across 18 files. This is not a semantic
source-line metric, but it is an explicit, reproducible comparison and is
incompatible with an unqualified claim that the present open-module change is
approximately 100 LOC. Detailed counts are in
[`raw/open-module-delta.tsv`](raw/open-module-delta.tsv).

Five `.ko` artifacts identify version 575.57.08 and the running kernel's
vermagic. `nvidia.ko` exposes 23 audited scheduler-related defined symbols and
`nvidia-uvm.ko` exposes 16 UVM extension symbols. This is artifact/symbol
inspection only: the audit did not load modules or exercise GPU hardware. See
[`raw/module-artifacts.tsv`](raw/module-artifacts.tsv) and
[`raw/module-symbols.txt`](raw/module-symbols.txt).

## Source identity and limitations

The audited bpftime revision is
`d6316fa73edaac4fdfe21b89d4470da6cd9b8ae8` on
`refs/heads/review/pr-253-fix`. The local 575 driver revision is
`6a5b3bb5857e5e2890559750f51e0a0e5834d09d` on
`refs/heads/test-sched`. Exact inputs are recorded in
[`raw/revisions.tsv`](raw/revisions.tsv).

No GPU command was run, no module was loaded, no elevated privilege was used,
and no repository state was changed through Git. The retained measurements do
not recover or corroborate 273 ms. The only defensible next step for that
number is the protocol in [`future-gpu-protocol.md`](future-gpu-protocol.md).
