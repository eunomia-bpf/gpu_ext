# Table 1: component-only build completed

2026-09-09, 11:07:17–11:07:53 PDT: the new `--build-only` path completed
with exit zero, building kernelretsnoop, threadhist, launchlate and the
NVBit observability adapter in a new temporary directory. No probe loader,
syscall server, model or benchmark was launched. This is build evidence,
not a new throughput measurement.

## Reproduce this step

[build.sh](build.sh) records the exact command, using explicit bpftime
source/build and NVBit release paths. Root held both shared GPU and
struct-ops leases. The runner used the workload's uv environment.
The output directory was empty before invocation; stdout/stderr were
written here, outside it.

The new command prepares the same default seven-arm tools; it returns
before the campaign loop and writes [prepared-tools.json](prepared-tools.json)
with resolved component locations. It does not emit measurement cells.
`--nvbit-root` supplies the release path to the existing build helper
without changing its global default. `--dry-run --build-only` was also
run with a different, nonexistent NVBit path: it printed the resolved
configuration and did not create the requested output directory.

| Built component | Size (bytes) | Build log |
| --- | ---: | --- |
| kernelretsnoop | 1550248 | [log](kernelretsnoop-build.log) |
| threadhist | 1535760 | [log](threadhist-build.log) |
| launchlate | 1554424 | [log](launchlate-build.log) |
| NVBit observability.so | 3204584 | [log](nvbit-build.log) |

[Top-level output](build.log) and the two changed runner source captures
are retained. The source captures record this candidate on main base
`04f3ec98`; they are not standalone launchers, because their imports
depend on the normal source-directory layout. Build binaries/caches are
not included in this record.

After this build, the published runner's preparation note was clarified to
describe only the tools selected by the mode (auto-warp does not build NVBit).
The source captures and original preparation JSON retain their build-time
wording. The default build logic is unchanged. Root also ran the local
model's 11 CPU-only mocked checks successfully, including explicit NVBit
paths, side-effect-free dry-run and default/auto-warp preparation paths.

## Scope and remaining work

This used the existing published bpftime checkout
`revision/table1-host-plt-fix` (current source revision `eef8a51`),
its already-built dependencies, the local NVBit 1.8 release and CUDA 12.9.
It does not establish rebuilding the entire bpftime runtime or llama.cpp
from a clean host, nor that a current runtime binary exactly matches a
historically measured binary. No existing Table-1, P40, GPU-array or
full-record performance result was rerun or replaced.
