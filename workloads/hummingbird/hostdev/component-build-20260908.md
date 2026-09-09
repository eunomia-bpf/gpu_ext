# Hummingbird host/device mapping: first compiled component

2026-09-08. This is incremental implementation evidence, not a completed
host/device experiment or a new performance result. The original host-only
and pipeline measurements remain unchanged.

Local OpenCode session `ses_f80c496aeffeBh330rKhrRwPgX` implemented the
shared coordinate ABI and mapping policy. For each dimension, the mapping
adds the host tile offset to the local block index. Native and BPF engines
share this arithmetic; their engine identifiers intentionally differ.
Actual consumption by the existing DNN kernels remains implementation work.

Root's initial component builds found and corrected two compatibility issues:
the context initially occupied 52 bytes instead of 48, and the shared header
needed separate C `_Static_assert` / C++ `static_assert` spellings. The
model corrected the field count; root applied the conditional assertion.
The first export also rejected the unrecognized `hb_device_map` section.
Root changed it to the existing GPU section convention
`cuda__/hb_device_map`; no verifier or runtime changes were made.

The following commands then both exited zero, from the repository root:

```sh
clang -g -O2 -target bpf -c workloads/hummingbird/hostdev/hb_map.bpf.c \
  -o /tmp/hummingbird-hostdev-build-20260908.nxilLJ/hb_map.bpf.o
workloads/xsched/level2-build/.output/compiler/bpf_to_ptx \
  /tmp/hummingbird-hostdev-build-20260908.nxilLJ/hb_map.bpf.o \
  cuda__/hb_device_map \
  /tmp/hummingbird-hostdev-build-20260908.nxilLJ/hb_map.ptx \
  hb_device_bpf_map sm_120
```

The exporter loaded 18 eBPF instruction words, accepted the program against
its existing 48-byte PREVAIL context and uniformity/SIMT checks, and emitted
the device-callable PTX function. It reported PTX target `sm_120` and LLVM
codegen target `sm_86`. This is compilation evidence only: no cubin assembly,
GPU execution, workload integration, or latency/throughput measurement is
established by this step. Build outputs remain temporary and are not committed.

## Local exporter and native mapping build

The subsequent local-model handoff adds `compiler.mk`, `hb_map_exporter.cpp`
and the native `hb_map_cuda.cuh` device function. Root built these components
without modifying shared bpftime sources or archives:

```sh
make -C workloads/hummingbird/hostdev -f compiler.mk -j2 \
  BUILD=/tmp/hummingbird-hostdev-build-20260908.nxilLJ/compiler
/tmp/hummingbird-hostdev-build-20260908.nxilLJ/compiler/hb_map_exporter \
  /tmp/hummingbird-hostdev-build-20260908.nxilLJ/hb_map.bpf.o \
  cuda__/hb_device_map \
  /tmp/hummingbird-hostdev-build-20260908.nxilLJ/hb_map-own.ptx \
  hb_device_bpf_map sm_120
/usr/local/cuda-12.9/bin/nvcc -x cu -ptx -rdc=true \
  --keep-device-functions -arch=sm_120 -std=c++14 -O3 \
  workloads/hummingbird/hostdev/hb_map_cuda.cuh \
  -o /tmp/hummingbird-hostdev-build-20260908.nxilLJ/hb_map_native.ptx
```

All three commands exited zero. The local exporter accepted and translated
the same 18-instruction program; the native output contains the device
function `hb_device_map`. The build log is temporary at
`/tmp/hummingbird-hostdev-build-20260908.nxilLJ/exporter-build.log`.
The native wrapper deliberately remains callable for the upcoming matched
integration. This does not establish its overhead relative to the original
inline mapping. Actual kernel consumption, cubin integration and the
host/device performance comparison remain unfinished; no GPU was run here.
