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

## Real ResNet-152 source transformation and PTX build

Root ran the local-model `hb_split_model.py` against the existing original
ResNet-152 source (299940 bytes) and host metadata (121968 bytes):

```sh
python3 -B workloads/hummingbird/hostdev/hb_split_model.py \
  --source workloads/gpreempt/deps/upstream/model/resnet152/mod.cu \
  --host workloads/gpreempt/deps/upstream/model/resnet152/host.json \
  --output /tmp/hummingbird-hostdev-build-20260908.nxilLJ/resnet152-callable
/usr/local/cuda-12.9/bin/nvcc -ptx -rdc=true --keep-device-functions \
  -arch=sm_120 -std=c++14 -O3 \
  -I/home/yunwei37/workspace/gpu/gpu_ext/workloads/hummingbird/hostdev \
  /tmp/hummingbird-hostdev-build-20260908.nxilLJ/resnet152-callable/mod-bpf.cu \
  -o /tmp/hummingbird-hostdev-build-20260908.nxilLJ/resnet152-callable/mod-native.ptx
```

Both commands exited zero. The transform identifies 44 kernel entrypoints
and 307 recorded launches; the generated CUDA/PTX sizes are 318837/416892
bytes. These are generated build inputs, kept outside Git. The transformed
source uses `hb_ctx.out_*` for the original kernel's coordinate-dependent
computations; it currently calls the **native callable adapter**, not BPF.

The actual optimized PTX has 43 call sites targeting `hb_device_map$13`,
a compiler-specialized local-memory clone. The separate visible
`hb_device_map` definition is not their target. Replacing only that visible
definition would therefore leave the real kernels on the native path.
The clone receives a generic local-context pointer and does not read its
second, unused length parameter. Root supplied these facts to the local
model for the unfinished PTX call-target replacement. No BPF consumption,
cubin integration, GPU execution or performance result is claimed here.

The callable-native PTX was subsequently assembled successfully (exit 0):

```sh
/usr/local/cuda-12.9/bin/ptxas -arch=sm_120 -O3 -v \
  /tmp/hummingbird-hostdev-build-20260908.nxilLJ/resnet152-callable/mod-native.ptx \
  -o /tmp/hummingbird-hostdev-build-20260908.nxilLJ/resnet152-callable/mod-native.cubin
```

The cubin is 441616 bytes; `ptxas-native.log` in the same temporary directory
retains assembly output. This advances the native adapter build only.
The BPF-call replacement, integrated model loading and paired GPU
measurements are still unfinished; no generated binary is committed.

## Integrated BPF cubin and first execution attempt

The local-model `patch_device_map.py` now replaces all 43 actual specialized
call targets with `hb_device_bpf_map` and inserts the exported BPF function.
Root executed:

```sh
python3 -B workloads/hummingbird/hostdev/patch_device_map.py \
  --kernel /tmp/hummingbird-hostdev-build-20260908.nxilLJ/resnet152-callable/mod-native.ptx \
  --bpf /tmp/hummingbird-hostdev-build-20260908.nxilLJ/hb_map-own.ptx \
  --output /tmp/hummingbird-hostdev-build-20260908.nxilLJ/resnet152-callable/mod-bpf.ptx
/usr/local/cuda-12.9/bin/ptxas -arch=sm_120 -O3 -v \
  /tmp/hummingbird-hostdev-build-20260908.nxilLJ/resnet152-callable/mod-bpf.ptx \
  -o /tmp/hummingbird-hostdev-build-20260908.nxilLJ/resnet152-callable/mod-bpf.cubin
```

Both exited zero; the complete BPF cubin is 439632 bytes. This is a build
result, not a performance comparison. Initial non-root and root client
attempts both exited 134 at `NvRmQuery` with ioctl errno 22, before device
execution. Raw logs are in `raw-first-real-20260908.08lUpN/`.
The loaded NVIDIA core lacked `nv_gpu_sched_gsp_control_complete`, whereas
the saved compatible core contained it. The previous lifecycle restored the
core using `modprobe nvidia` and the custom UVM using `insmod`.

Root's subsequent temporary compatible-core run successfully queried both
owned contexts and set their timeslices. The execution and restoration log,
not the assembly result, determines the eventual client outcome. The
original inline implementation remains a required comparison; a callable
native adapter alone must not be labeled the unchanged original algorithm.

The compatible-core client subsequently finished successfully at 20:49:57
PDT: 6000 foreground and 7718 background requests completed in the 60-second
timed interval (100.0 and 128.633333 requests/s). The existing client reports
zero maximum absolute output error for both models. It records 193330285
host JIT decisions, 3581 split launches and 2368 output-small launches.
These launch statistics are host-side counters, not a new device-call count
measurement. There is **no paired overhead estimate yet**.

`run-compatible-driver.sh` preserves the actual invocation and temporary
module lifecycle. It exited zero and records `RESTORATION_OK=1`. GDS/KV
loaders 4070246/4070247 both report `attached`; GDM and persistence services
are active, and the GPU returned to 0% / 1 MiB. The script's explicit PIDs
describe this invocation and must be refreshed before any future run.
Both earlier failed startup logs are retained alongside the successful log.
