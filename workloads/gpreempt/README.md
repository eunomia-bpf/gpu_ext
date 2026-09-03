# GPreempt original-policy 575 / sm_120 compatibility build

Upstream: <https://github.com/thustorage/GPreempt>, pinned to `249ee3e`.
This directory preserves the original clients, executors, workload definitions,
1,000,000 µs LC / 1 µs BE timeslices, hint daemon, blocking kernel, CUDA graphs,
and GDRCopy signaling. Compilation is not an experimental reproduction.
The [source audit](../../docs/driver_docs/sched/gpreempt-analysis/feasibility-575-20260902.md)
defines the remaining driver, model, and comparison work.

## Build and CPU tests

Run from this directory, with no active timed experiment:

```bash
make build JOBS=4 CPUSET=8-15
make test-cpu
```

`prepare.sh` makes a separate ignored clone at `deps/upstream`; it never patches
the read-only survey cache. It applies `compatibility.patch` using a patch check,
retains original submodule revisions, and switches only local clone transport
from SSH to HTTPS. No system package installation or GPU execution occurs.
Targets are the original CMake `gpreempt`, `baseclient`, `gpreemptclient`,
`gpreemptclient_wo`, `block_cubin`, and `test-basic`, under `build/ninja`.
The mock transport test wraps `open` and `ioctl`, so it cannot contact a GPU.

Compatibility changes are limited to configurable build architectures,
out-of-tree block-module lookup, initialized/checked ioctl arguments, the narrow
575 timeslice transport below, and correct finite-smoke allocation/GDR cleanup.
The first configure incorrectly accepted failing glog platform probes (`io.h`,
`REG_PC`); evidence remains in `build/CMakeFiles/CMakeConfigureLog.yaml`.
The actual cause was this host's `/usr/bin/c++` returning exit status 0 after a
deliberate `#error`, while `/usr/bin/g++-13` correctly returned 1. The supported
build explicitly selects GCC/G++ 13, including NVCC's host compiler, and uses
Ninja rather than changing glog or the system compiler. Use `FRESH=--fresh`
once when replacing a configure cache produced by the broken compiler.

The first successful CPU-only full build on 2026-09-02 produced `baseclient`
(1,176,624 bytes), `gpreemptclient` (1,191,504), `gpreemptclient_wo` (1,185,032),
`test-basic` (1,028,960), `libgpreempt.so` (16,496), and `block.cubin` (7,384).
`cuobjdump --list-elf` identified `block.sm_120.cubin`; `ldd gpreemptclient`
resolved every dependency. The mock transport test passed 13 ioctl calls,
including syscall failure, NV status failure, malformed query responses, and
both original timeslices. None of these checks initialized CUDA.

## Frozen userspace/driver ABI

- Both operations use `OP_QUERY = 0xc0204660`, a 32-byte `NVOS54_PARAMETERS`.
- Query: `flags=0`, `cmd=0`, input `hClient=creator TID`, zero `hObject`,
  `params=&NvChannels`, `paramsSize=sizeof(NvChannels)`. The driver must return
  exactly one owned GR TSG and 1–64 channels; ambiguity is a failure, not a
  reason to select the first match.
- Timeslice: `flags=0x00010001` (version 1 / operation 1),
  `cmd=0xa06c0103`, query-returned `hClient/hObject`, `paramsSize=8`, and the
  original 64-bit timeslice value. The driver rechecks process ownership under
  its API/GPU locks before allowing only this operation.
- General `OP_CONTROL` remains separate and preserves the driver's ordinary
  security checks; no global `Nv04ControlWithSecInfo` bypass is introduced.
  All syscall and returned NV status failures propagate to the caller.

This is an explicit compatibility transport change, not a scheduling-policy
change. The driver port is developed separately; building userspace does not
mean that the running kernel implements this ABI.

## GDRCopy dependency: build, do not install/load automatically

The current dependency is official GDRCopy v2.5.2 at revision `c91ad9f`, in
the independent `deps/gdrcopy-2.5.2` checkout, without vendor edits. The
[official changelog](https://github.com/NVIDIA/gdrcopy/blob/v2.5.2/CHANGELOG.md)
adds Blackwell support in 2.5.1 and fixes the Linux 6.15 conftest in 2.5.2.
Rebuild only during a coordinated non-measurement window:

```bash
# Only where deps/gdrcopy-2.5.2 does not yet exist:
git clone --branch v2.5.2 --depth 1 https://github.com/NVIDIA/gdrcopy.git deps/gdrcopy-2.5.2
make gdrcopy-driver JOBS=4 CPUSET=8-15
taskset -c 8-15 make -C deps/gdrcopy-2.5.2 exes -j2 CUDA=/usr/local/cuda-12.9 \
  CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 NVCCFLAGS='-arch=sm_120 -ccbin /usr/bin/g++-13'
```

The exact build uses CUDA 12.9, the 575 driver's `kernel-open/nvidia` headers,
Linux `6.15.11-061511-generic`, and GCC 14. Its unmodified conftest correctly
selected `HAVE_VM_FLAGS_SET=y`; no override is needed. The successful build
produced `gdrdrv.ko` (534,936 bytes), module/API version 2.5 with the 6.15.11
vermagic. The release revision, not that unchanged API version, identifies 2.5.2.
It has not been loaded by this workflow. Never load `nv-p2p-dummy.ko` (a link
stub), and do not use the upstream install/load script's device deletion or
world-writable defaults. Module loading and exact device-node ownership are
handled separately by the driver owner.

The executable build also completed without GPU execution: official
`gdrcopy_sanity` is 196,480 bytes, `gdrcopy_pplat` is 1,100,488 bytes and contains
an sm_120 cubin, and the private `libgdrapi.so.2.5` is 26,464 bytes. All builds
finished in the MoE campaign's existing cooldown windows. This is dependency readiness,
not evidence that pin/map works on this GPU. After a separately coordinated
module load, official `sanity -t basic_cumemalloc` and
`sanity -t data_validation_cumemalloc` are candidate bounded checks; the local
finite smoke below additionally tests the compatibility query/timeslice path.

The earlier v2.5 `bda1f60` checkout/build is retained at `deps/gdrcopy`.
Its conftest incorrectly selected `n`, causing duplicate-helper/read-only
`vm_flags` errors; the explicitly inspected `HAVE_VM_FLAGS_SET=y` workaround
built a 535,584-byte module. The new release supersedes that workaround without
altering the historical build or editing vendor source. Official support lists
data-center/professional RTX GPUs; it does not by itself establish that GeForce
RTX 5090 pin/map succeeds. The original finite GDRCopy path still must run.

## Finite smoke, only after the driver owner releases a GPU slot

```bash
python3 -B run_smoke.py --output raw/575-smoke-01
```

This acquires the shared GPU and struct-ops leases, checks the idle GPU, requires
the separately prepared `/dev/gdrdrv`, and runs the original `test-basic` target
with a 30-second bound. The compatibility smoke checks query/control errors,
a GDRCopy flag roundtrip, and correct unmap/unpin/free/context cleanup. It does
not launch the unbounded blocking kernel or claim effective timeslice/preemption
performance; actual driver actuation needs independent evidence. Every outcome,
including failure, retains `smoke.log` and `result.json`, and cleanup targets only
the owned process group. The runner never changes modules, devices, or services.
It starts the child with an explicit minimal environment, excluding inherited
preload/injection variables and resolving GDRCopy from the private build first.

## Model assets and fair cells still required

The official checkout contains only `model/makefile`, not ready model assets.
Its generation script uses patched TVM `513c2be0c3b853`, defaults to Inception
instead of config A's VGG/ResNet152, contains an additional `sm_80` target and a
hard-coded `get_source("hip")`, and executes `module.run()` to collect host
metadata. Consequently it is not a CPU-only ready-made model exporter. Model
generation needs explicit model selection, CUDA-source export, sm_120 targeting,
the patched TVM host metadata path, and a later authorized GPU validation slot.
Do not substitute a hand-written vector kernel and call it the model experiment.

The source preparation and explicit model exporter are now available:

```bash
make configure-tvm
uv venv --python /usr/bin/python3.10 deps/tvm-venv
uv pip install --python deps/tvm-venv/bin/python -r tvm-requirements.txt
# Heavy compilation: only with the experiment owner's cooldown coordination.
make build-tvm JOBS=4 CPUSET=8-15
python3 -B export_model.py --model vgg --plan
# The following commands acquire GPU leases and really execute CUDA. Wait for a slot.
python3 -B export_model.py --model vgg --output deps/upstream/model/vgg
python3 -B export_model.py --model resnet152 --output deps/upstream/model/resnet152
```

`prepare_tvm.sh` applied the original recorder patch successfully at the pinned
revision, with the three required source submodules. The CPU-only configure
completed with LLVM 14, GCC 13 and CUDA 12.9; the full TVM build and model exports
have not yet passed. The exporter has a 1,200-second process-group bound, keeps
failed exports, and never replaces an existing model directory. It chooses the
upstream script's VGG19/ResNet152 variants with 224×224 FP32 input, testing
parameters generated with seed 0, and the same nonconstant deterministic input
formula for every cell. These are TVM testing networks, not pretrained-accuracy
results or recovered copies of the authors' unavailable model binaries.

Each successful export will retain CUDA source, the sm_120 cubin, graph/actual
launch metadata, parameters, and the full 1,000-value isolated native TVM output
as `reference.f32`; that output is a numerical reference, not a performance
measurement. CUDA source export and model choice are explicit fixes to the
provided generation script. Runtime output agreement still must be checked in
the original standalone executor and every comparison cell.

Primary comparison remains upstream native `baseclient` (single context with
stream priorities), complete original `gpreemptclient` on the 575 port (two role
contexts), and the equivalent BPF policy on the same two-context topology.
`gpreemptclient_wo` is an optional timeslice-only ablation. Preserve original
config A's 60 seconds, 100 requests/s per role, 200 µs preprocessing and graphs;
initialize identical deterministic input in all three cells and verify outputs
against isolated native execution. Role-to-TSG and hint bridges for the BPF cell
are separate work, not implied by this compatibility build.

## Integrated BPF policy and common numerical instrumentation

`policy-bridge.patch` is an additional, explicit comparison patch after the
compatibility patch. Build it only outside timed GPU measurements:

```bash
make -C ../../extension -f gpreempt.mk bridge
make build-bridge JOBS=4 CPUSET=8-15
make test-cpu
deps/tvm-venv/bin/python -B test_model_export.py
```

The three client sources have passed CPU-only syntax checks against the real
bridge header. The bridge-integrated client binaries have not yet been rebuilt
or run. Once this patch is applied, all three DNN clients require the exported
full `reference.f32` and initialize exactly the input formula above, including
after context reinitialization. Every output is finite-checked and compared
elementwise with `atol=1e-6`, `rtol=1e-4`; `GPREEMPT_VALIDATION` records total and
timed checked requests separately. The original 10 warmups and 100 standalone
samples still precede the timed phase. Numerical checks are in the same
postprocess position in every arm and their overhead is included consistently.
This comparison patch intentionally covers the FP32 DNN cells; zero-output
graph/scientific workloads are rejected rather than labeled numerically valid.

Both full-policy arms use the same strong-linked bridge and original CUDA/GDR
actuators. Default `GPREEMPT_POLICY=original` runs the original C decisions and
the narrow compatibility timeslice ioctl. `GPREEMPT_POLICY=bpf` requires the
loader's unique `GPREEMPT_BPF_MAPS` directory and absolute `GPREEMPT_HINT_CODE`
bytecode path; missing engagement is fatal, and this arm skips the original C
`set_priority`. Context begin/register/end surrounds each new role context,
while the original initial primary-context warmup remains unchanged. Native
`baseclient` retains its single-context stream priorities and rejects a BPF
label. `_wo` remains a distinct timeslice-only ablation, not the full policy.

For full GPreempt, PREPROCESS still resets the GDR flag and reserves the hint
at preprocessing minus 100 µs; DUE uses the same `system_clock` domain and strict
`>` comparison. The selected BLOCK action enqueues the same two blocking kernels.
INFER still enqueues the model before releasing the flag. Role values outside
0/1 and preprocessing of 100 µs or less are explicitly rejected in all three
clients. The source-level bridge and CPU tests establish wiring and decision
agreement, not GPU actuation or scheduling performance.

`measurement.patch`, applied by the same `--bridge` preparation, exports the
existing analyzer's completed count and every request's six-stage duration only
while generating the final report. It does not add work inside the timed
recording path. Percentiles from these samples are source-native six-stage
service latency, not arrival-to-completion latency. A 100 requests/s offered
load does not imply exactly 6,000 completed requests in the 60-second window.
