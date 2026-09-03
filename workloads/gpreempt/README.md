# GPreempt original-policy 575 / sm_120 compatibility build

## Completed comparison — 2026-09-03

All **five paired blocks / 15 cells** of native baseline, original-C policy,
and actual BPF policy passed independent raw-output and engagement checks.
LC service-stage p99 medians are **1.414351 / 1.415130 / 1.419397 ms**,
respectively. The BPF/original-C paired overhead is **0.258%** (95% interval
0.128–0.374%); this is successful scoped policy implementation with a small
measured cost, not a performance win. Both policy arms use the explicit
**host-mapped flag compatibility transport**, not original GDRCopy.
See the [complete results and replay command](results-575-host-mapped-20260903.md).
The preparation history below is retained; its earlier build-only milestones
do not describe the current completion state.

## New contention / full-response study

The [fixed follow-on plan](load-study-plan.md) tests LC 100 requests/s with
BE 100, 200, or closed-loop continuous supply. This explicitly changes the old
newest-only generator to a common-phase FIFO schedule. It measures scheduled
arrival to synchronized, numerically verified output, not just six service
stages; all unfinished backlog and window-crossing completions remain visible.
Build and run separately from the completed config-A experiment:

```bash
make build-load-study JOBS=4 CPUSET=8-15
make test-load-study CPUSET=8-15
python3 -B test_plot_load_study.py
python3 -B run_load_study.py full --plan
# Actual runs require an idle GPU and the existing 575 scheduling port.
sudo -n python3 -B run_load_study.py preflight --output raw/load-study-preflight-01
sudo -n python3 -B run_load_study.py full --output raw/load-study-full-01
```

Use new output directories for every attempt; never overwrite or silently retry.
Preflight is three 10-second cells and is excluded from the 45-cell full study.
The runner acquires the existing GPU/struct-ops leases and never changes drivers
or services. It uses `build/load-study`; `build/ninja` and the old runner remain
untouched. `GPREEMPT_LOAD_STUDY` defaults to OFF for ordinary builds.

The new build explicitly binds worker CUDA contexts, checks graph/stream/sync
errors in all three arms, records actual native stream priorities, and removes
the native client's overwritten extra stream. These are scoped correctness and
measurement changes, not a new GPreempt policy. C and BPF keep identical policy
inputs and execution order, including the existing host-mapped transport.
The standalone audit and plotting tools consume raw reports; a successful build
or preflight alone must not be reported as completed performance measurements.

## Source and preparation history

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
The driver owner loaded it separately at 2026-09-03 00:37:22 UTC after loading
the `e3bb2938` 575 compatibility driver. The Makefile still never loads it.
Never load `nv-p2p-dummy.ko` (a link
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
RTX 5090 pin/map succeeds. The original finite GDRCopy path has now failed at
pinning: attempt `raw/575-gdr-context-smoke-02/` accepted the query/timeslice
request, but `nvidia_p2p_get_pages` returned `-22`. Post-cleanup state was clean.
This is not a successful GDRCopy or scheduling reproduction. Attempt 01 was
rejected before execution because module reload had reset the 400 W limit to
575 W; the limit was restored before attempt 02.

The independent official `basic_cumemalloc` test subsequently failed in
`raw/575-official-gdr-basic-01/`: CUDA's GPUDirect-RDMA support attribute was
false, and the unmodified test reported that the GPU does not support GPUDirect
RDMA. Thus the original GDR actuator is unavailable on this device, not merely
unbuilt or lacking device-node permissions. No capability check was bypassed.

## Finite smoke, only after the driver owner releases a GPU slot

```bash
python3 -B run_smoke.py --output raw/575-smoke-01
python3 -B run_smoke.py --case basic_cumemalloc --output raw/575-official-gdr-basic-01
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
GDRCopy diagnostic logging is enabled. Optional official `basic_cumemalloc`
and `data_validation_cumemalloc` cases use the same bounded wrapper; a waived
test is explicitly not accepted as a pass, even if the upstream suite exits 0.

An explicitly different transport canary, `make build-host-flag` followed by
`run_smoke.py --case host_flag`, completed 64 exact CPU-write/GPU-poll roundtrips
in `raw/575-host-flag-smoke-01/`, with clean cleanup. It uses mapped pinned host
memory and a device-side polling deadline, not GDR-mapped GPU memory. This
establishes a possible compatibility route, not original GPreempt reproduction
or scheduling performance. The full clients still default to original GDR.

## Explicit host-mapped flag compatibility arm

`flag-transport.patch` extends the integrated full client with an opt-in
`--flag-transport host_mapped`. **This is not original GDRCopy replication.**
The default remains `gdr`, and an unsupported/failed GDR pin never switches
transports automatically. Both original-C and BPF policy arms must select the
same transport; the native single-context baseline does not use a flag transport.

The compatibility option allocates portable, device-mapped pinned host memory,
sets `CU_CTX_MAP_HOST` on both role contexts, and obtains the LC context's device
pointer. CPU resets/releases use aligned release stores, while the unmodified
`gpu_block` kernel still polls its volatile pointer. Two role contexts, the
original 1,000,000/1 µs timeslices, preprocessing-minus-100 µs hint deadline,
two blocking launches, CUDA graph, and model-enqueue-before-release ordering
are unchanged. Neither BPF nor native policy chooses a different actuator.

`flag_transport.h` reports the actual transport and owns the exact allocation
base/mapping. For GDR it retains pin/map semantics, corrects allocation-base and
mapping-offset bookkeeping, and uses the official `gdr_copy_to_mapping` helper
for mapped-memory write barriers. After the original benchmark joins its worker
and daemon threads, cleanup releases every used flag, records completion events
on the tracked LC streams, and polls for at most five seconds. It only frees
flag storage after those events complete; an error/timeout fails the cell and
leaves in-flight storage for the outer bounded process cleanup. Successful
cleanup is explicitly reported and required by the runner.

```bash
make build-bridge JOBS=2 CPUSET=8-15
make test-cpu
python3 -B run_three_way.py --plan --flag-transport host_mapped
# GPU execution only in a coordinator-released slot, after driver validation:
sudo -n env -i PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin LANG=C.UTF-8 \
  /usr/bin/python3 -B run_three_way.py --output raw/575-host-mapped-three-way-01 \
    --blocks 5 --flag-transport host_mapped
```

The runner records `comparison_variant=host_mapped_compatibility` and actual
transport in each policy cell, rejects missing/mismatched readiness or cleanup
records, and cannot combine two different transports into one valid pair.
Unlike GDR policy cells, host-mapped cells do not require `/dev/gdrdrv`; no
runner changes any module, device node, or service. Native command lines and
config A's 60-second workload remain unchanged. Failed original GDR attempts
listed above remain retained and are not relabeled as compatibility passes.

CPU-only validation: the incrementally rebuilt full client linked successfully;
49 fake CUDA/GDR lifecycle checks passed without linking or calling a GPU
runtime. Tests include exact allocation/free bases, GDR pin failure without
fallback, portable mapping flags, aligned flag stores, bounded incomplete-event
cleanup without premature free, and idempotent completed cleanup. The 14 runner
tests also pass, including transport mismatch and mixed-pair rejection. These
are build/component results, distinct from the GPU canary below.

The full **original-C policy with host-mapped flags** subsequently passed the
original 60-second config-A correctness canary in
[`raw/575-host-mapped-original-canary-01/result.json`](raw/575-host-mapped-original-canary-01/result.json).
Both models completed 6,000 timed and 110 warmup/calibration requests, checking
every one of their 1,000 outputs with maximum absolute error zero. Reset, hint,
block, and release each executed 6,000 times; bridge errors were zero, and the
host-mapped allocation reported successful cleanup. Post-run UVM references,
compute clients, struct-ops state, and new Xids were all clear. Concurrent CPU
builds were permitted, so this result is explicitly **correctness-only**, not a
formal paired performance measurement. The result, full client log (including
the original six-stage samples), and telemetry are retained. It does not turn
the failed original-GDR attempts into passes, and does not validate the BPF arm.

The BPF comparison additionally requires the runtime timeslice-control hook:
both LC and BE must report at least one `control_override`, with exact agreement
between total and per-role counters and zero setter errors. This prevents the
old init-only implementation from passing when CUDA subsequently writes its
2048 µs default. These are accepted BPF **request** counters, not independent
proof of firmware actuation; the coordinator's real GSP canary remains distinct.

The full five-block, 15-cell comparison subsequently completed with all raw
audits passing. BPF/original-C LC p99 has paired geometric ratio 1.002575
(95% interval [1.001278, 1.003740]): a small measured latency cost, not a BPF win
or a formal equivalence claim. Median BE throughput is 100 requests/s for all
three arms at the fixed offered rate. See
[the complete result and transport limitations](results-575-host-mapped-20260903.md).
This completes the explicit host-mapped compatibility comparison, not the
unsupported original GDRCopy transport.

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
completed with LLVM 14, GCC 13 and CUDA 12.9. The full TVM build passed, producing
`libtvm.so` (86,473,272 bytes) and `libtvm_runtime.so` (4,709,720 bytes); importing
the pinned runtime reports `0.19.dev0` with CUDA enabled. Both model exports
passed with actual CUDA execution: VGG19 recorded 55 kernels, ResNet152 recorded
307, and each retained a finite 1,000-value native reference. The first VGG
attempt failed before model creation because `tvm.relay.testing` also imports
pytest; the dependency was added explicitly and the failed attempt is retained.
Successful export records are `raw/model-vgg-export-02/` and
`raw/model-resnet152-export-01/`. The exporter has a 1,200-second process-group bound, keeps
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

The original native executor has now passed a complete config-A correctness
canary in `raw/575-native-executor-canary-01/`: both VGG19 and ResNet152 completed
6,000 timed requests plus 110 warmup/calibration requests, with all 1,000 output
values checked each time and maximum absolute error zero. Post-run UVM,
struct-ops and compute-client state was clean, with no new Xid. Other CPU builds
were permitted during this canary, so its retained latency samples are diagnostic
and are not a paired comparative performance result. Original/BPF policy arms
still require their own numeric checks and timed comparison.

Primary comparison remains upstream native `baseclient` (single context with
stream priorities), complete original `gpreemptclient` on the 575 port (two role
contexts), and the equivalent BPF policy on the same two-context topology.
`gpreemptclient_wo` is an optional timeslice-only ablation. Preserve original
config A's 60 seconds, 100 requests/s per role, 200 µs preprocessing and graphs;
initialize identical deterministic input in all three cells and verify outputs
against isolated native execution. Role-to-TSG and hint bridges were separate
from the initial compatibility build; the integrated implementation and completed
comparison are documented below and in the result report above.

## Integrated BPF policy and common numerical instrumentation

`policy-bridge.patch` is an additional, explicit comparison patch after the
compatibility patch. Build it only outside timed GPU measurements:

```bash
make -C ../../extension -f gpreempt.mk bridge
make build-bridge JOBS=4 CPUSET=8-15
make test-cpu
deps/tvm-venv/bin/python -B test_model_export.py
```

The three client sources passed CPU-only syntax checks against the real bridge
header, and all bridge-integrated clients plus the measurement analyzer rebuilt
successfully in a coordinated cooldown window. `gpreemptclient` resolves the
strong bridge library and its begin/register/end, hint and backend symbols;
the private GDRCopy 2.5.2 library resolves when selected in the runtime environment.
That rebuild alone did not establish GPU execution; the later full campaign
has now passed. All three DNN clients require the exported
full `reference.f32` and initialize exactly the input formula above, including
after context reinitialization. Every output is finite-checked and compared
elementwise with `atol=1e-6`, `rtol=1e-4`; `GPREEMPT_VALIDATION` records total and
timed checked requests separately. The original 10 warmups and 100 standalone
samples still precede the timed phase. Numerical checks are in the same
postprocess position in every arm and their overhead is included consistently.
This comparison patch intentionally covers the FP32 DNN cells; zero-output
graph/scientific workloads are rejected rather than labeled numerically valid.

Both full-policy arms use the same strong-linked bridge and CUDA/flag actuators
(GDR by default; host-mapped only by explicit CLI selection). Default
`GPREEMPT_POLICY=original` runs the original C decisions and
the narrow compatibility timeslice ioctl. `GPREEMPT_POLICY=bpf` requires the
loader's unique `GPREEMPT_BPF_MAPS` directory and absolute `GPREEMPT_HINT_CODE`
bytecode path; missing engagement is fatal, and this arm skips the original C
`set_priority`. Context begin/register/end surrounds each new role context,
while the original initial primary-context warmup remains unchanged. Native
`baseclient` retains its single-context stream priorities and rejects a BPF
label. `_wo` remains a distinct timeslice-only ablation, not the full policy.

For full GPreempt, PREPROCESS still resets the selected flag and reserves the hint
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

## Five paired 60-second comparison blocks

```bash
python3 -B test_three_way.py
python3 -B run_three_way.py --plan
# Only after the coordinator releases the GPU and verifies driver/GDR canaries:
sudo -n env -i PATH=/usr/local/cuda-12.9/bin:/usr/bin:/bin LANG=C.UTF-8 \
  /usr/bin/python3 -B run_three_way.py --output raw/575-three-way-01 --blocks 5
```

The runner preserves config A, uses five randomized distinct permutations of
the three arms (position counts differ by at most one), and pauses 10 seconds
between cells. Each process has a 240-second bound including original warmup,
standalone measurement, initialization, and the 60-second measured interval.
All arms run with the same root privilege and a minimal environment because the
BPF control maps remain private/root-only; existing shared lease files are opened
without `O_CREAT`, avoiding sticky-`/tmp` ownership surprises. No module, service,
existing pin, or device permission is changed. An optional `--gdrcopy-dir` selects
the private dependency tree, defaulting to the separately built 2.5.2 checkout.

Original DISB periodic loads are newest-only, not unbounded queues
(`deps/upstream/third_party/disb/src/load.cpp:143`). They skip stale slots when
the following slot is already in the past and admit no new slot at or after the
cutoff; the last admitted request can finish after that cutoff. The original
coordinator also phase-offsets the two equal-rate, load-priority-zero tasks using
their measured standalone latency. Arrival/drop counts are not instrumented;
results explicitly leave them unknown rather than calling `6000-completed`
the drop count. Completed requests, service-latency samples, original 60-second
throughput, and whole-process wall time are reported separately.

Every cell must match all 110 original untimed requests plus every completed
timed request against the complete native numerical reference. At least 100
real samples per role are required to report p99. BPF additionally requires two
distinct role/TSG registrations, exact two-context scope/setter/allocation/destroy
counters, real JIT readiness, nonzero full hint/block/release decisions, and zero
policy errors. A clean loader exit or host-shadow counter alone is insufficient.
All raw request samples, telemetry, cell outcomes and partial/failed blocks are
retained; only five complete paired blocks set `formal_5_block_complete=true`.

`analyze_three_way.py CAMPAIGN_DIRECTORY --output NEW_RESULT.json` independently
re-parses retained request samples, checks numerical/engagement records,
commands, environments, telemetry and cleanup, and uses only complete paired
blocks. It reports geometric paired ratios and 95% block-bootstrap intervals.
The interval is omitted for one block; a near-one point estimate is not an
equivalence claim. The explicit `host_mapped` variant retains its compatibility
label and never becomes an original-GDR result through analysis. Config A's
100 requests/s per role also caps observed throughput: near 100 requests/s on
this RTX 5090 workload is not evidence of a saturated-throughput advantage.

The nine CPU-only runner tests include count/tolerance/zero-engagement rejection,
balanced ordering, partial-block rejection and actual cleanup of an owned CPU
orphan after its group leader exits. None is a GPU performance result.
