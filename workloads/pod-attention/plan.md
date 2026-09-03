# POD-Attention: real device-BPF task selection

Status: complete official CPU build succeeded: all 21 fused CUDA compilation
units and the original FA extension (four CUDA TUs plus API) compiled and linked.
The resulting `fused_attn`/`flash_attn_og` shared libraries are 719,807,608 /
29,503,152 bytes. Selector/ABI and launch-bridge CPU checks, ten PTX-adapter
tests, eighteen preparation/pruning/partition tests and four host-link tests pass;
the numerical/benchmark/coordinator CPU tests also pass.
Final linked PTX extraction also passed all four planned TUs/six packets.
Two real FA-only preflights stopped at the v1 extra FP32 gate before any POD
operator, device selector or performance cell completed. Protocol v2 below is
fixed before any actual POD/BPF result. This is an operator port,
not a Sarathi/full-paper reproduction.

The first build exposed NVCC removing the constant length argument. The fix is
`--keep-device-functions`, not a one-argument adapter fallback. Rebuilt official
causal h128 fo10/fo11 PTX now has the real two-parameter declaration and call,
followed by volatile device output/engine reads used by the attention branch.
An isolated ABI compile also exercises NVCC's two-parameter address-space clone;
the adapter checks its declaration/call and still rejects a one-parameter clone.
The thin device-BPF loader and actual POD operator are not yet GPU-tested.
The scoped driver-launch bridge is compiled and its CPU driver-double tests
pass; it is not GPU-tested. Existing bpftime does not transfer the original
kernel's dynamic shared-memory opt-in to its newly loaded CUfunction.

## CPU preparation milestone (not GPU results)

The first final host link failed because NVCC emitted the same strong
`pod_device_selector` CPU stub in twenty objects. It is an unused 25-byte
`exit(1)` guard, not the GPU implementation. `host_link.py` checks zero host or
CUDA-registration references to that symbol, then localizes only its host
binding in private link copies. All originals remain untouched; every other
defined symbol and semantic relocation is checked unchanged, including all
CUDA-registration references (135 in each planned TU). The embedded `.nv_fatbin` sections
are compared directly byte-for-byte; no content identifiers are generated.
The explicit objects/sizes/disassemblies live in `build/link/attempt-01/inventory.json`.
The twenty originals and twenty copies each total 713,660,864 bytes; their
704,796,024 embedded GPU bytes compare identical. Host references to the stub
are zero in all twenty objects. Registration counts differ across unused
official variants (103/135/151/199); no variant's references were changed.
Four CPU fixture tests pass. The subsequent fused build reports `ninja: no work
to do` and links normally; none of its twenty large numerical TUs was rebuilt.
Only the separate original FA extension omits `--keep-device-functions`, since
it has no typed selector ABI. Both retain the original numerical optimization
flags, all official TUs, and sm_120 plus PTX; fused flags/order are unchanged.

`--keep-device-functions` also emits many unused helpers. All four official
planned TUs now have a real adapter path within the unchanged 67,108,864-byte
response limit. Values below are bytes; response sizes come from the actual
C++ adapter including generated device-BPF code, not an estimate.

| Causal h128 TU | Original PTX | After unused-function removal | Actual response packet(s) |
| --- | ---: | ---: | ---: |
| fo9 | 117,759,871 | 57,041,536 | 58,991,084 |
| split fo9 | 123,530,687 | 75,368,993 | 39,481,598 + 39,641,774 |
| fo11 | 108,647,880 | 48,814,358 | 50,550,084 |
| split fo11 | 118,773,218 | 67,207,085 | 35,484,089 + 35,313,116 |

Each TU retains all 135 original entries and all 128 actual two-argument
selector calls. The two split TUs still exceed the transport after pruning;
`ptx_partition.py` groups whole entries by code size, preserving their complete
union without duplicates and copying no instruction body differently. The fo9
split packets contain 73/62 entries and 66/62 calls; fo11 has 62/73 entries and
62/66 calls. All six real adapter responses preserve each call's argument pair.
This is not a smaller attention workload or a successful GPU execution.

Pruning keeps all module data/extern declarations and complete named-reference
closure, including address-taking; unknown or indirect reachable calls fail.
Partitioning is allowed only after stronger checks: in both actual split TUs,
all five retained helpers have visible bodies, unknown external helpers are
absent, all 30 module-global objects are unreferenced, and the eight referenced
shared objects are CTA-local. Function/entry address identity and module-level
function aliases/tables are absent. Any contrary evidence rejects partitioning.
Original PTX, linked extension and explicit function/packet inventories remain
preserved; no dirty bpftime code is changed. `prepare_ptx.py` completed these
checks against the linked fused extension in `build/ptx-runtime-01/inventory.json`:
all 540 original entries and 512 typed calls are retained, each packet's exact
representative is present in the linked symbols, and all six actual adapter
response sizes match the table above. The real loader lifecycle, launch bridge,
full-shape numerical correctness and device engagement are still unverified.

## Question and decision value

The existing-policy question is “您的机制是否能够实现许多现有策略”; revision
R2 asks which policy ideas are expressible. The paper's RQ4 is: “What is the
overhead of \\sys{}’s core mechanisms and observability capabilities?”
The specific uncertainty here is whether real eBPF-derived GPU code
can perform POD's SM-local work selection inside the actual attention kernel,
with correct results and useful operator performance. The existing three-system
results use host policy execution and cannot answer this device-side question.
This is supporting expressibility/mechanism evidence, not a claim that BPF
invented fusion or must outperform native CUDA. Success expands the measured
device-policy scope; failure bounds this operator/backend, not the entire thesis.

## Original artifact and compatibility

- [Author paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/03/POD-Attention-ASPLOS25.pdf),
  Sections 4.1–4.2 and Figure 14; local [PDF](../../docs/reference/2025-pod-attention.pdf).
- [Official source](https://github.com/microsoft/vattention/tree/71a0e91aa46ff8fa985bcca3327efe0ab9929a39/pod_attn),
  revision `71a0e91aa46ff8fa985bcca3327efe0ab9929a39`, fetched into ignored
  `deps/vattention` from the official archive; included CUTLASS headers retained.
- The source targets sm_80/sm_90 and rejects CC 12.0 in `fused_api.cpp` and
  `flash_api.cpp`. The port must explicitly build sm_120 with CUDA 12.9 and
  the prepared 575 driver; PyTorch 2.4/CUDA 12.4 is the author's environment,
  not a suitable assumed Blackwell runtime. No driver changes are required.
- `fused_fwd_launch_template.h:406` allocates counters using active SM count,
  but `fused_fwd_kernel.h:1453` indexes global counters after `%nsmid`.
  Query the identifier bound once, allocate `%nsmid + 2`, validate `%smid`,
  and use the same layout in all POD arms. NVIDIA explicitly says
  [%nsmid can exceed physical SM count](https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-nsmid).
- Correct the unchecked counter allocation/zeroing and unreleased workspace
  equally in native/BPF arms, with stream-safe lifetime and reset each launch.
  Record dynamic shared memory/register/occupancy limits: CC 12.0 has a
  [99 KiB per-block limit](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html#occupancy),
  not the sm_100 datacenter limit. Do not claim 2/4 resident CTAs from an A100 comment.

## Actual device hook and algorithm

The hook replaces only the leader-thread selection at
`pod_attn/fused_fwd_kernel.h:1444–1498`. Preserve the original proportional
integer rule (`FusedOp & 1`), SM-local atomic ticket, global operation-slot
atomic claim and one exhausted-operation fallback. The host tile/split
heuristics and the prefill/decode numerical kernels remain upstream code.

Use a CUDA-owned, per-CTA global context containing the counter pointer/bounds,
SM ID, prefill/decode slot limits and explicit `out_op/out_cta` fields.
The eBPF program reads the actual device state and performs its own atomic
claims. Only thread 0 calls it; the executor validates output bounds, publishes
the operation/slot to shared memory, then follows the existing barriers and
**uses those outputs to select the real attention branch**. Logical sub-CTA
mapping and tail-slot checks remain unchanged. Invalid decisions fail the run,
never silently fall back to the native selector.

The existing bpftime LLVM/NVPTX compiler supports atomic fetch-add, but its GPU
entry returns void and discards eBPF r0. Therefore this uses explicit device
output fields, not a claimed scalar return. Reuse its compiled PTX-pass/compiler
library; a local typed-call PTX adapter preserves the two context arguments and
replaces the named device selector. Existing generic entry probes discard this
decision and are insufficient. No host JIT, host callback or trace replay makes
the task choice, and no dirty bpftime checkout is edited. This does not by itself
establish verifier safety or transparent attachment to arbitrary binaries.

Both context-adapter arms preload the same POD-only launch bridge. The CUDA
control resolves the original registered function with `cudaGetFuncBySymbol`
and uses the driver launch path; BPF already uses that path in the existing
agent. At the final real CUfunction, the bridge checks the current device and
function's shared-memory limits, sets an opt-in covering that launch's actual
`sharedMemBytes`, and checks the readback before launching. Unsupported resource
requests or CUDA errors terminate the process, not a native fallback. Per-cell
bridge counts must cover every diagnostic, warmup and timed POD launch. The
inline and non-fused baselines keep their original runtime launches/opt-in.
The adapter comparison includes this host launch compatibility layer; it is
not an isolated estimate of a device call's instruction cost. Function/module
lifetime is the cell process lifetime; do not unload/reload modules in a cell.

## Matched comparison

| Arm | Role |
| --- | --- |
| Official FlashAttention serial | Original non-fused baseline |
| Official FlashAttention two streams | Strong non-fused overlap baseline |
| POD original inline CUDA selector | Original operator, common sm_120 safety port |
| POD CUDA selector through the explicit context | Matched adapter-cost control |
| POD device-BPF selector through the same context | Proposed policy execution |

Use the official `true_fused_attn_with_kvcache` / `flash_attn_with_kvcache`
operators and Figure-14 shapes: Llama-3-8B and Yi-6B attention (32 query heads,
8/4 KV heads, head dimension 128, FP16), prefill batch 1 and length 8192,
decode length 1, KV length 8192, decode batches 32/64/96/128/192. These are real
attention operators with synthetic Q/K/V, not model-accuracy or server results.
Keep `fused_params=15` upstream automatic tile choice identical in POD arms;
do not select each arm's fastest tile after seeing final measurements.

Primary: complete prefill+decode operator CUDA-event latency, including required
workspace resets and split combines; secondary synchronized host-wall latency.
For two streams use a common start event and join both completion events before
the end event. Reuse official numerical kernels, not its exception-swallowing
`None/-1` benchmark path. Report native/BPF paired ratios separately from
fusion-vs-nonfusion gains. Warm up 10 times; the per-cell estimator is the
arithmetic mean of 100 unfiltered, unrounded timed operator latencies. Retain
every observation. Pair at the cell/block level, not the individual launch.
Five randomized paired blocks use seed 20260903; input seed is 20260904 and
paired percentile-bootstrap seed is 20260905 (10,000 draws). No dropped adverse cells.

Correctness under `pod-fp16-upstream-match-v2`: full finite outputs against
separate original FA, with the unchanged hard `atol=1e-3, rtol=1e-5` check for
every arm. Full prefill/decode FA-versus-FP32 scans characterize cross-precision
error; their finite/shape/mask checks remain hard failures, but exceeding the
same reporting threshold is not a cross-precision pass/fail gate. Real
diagnostic launches retain each CTA's SM,
ticket, operation and slot, verify each actual claim and exactly-once logical
work, including exhaustion/tails; do not require identical nondeterministic
SM assignments across runs. Confirm device-BPF-written fields and PTX call
feed the branch. Disable detailed tracing equally for performance, retaining
numerical checks outside event timing. Oracle changes require independent
numerical evidence, an explicit protocol version and a fresh run of every
affected cell; earlier failures are never relabeled.

## Explicit numerical protocol revision, before POD/BPF results

V1 additionally applied that absolute gate to original FA versus a full-FP32
oracle. Both `raw/preflight-575-01` and `raw/preflight-575-02` remain failed and
cannot enter formal statistics. In 02, the complete prefill output had one
exceedance among 33,554,432 elements: max 0.0013279915, MAE 0.00000753793,
RMS 0.0000131618, at `[0,1,19,7]`; decode's oracle was not reached under v1.
The saved two-key row's independent CPU-FP64 value is 3.2291408396217083,
versus FA 3.23046875 and nearest FP16 3.228515625. Thus the failure is not just
unavoidable final-output quantization; FP32 and FP64 agree within 4.28e-7 over
that row. A source-consistency model of unnormalized exp → half P → PV /
unquantized denominator → half O matches all 128 actual components; final-half
rounding alone matches 79. This is not a hardware-step rounding proof or an
all-shape accuracy claim. Code: `bench.recompute_saved_fp64`; data and generated
`cpu-fp64-report.json`: `raw/preflight-575-02/block-01-official_streams/fp32-failure-prefill/`.

The official `tests/attn_sweep.py:82,84` gate compares two FP16 implementations,
not FA to full FP32. V2 retains that gate unchanged, without a BPF exception,
and records full cross-precision count/exceedances/max/mean/RMS for both phases
of every shape. Each excess saves its worst real query/head, effective Q/K/V,
and actual/reference rows in a unique model/batch/phase directory for CPU-FP64
recalculation. Completed reference scans are checkpointed before launching the
tested operator; a later hard FA↔arm failure saves corresponding real rows too.
No observed-max multiplier or selected favorable error threshold is used.
All new manifest/execution/operator cells carry `numeric_protocol`; all five
arms and ten shapes must be rerun, and v1 data cannot satisfy v2 admission.

## Minimal implementation and execution

Only this directory: a replayable `pod-compat.patch`, shared selector context,
`selector.bpf.c`, CUDA selector adapter, small PTX compiler/call adapter,
`prepare.sh`/`Makefile`, and an adapter around the official attention benchmark.
Do not add a second control framework, replace attention with a toy kernel,
modify canonical paper docs, or rebuild bpftime in place.

Build: `taskset -c 12-15 make -j2 build` (CUDA 12.9; reuse the existing
`../moe-infinity/.venv/bin/python`, currently PyTorch 2.13.0+cu129, without
installing into it; isolated output in `build/python` and `build/torch`).
After the complete official extension links, run `taskset -c 12
python3 prepare_ptx.py --extension build/python/fused_attn.cpython-312-x86_64-linux-gnu.so
--output-dir build/ptx-runtime-01`. The helper checks the four precise official
causal fo9/fo11 × split/non-split input objects and linked symbols, preserves
original/reduced PTX and per-function inventories, and exercises the real
adapter's full response buffer before producing exact representative names.
Use its `device/` directory with existing `BPFTIME_CUDA_LATE_PTX_DIR`,
`BPFTIME_CUDA_DEFER_PTX_EXTRACTION=1`, `BPFTIME_CUDA_DISABLE_CUOBJDUMP=1` and
`BPFTIME_PTXPASS_LIBRARIES=$PWD/build/libpod_ptx_adapter.so`. One exact kernel per
transport packet replaces every typed selector call in that packet; the existing
agent registers all its entry points. Every real shape must still pass device engine/atomic
claim/exactly-once/bridge checks. The full original linked extension and all
hard-referenced official templates remain intact; no unused host dispatch
branch is replaced with a fallback.
Under the coordinator's exclusive GPU lease, each client is
`../moe-infinity/.venv/bin/python bench.py --arm ARM --block BLOCK --output NEW.json`;
add `--preflight` for the one-shape real check. Run all five arms in each block's
predeclared seeded order. CUDA-control clients require
`LD_PRELOAD=$PWD/build/libpod_launch_bridge.so POD_LAUNCH_BRIDGE=cuda`;
BPF clients use the same bridge with `POD_LAUNCH_BRIDGE=bpf`, followed by the
existing `libbpftime-agent.so` in `LD_PRELOAD`, the local typed PTX pass and an
owned `BPFTIME_GLOBAL_SHM_NAME=pod_attention_...`. A separate preloaded syscall
server runs `build/pod-loader build/selector.bpf.o EXACT_KERNEL_NAMES.txt` and
remains alive until the BPF client exits. Exact kernel names must come from the
linked official extension, not a wildcard attach or assumed symbol. Private
device-loader lifecycle is not yet run or validated. Set
`SPDLOG_LEVEL=warn` to avoid recording vendor internal cache identifiers.
`run_study.py` now supplies this thin coordination using the existing GPU/struct-ops
lease paths and safety/telemetry helpers. `python3 run_study.py preflight --output
NEW_DIR --ptx build/ptx-runtime-01` runs five one-shape cells, with three timed
samples excluded from formal analysis. The coordinator must retain CPU 16 for
telemetry; clients use CPUs 8–15. All CPU compilation must have ended first.
Injected clients now launch as `taskset -c 8-15 /usr/bin/env LD_PRELOAD=...`
followed by the unchanged Python operator command. The wrapper's environment
omits `LD_PRELOAD`; both launch and target environments are retained and checked
offline. Previously the agent was preloaded into `taskset`, whose intercepted
`__libc_start_main` initialized the agent before taskset could pin CPUs or exec
Python. In `raw/preflight-575-03`, only `official_streams` completed; the BPF
client was observed still named `taskset` after SHM initialization, with no
operator result. The subsequent host reboot at 09:04 UTC interrupted the run;
the reboot's cause is unknown. This is not a demonstrated client timeout or
BPF operator failure. The attempt remains incomplete; the corrected launch
needs a fresh full five-arm preflight.
`python3 run_study.py full --output NEW_DIR --ptx build/ptx-runtime-01 --preflight
PASSED_DIR` requires that complete five-arm preflight with unchanged runtime
files. Each cell checks the campaign's original file inventory; a live loader
keeps its private SHM until the CUDA client exits, then orderly detaches and
removes only its newly created exact segment. Every failure/cleanup check is
retained. The two v1 coordinated invocations reached only the original FA
reference failure; neither is a completed five-arm preflight.
Real preflight uses the actual two
attention outputs, one Figure-14 shape and all five arms, plus exhaustion tests;
CPU selector tests/build success alone are not preflight. Retain source versions,
commands, build sizes, configuration, raw timings/decisions, errors and final
cleanup. Stop on CUDA errors; only owned experiment processes may be cleaned up.
Expected cost: a potentially substantial one-time template build, then tens of
minutes for correctness and the matched sweep; notify the coordinator before
heavy compilation or any GPU work. No runtime deadline is promised before build.

Positive: correct real device-selected work with quantified matched overhead.
Neutral/negative: retain matching or worse BPF latency and any lost fusion benefit.
Inconclusive: unsupported shared-memory/ABI/backend behavior or failed numerics;
do not call this reproduction or convert a setup milestone into a result.
