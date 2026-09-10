# XSched Level-2 sm_120 source checkpoint

Current status (2026-09-10): the completed 15-cell NVBit-actuator
policy-port comparison is reported in
`../raw/level2-device-policy-pair-20260909.buBjns/README.md`. The canonical
tool source was reconciled to the tested per-launch delivery order and rebuilt
as a host component in
`../raw/canonical-tool-build-20260909.MnNeOe/README.md`; that rebuild is not
a new timing measurement. The original cuXtra native SASS route is now
complete: the [matched multi-arm campaign](../raw/level2-native-route-20260910-082400/README.md)
ran five blocks / 20 cells against Level-1 and baseline using the
nine-patch reproducible source build.

Sibling lane handoff (MoE): the MoE adaptive byte-governor campaign is
complete and independently audited at
[results-adaptive-prefetch-575-20260910.md](../../moe-infinity/results-adaptive-prefetch-575-20260910.md)
with a no-GPU audit entrypoint
`scripts/artifact/reanalyze_adaptive.py` (AUDIT PASSED) — see its ARTIFACT.md
rows for the artifact evaluation boundary.


## Historical 2026-09-08 source checkpoint

Status: the device BPF guardian, sm_120 cubin and NVBit tool library are
**built**, but end-to-end Level-2 execution is **not measured**. The existing
Level-1 measurements and dependency checkout are unchanged. This is not a
completed reproduction of the original Level-2 system.

The 2026-09-08 component build compiled 49 eBPF instruction words; the
48-byte-context exporter accepted them with its existing GPU checks, emitted
device-callable PTX, and ptxas generated the sm_120 cubin. The tool library
linked successfully. See `../level2-build/build-20260908.kMcZTl/README.md`
and `branchless-barrier-components.log`. No verifier rule was disabled.
The native LDC adapter and HAL/end-to-end replay still require completion.

The metadata/main checkpoint now also compiles and runs against the three
real sm_120 cubins. It reads parameter windows of 0x380/0x1520 for the padded
probe, 0x380/0x18 for check, and 0x380/0x10 for restore, with width-aware
consumed-byte coverage. The process deliberately returns 1 at
`native encoding remains pending`; it does not emit runnable guardian arrays.
See `../level2-build/build-20260908.kMcZTl/native-metadata-main-build.log`
and `native-metadata-main-run.log`. Root mechanically integrated source
blocks already authored in local GLM's terminal-length output. Per-form
offset encoding, prefix fallthrough and restore transfer remain unfinished.

The source includes a device BPF guardian, shared native-C/BPF trampoline,
NVBit launch-context adapter, native probe/guardian/restore stubs, LDC
patcher, and an opt-in HAL patch. Component build commands are in
`../level2-build/`. The HAL patch passed a non-mutating `git apply --check`
against the local XSched source; it has not been applied to that checkout.

Integration status: the shared decision context is now a bounded 6 x u64
scalar snapshot (48 bytes, xsched_guardian_abi.h) materialized by the
trusted trampoline from the real device words; both the native-C decision
and the compiled eBPF program consume that same snapshot, and the BPF
program dereferences no device pointer. The reused SASS exporter is
adapted only through level2/bpf/bpf_to_ptx_ctx48.patch applied to a copy
under ../level2-build (.output/adapter), widening the strict PREVAIL
verify-time context from 8 to 48 bytes; granted size equals used size.
Still to build and exercise: the native sm_120 artifact path and
end-to-end replay. Both decision arms keep the same trusted actuator. No
host decision fallback or performance gain is claimed.

The compiled eBPF guardian is expressed branchless (straight-line u64
arithmetic, no conditional jumps): the strict PREVAIL GPU warp-uniform
branch verifier rejects branch predicates that vary per lane, so the same
policy as the native-C decision is computed with a nonzero mask via
`(x | -x) >> 63`, full-u64 equality via XOR, and decision selection via a
`(0 - c)` all-bits/zero mask. Each computed mask passes an empty volatile
asm compiler barrier so the BPF backend cannot re-fold the idiom back into
a lane-varying comparison; the emitted instructions stay plain
straight-line ALU ops. Decision semantics are unchanged, including the
`recorded==0` precedence over `recorded==kernel_idx` in the else-if chain;
no snapshot value is marked uniform and no verifier check is weakened or
disabled.

The stored HAL patch now includes `GuardianSM120`, its factory case and the
generated-array consumer, enabled with `XG_SM120_GENERATED_HEADER` (the exact
value is printed by `make hal-arrays` in `../level2-build`). The matched
NVBit actuator path returns before cuXtra-only Guardian/InstrMemAllocator
initialization. The real HAL/tool-actuator build now succeeds in an isolated
source/build/install directory after two small compile fixes; frozen Level-1
dependencies are unchanged. End-to-end execution remains unfinished. The native cuXtra
reference remains separate from the native-C/BPF pair, which uses the same
NVBit actuator. Existing Level-1 dependencies and measurements are unchanged.

The current HAL patch uses an 8192-slot mapped context pool, reserves slot
zero, and does not retire assigned slots. This is a bring-up limitation,
not support for unbounded long-running service. No binaries, cache payloads
or historical-result replacements are included in this checkpoint.

## 2026-09-09 source reconciliation

Canonical tool source: `tool/xsched_guard_tool.cu`. Tested reference source:
`tool/xsched_guard_tool_callback.cu`. The per-launch ordering validated in
`../raw/level2-perlaunch-enable-20260909.bOiYh5/README.md` is now reconciled
into the canonical tool: after target-launch instrumentation, the callback
sets the per-launch value, then re-enables instrumentation. The host-side
`xg_host_prepare` set remains the baseline fallback, and zero/disarm behavior
is retained. The completed 15-cell comparison is in
`../raw/level2-device-policy-pair-20260909.buBjns/README.md`; it covers this
same NVBit-actuator native/BPF policy-port scope, not the original cuXtra
native SASS route. This source reconciliation does not claim that a newly
built canonical binary was measured.

## 2026-09-09 runner path overrides

`run_tool_pair.py` now accepts `--xserver-native`, `--xserver-bpftime`,
`--hpf-bin`, `--guard-tool`, and `--hal-lib-dir` overrides; defaults are
unchanged. The selected paths are resolved once and used consistently for
required-file checks, worker/server environment setup, and protocol paths.
Baseline remains free of policy libraries. Template: replace the quoted
local paths and target symbol, then run under the existing root-held
GPU/experiment locks. The geometry below matches the recorded 50-task run:

```sh
python3 -B workloads/xsched/level2/run_tool_pair.py run \
    --workload '/path/to/priority_workload' \
    --xserver-native '/path/to/native/xserver' \
    --xserver-bpftime '/path/to/xserver-bpftime' \
    --hpf-bin '/path/to/bpftime_hpf.bin' \
    --guard-tool '/path/to/xsched_guard_tool.so' \
    --hal-lib-dir '/path/to/hal/lib' \
    --target-symbol '<exact symbol for the workload>' \
    --no-initial --repetitions 5 --reps 9511106 \
    --tasks 50 --blocks 340 --threads 256 \
    --output '/path/to/new-results-directory'
```

This runner change was validated CPU-only for syntax/help and direct
path/environment behavior; it adds no dry-run mode, preflight, build, or GPU
run.

## 2026-09-09 runner log decoding

`run_tool_pair.py` now decodes managed-process stdout/stderr as UTF-8 with
`backslashreplace`. This preserves later stderr lines if a diagnostic byte is
not valid UTF-8; it does not change JSON parsing, metrics, scheduling, or
cleanup. The separate native original cuXtra failure is a CUDA 700 failure in
that route, not a consequence of this log-decoding change.
