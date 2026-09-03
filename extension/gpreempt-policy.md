# GPReempt BPF policy arm: implemented, runtime persistence failure identified

This arm implements GPReempt decisions from upstream `249ee3e` rather
than calling a timeslice-only program a full port. The original CUDA executor,
two blocking-kernel launches, GDRCopy mapping/writes, and double-context topology
remain in the client. Kernel BPF chooses the context timeslice; a real bpftime
ubpf JIT chooses the preprocessing/hint/release actions. This is a combined
kernel-BPF + host-JIT port, not a claim that the driver hook alone implements
the full system.

The original native `baseclient` keeps its upstream single CUDA context and
stream priorities. Original full GPReempt and this BPF arm use two contexts.
A dual-context native control is a separate topology ablation, not a relabeling
of upstream's primary baseline.

## Dependencies and boundary

- The separate NVIDIA 575 driver repository needs the GSP propagation repair
  `363416c4`, owned GR query/narrow control transport `e3bb2938`, and per-runlist
  destruction identity / real GSP completion hook `e7d46fa5`. The persistent
  control callback additionally needs `849ea75d`. Building
  these commits is not a hardware validation. No module is installed or loaded
  by this implementation or its CPU tests.
- `workloads/gpreempt/policy-bridge.patch` owns integration with the original
  clients. All compared clients strongly link the bridge. In BPF mode the
  original C `set_priority` **must be skipped**, or the experiment is invalid.
- The existing prebuilt bpftime, libbpf, and bpftool archives are reused.
  The standalone makefile does not rebuild third-party dependencies.
- Scope is one owned process on one GPU, host PID namespace, two newly-created
  role contexts, one GR TSG per role. Multiple matching GR TSGs fail closed;
  do not silently choose one. The driver `task_init` identity lacks a GPU
  identifier, so this implementation does not claim multi-GPU correctness.

## Context/TSG correspondence

Before `cuCtxCreate`, `gpreempt_ctx_begin(role)` invokes an exported marker.
Its uprobe places a role in a bounded map keyed by TGID/TID. The bridge checks
that this record exists, is clean, and has the requested role before allowing
the caller to proceed. Missing attachment is a fatal error, not a native-policy
fallback. Nested scopes, unknown roles, and allocation/map failures are errors.

Inside that scope, an ioctl-entry probe captures only an A06C TSG allocation.
It supports the source-confirmed NVOS21/NVOS64 layouts (32/48 bytes, returned
object offset 8, status offsets 28/40) and the transfer envelope number 211.
The actual `task_init` BPF callback sees the engine already assigned by 575:
**RM GR0..GR7 are 1..8**, unlike the misleading old extension engine aliases.
Only these GR engines receive LC 1,000,000 us or BE 1 us. Other engines preserve
native settings; engine zero is rejected. The current 575 minimum-timeslice HAL
returns zero, so the validator does not clamp a requested 1 us to a larger
minimum. Actual GSP acceptance was observed, but CUDA subsequently reset both
roles to 2,048 us before execution; init-only actuation is not yet equivalent.

The ioctl-return probe publishes a handle record only when both the syscall and
NV status succeed and the returned handle is nonzero. The owned driver query
returns the context's actual `hClient/hTsg`; a second marker joins these handles
back to the captured GR allocation and role. `gpreempt_ctx_register` checks the
kernel record, creator thread, engine, exactly one GR initialization, CUDA
context identity, and requested timeslice. `gpreempt_ctx_end` verifies that the
scope was removed. TSG state uses `(runlist_id, tsg_id)` because the driver
allocates numeric grpIDs in per-runlist CHID managers. GR-only destruction
uses the appended actual runlist/engine identity and removes TSG/handle records.
CE bind/destruction must not alias GR records with the same numeric grpID.
Bounded maps contain
64 scopes/allocations and 128 TSG/handle records.

`bind_shadow_match` is explicitly a check of host bookkeeping. Neither a
successful setter nor this counter proves that the hardware timeslice changed.
Hardware/GSP canaries and real contention measurements are still required.

## Hint semantics

`gpreempt_hint_decide` returns an action mask, or `-1` on an error that the
caller must treat as fatal. The original and BPF paths use the same interface:

| Event | Original condition | Actions |
| --- | --- | --- |
| `GP_PREPROCESS` | LC and initialized | reset GDR flag; reserve a hint if enabled, otherwise block now |
| `GP_DUE` | LC, initialized, **now > deadline** | execute the existing two blocking launches |
| `GP_INFER` | LC and initialized | release GDR flag **after** model enqueue |

BE or uninitialized clients request no actions. The JIT receives both `now`
and `deadline` from the original `std::chrono::system_clock`; there is no
comparison against kernel monotonic time. The original 100 us anticipation
offset and reset → hint/block → model enqueue → release order are retained.
The client integration must reject preprocessing times at or below 100 us
instead of allowing upstream's unsigned subtraction to wrap. The benchmark
default is 200 us. GPU code and GDRCopy actuation remain upstream.

## Build and CPU evidence

```sh
taskset -c 8-15 make -C extension -f gpreempt.mk -j2 all test
```

CPU checks completed on 2026-09-03 UTC:

- BPF object compilation, skeleton generation, loader compilation, and the
  strongly linked shared bridge succeeded with the real `g++` compiler.
- Actual kernel-policy C with mocked helper/maps/setter: 78 cases and 2,460
  assertions. Includes both roles, every GR engine 1..8, direct/transferred
  NVOS21/NVOS64, untouched native/CE, nested/unknown roles, unknown engine,
  missing allocation correlation, syscall/NV-status/copy/handle failures,
  setter/map failures, wrong creator/role registration, shadow mismatch, and
  destruction cleanup, and two GR runlists sharing a grpID while a CE runlist
  binds/destroys the same numeric ID without changing either GR record.
  The same tests passed AddressSanitizer and
  UndefinedBehaviorSanitizer. This is **not** a verifier, concurrency, GSP, or GPU test.
- Real ubpf JIT versus an independent transcription of original hint branches:
  101,536 decisions, zero mismatches. Covers equal/before/after deadlines,
  role/initialized/reserve branches, large timestamps, and seeded random input.
  The original-C bridge passed the same cases. Five invalid inputs were
  rejected; missing pinned kernel attachment was also rejected in BPF mode.
  Error counters printed by this CPU negative-test binary are intentional;
  measured workloads must have zero errors.

## Runtime use and mandatory engagement checks

Only after the coordinator grants the GPU slot and admits the updated driver:

```sh
extension/.output/gpreempt_policy \
  --library /ABS/gpu_ext/extension/.output/libgpreempt_bridge.so \
  --pin-dir /sys/fs/bpf/gpreempt-UNIQUE --duration 300
```

Wait for `gpreempt_policy_ready`, then run the strongly linked full client with:

```sh
GPREEMPT_POLICY=bpf \
GPREEMPT_HINT_CODE=/ABS/gpu_ext/extension/.output/gpreempt_hint.bin \
GPREEMPT_BPF_MAPS=/sys/fs/bpf/gpreempt-UNIQUE \
  /ABS/gpreemptclient CONFIG
```

The loader requires a fresh private directory and removes only pins it created.
Default uprobes cover this specific shared-library inode in all processes;
`--pid` can narrow to a prestarted worker. Run a single workload at a time.
The loader has a bounded duration and signal-controlled cleanup. A clean exit
with no events is **not** successful engagement. A valid full two-role cell
must independently check all of the following:

1. Two role records, LC and BE, with distinct CUDA contexts and queried TSG
   handles, correct engines and requested timeslices; exactly two clean scope
   entries/leaves, GR initializations, captured allocations, and registrations.
2. JIT ready plus nonzero expected preprocessing/hint/due/block/release counts;
   original two blocking launches and post-enqueue GDR release observed by the
   client. The default reserved-hint workload must actually exercise `GP_DUE`.
3. Zero unknown-engine/setter/allocation/registration/map/scope/shadow-mismatch
   errors. Do not mistake ignored CE bindings for successfully controlled GR.
4. Independent numerical correctness, all required request counts, real original
   and BPF performance on identical fixed inputs, and balanced repeat blocks.
5. Post-run process exit, no Xid, zero UVM references, empty struct_ops, and
   removal of the owned pins. Keep failed canaries and unsuccessful arms.

The real canary and failed historical runs are recorded in
`gpreempt_context_smoke.md`. Corrected BPF identity/registration/numerics pass,
but strict firmware evidence rejects the CUDA 2,048-us overwrite. No GPReempt
performance comparison or complete original-GDR reproduction is claimed.

### Control-boundary repair (CPU-built, fresh runtime validation pending)

The optional `on_timeslice_control` callback keeps the policy value when CUDA
submits its later default. GP matches the captured RM handles, composite TSG,
actual GR engine, owning TGID and GPU instance zero; it deliberately does not
require the later user registration marker. A different thread in the same
process is allowed. Wrong handles/runlists/GPUs/CE/phase/TGID are ignored.
The kfunc only records a bounded request; the driver validates it after the
callback and then uses the original authorized, locked physical RPC path.

`control_override`, `control_lc` and `control_be` count successful policy
decisions, not completed hardware actions. Both role counters must be nonzero
with total equal to their sum, setter errors zero, and the real GSP completion
canary must still confirm the final timeslices. All original failed evidence
remains; this repair does not retroactively make any earlier run valid.
The XSched process-name timeslice program implements the same callback using
its existing all-engine process semantics, separately from its GR-only
RM-handle preemption targets.
