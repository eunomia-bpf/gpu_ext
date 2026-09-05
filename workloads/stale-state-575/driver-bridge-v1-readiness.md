# Stale-state driver bridge v1: source/build readiness

Date: 2026-09-04  
Scope: CPU/source/build gate only; no module load, BPF attach, or GPU run

## Outcome

The version-1 bridge is ready for a controlled live load/attach preflight, but
it is not a live result. `driver-bridge-v1.patch` applies to the sibling NVIDIA
575 source at revision `6a5b3bb5857e5e2890559750f51e0a0e5834d09d`, and a fresh
copy of that source builds a complete `nvidia-uvm.ko` for
`6.15.11-061511-generic`. The installed module and sibling source were not
modified.

The patch changes ten driver files:

- adds `nv-gpu-stale-state-v1.h`, `uvm_stale_state_v1.c`, and
  `uvm_stale_state_v1.h`;
- appends one callback to `gpu_mem_ops` and adds one STRUCT_OPS-only trusted
  setter in `uvm_bpf_struct_ops.c`/`.h`;
- routes the versioned mode through the existing validated prefetch effect
  point in `uvm_perf_prefetch.c`;
- adds the new source to Kbuild; and
- adds a 30-assertion CPU test under `kernel-open/tests/stale-state-v1`.

The matching workload-side source is in `driver-bridge-v1/`: an exact ABI
mirror, a read-only struct_ops policy, a 15-check ABI/model test, and a
CPU-only Makefile.

## Contract implemented

`/proc/uvm_stale_state_v1` is root-only and accepts:

```text
configure native GENERATION
configure bpf GENERATION
publish GENERATION SEQUENCE PHASE SOURCE_MONO_NS
disable GENERATION
```

Publication creates one immutable RCU object containing `(sequence, phase,
source_mono_ns, published_mono_ns)`. The driver captures
`published_mono_ns` with `ktime_get_ns()` and rejects zero, noncontiguous,
regressing, future, wrong-generation, or invalid-phase publications.
Configure/disable closes an active-callback gate and waits for in-flight
callbacks before changing generation-scoped state.

Native and BPF modes receive the same driver-captured snapshot, decision
timestamp, fault page, and legal maximum region. Native calls the pure policy
model directly. BPF may read only the immutable context prefix and must submit
exactly one legal action through `bpf_gpu_stale_state_v1_request`; the setter
is absent from the KPROBE kfunc set. The driver rejects context mutation,
missing/duplicate/conflicting requests, callback/request disagreement, and
invalid actions, then materializes either the complete legal maximum or the
empty region through the existing transition validator.

Both paths increment common snapshot-read, decision-request, decision-record,
effect-request, effect-record, and selected/finished diagnostic counters.
The address-free diagnostic contains snapshot identity/age, fault and bounds,
requested/output regions, action, validation result, final effect, mode, and
status. Separate native/BPF invocation counters establish which consumer ran.

## Checks completed

- `git apply --check` accepts `driver-bridge-v1.patch` against the untouched
  sibling source.
- Driver pure-model/order test: 30 assertions passed.
- Full NVIDIA module build: success with GCC 14; `nvidia-uvm.ko` is
  62,342,720 bytes and has the expected 6.15.11 vermagic. The build emitted
  only the source tree's existing module-description and compiler-package
  revision warnings.
- Built module BTF: `gpu_mem_ops` is 56 bytes with seven members; the new
  callback is the appended member at bit offset 384. The read-only input is
  88 bytes, decision context 104 bytes, diagnostic 176 bytes, and the trusted
  setter is present.
- Workload BPF object: clang-18 build succeeds; it contains the uniquely named
  callback section and 56-byte struct_ops map. Disassembly contains only stack
  stores plus the read helper and trusted setter call, with no store through
  the driver context.
- Workload ABI/model test: 15 checks passed.
- Existing stale-state offline suite: 13 Python tests passed; the native versus
  host-uBPF differential remains exact across 306,012 calls with zero contract
  errors.
- The active analyzer schema now names the BPF read-only context instead of a
  nonexistent snapshot helper and gates the counters exported by this bridge.

Append-only compatibility is source-backed rather than asserted from a live
load. The prior module BTF has six 8-byte `gpu_mem_ops` callbacks. Repository
libbpf sets the map value size from kernel BTF, zero-allocates that complete
size, and copies only members present in an older policy's local BTF
(`libbpf/src/libbpf.c`, `bpf_map__init_kern_struct_ops`). Therefore an existing
six-member policy leaves the appended callback null. The controlled preflight
must still prove this with the newly loaded module.

## Remaining live gate

Do not describe this as a completed experiment. A privileged maintenance
window must still:

1. apply/build/install and load the module, with pre/post kernel-safety checks;
2. load the BPF object and retain the real verifier log;
3. prove an old six-member policy still loads with a null versioned callback;
4. attach the diagnostic observer and exercise the implemented workload-truth
   FD coordinator against the real proc endpoint, reconciling every common
   counter with retained publication/decision/effect records; and
5. pass one excluded seven-cell preflight before enabling the 21-cell formal
   matrix.

Until then `run_study.py live` correctly remains fail-closed.
