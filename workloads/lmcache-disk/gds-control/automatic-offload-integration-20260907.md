# Automatic disk-backed KV offload: implementation target

This is the September 7 implementation direction, not a completed feature or
new performance result. It extends the existing real LMCache/cuFile storage
path; it does not restart completed baseline campaigns or change the paper.

## Target and ownership

Manage KV residency, backing copies, and pending storage operations together.
After registration, the runtime should automatically coordinate writeback,
reclaim, prefetch, and restore. BPF selects actions; LMCache/cuFile transports
bytes; vLLM owns live KV blocks and the point at which consumers may use them.
The current driver does not provide transparent same-address SSD paging.
See `disk-uvm-source-boundary-20260907.md` for the inspected 575 source boundary.

Residency and backing validity are separate facts: an object can have both
an HBM copy and a valid disk copy. A pending write is not a valid disk copy.
For the non-recompute route, release follows successful write completion and
the end of active source use. Restore must complete before its consumer runs.
A recomputable object may instead be discarded with an explicit recomputation
route; this must not be mislabeled as disk restoration.

## Current parallel implementation

- The vLLM victim-selection seam is committed in `e76a03aa`, but not installed.
  It keeps actual freeing and request rescheduling in upstream vLLM.
- Qwen 27B is implementing a bounded native/BPF candidate-selection interface,
  with a driver patch artifact and bindings. These files are still in progress.
- Qwen Next session `ses_f82a511f5ffeEc2docfRXrY3kU` repairs the real request-to-disk
  registry. Its predecessor's normal completion did not establish usability:
  `enable()` recursively acquires a non-reentrant lock, and the warm capture
  incorrectly interprets the boolean `retrieve()` return as KV chunks.
- GLM session `ses_f83256701ffek3qZigZd1Fj2qL` repairs completed-event ownership
  after async consumption. Existing attempt-02 data and its negative reference
  count warnings remain retained; no repaired-version result exists yet.

## Integration details that affect the algorithm

Use actual contiguous disk-backed prefixes, not the assumption that every
computed token is saved. Capture the engine's internal processed-chunk tuples,
and confirm disk backing through successful GDS writes or actual backend
presence. CPU-cache hits alone do not establish disk backing.

Price recovery using measured read cost and computed-token recompute cost.
Write duration is not a read estimate. Actual transfer bytes and freeable KV
bytes are distinct: shared KV blocks may not be reclaimable even when the
disk object contains their data. Do not clamp read bytes to freeable bytes.

Keep the stock victim's priority class; vLLM's priority convention must not
be silently inverted. The kernel validates the returned candidate and allowed
action, but must not hard-code the candidate-cost algorithm into its enforcer.
That algorithm belongs to the matched native and BPF policy implementations.

## Next real serving comparison

Compare stock victim selection, native disk-aware selection, and the same BPF
selection under identical arrivals, generation lengths, and KV capacity that
actually causes preemption. Attempt 02 had an 8192-token KV pool and at most
two short running sequences; repeating it unchanged cannot measure reclaim.

Report serving throughput, TTFT and end-to-end latency alongside preemptions,
disk bytes and recomputation. Retain adverse results. A selector-only result
does not establish automatic offload: request association, actual block reclaim,
write completion, and the selected recovery route still need to be connected.
Freeing a staging buffer or returning KV blocks to a fixed pool is not evidence
of releasing physical HBM to other processes.

## Execution update, September 7, 19:58 UTC

The consumed-event repair and its complete 20-cell comparison are published
in `8987a14b` / `c3c6d42e`; the warning disappears but the deadline-prefetch
policy still loses to demand FIFO. Those completed cells will not be repeated.
Root installed the committed vLLM selection seam after that campaign ended;
installation and the preserved original are recorded in `8d7493e5`.

The registry's Qwen Next request ended with an actual gateway API 524 after
automatic retries. Root confirmed the terminal CLI and absent live session,
then resumed the same task/session with GLM. This was not termination for
silence or an artificial execution deadline. The selector, registry repair,
and serving integration remain the three live local tasks; the last two now
use GLM because the Next request failed. No new reclaim performance is claimed.

Root's first ordinary selector build passed for the native shared library but
failed for BPF. The current `kv_reclaim_abi.h` overflow check in
`uvm_kv_reclaim_mul_hi` is optimized into unsupported `__multi3` calls by the
BPF compiler. The BPF file also places its `preserve_access_index` pragma
after the shared record definitions, producing an unused-attribute warning.
Both diagnostics were returned to the selector owner; no driver load or
performance claim follows from the native-only build. Build commands from
the repository root (outputs stay in the existing temporary task directory):

```sh
cc -O2 -g -Wall -Wextra -fPIC -shared workloads/lmcache-disk/gds-control/kv_reclaim_native.c -o /tmp/opencode/kv_reclaim_native-build-check.so
clang -g -O2 -target bpf -D__TARGET_ARCH_x86 -Ivmlinux/x86 -Ilibbpf/src -c workloads/lmcache-disk/gds-control/kv_reclaim_policy.bpf.c -o /tmp/opencode/kv_reclaim_policy-build-check.bpf.o
```

## Selector build follow-up, September 7

The local implementation replaced the overflow helper with explicit 32-bit
limb multiplication and updated its comparison call sites. Root rebuilt the
actual `kv_reclaim_policy.bpf.c` above successfully, with no compiler output.
The native shared library also builds with
`-Werror=implicit-function-declaration -Wl,-z,defs`, so the previously missing
helper is no longer left as an unresolved symbol. These are compilation and
link results, not a loaded-driver or serving-performance result.

The backing registry now captures the installed GDS read path separately from
write completion; ordinary Python compilation of the registry and selector
binding passes. Complete backing coverage and actual object-transfer byte
accounting are still being integrated with the serving consumer. All three
local OpenCode sessions remain live; no extra session or execution timeout
was introduced. The driver patch and real reclaim runner are not yet ready
for a new performance campaign. No paper files were changed.

## Driver implementation and build

The local-model driver patch is now applied and committed as `ff68a1d4` on
`revision/gpu-storage-decision-575`. It adds command 83 and
`gpu_kv_reclaim_ops`, preserving the existing storage and memory interfaces.
The first module build failed because the candidate-vector ioctl exceeds
the driver's existing 288-byte stack-parameter limit. Root changed this one
route to the existing `UVM_ROUTE_CMD_ALLOC_NO_INIT_CHECK`, made its local
handler static, and aligned the estimate-cap literal with the shared ABI.
The patch artifact includes these small integration fixes.

The following ordinary build then completed with exit 0:

```sh
make -C kernel-open modules -j8 KERNEL_UNAME=6.15.11-061511-generic CC=/usr/bin/gcc-14 NV_KERNEL_MODULES='nvidia nvidia-uvm'
```

Output `kernel-open/nvidia-uvm.ko` is 62,413,352 bytes; its vermagic is
`6.15.11-061511-generic SMP preempt mod_unload modversions`. Build warnings
report the differing GCC package revisions and missing module descriptions.
The module has not been loaded. This is a compiled selection interface,
not a completed serving or automatic-offload result. The first serving
adapter draft still needs its import, backing-field mapping, bootstrap and
actual recovery-route integration fixes; its local owner is continuing.

## Actual module and policy loading

Root built the loader against the repository libbpf and attempted loading
under both revision experiment locks, with no GPU compute process present.
The first attempt still used the old module: terminating its root-owned
storage loader without sudo failed, so rmmod reported in-use and insmod
reported file-exists. Its missing `gpu_kv_reclaim_ops` BTF error is retained
separately, not attributed to the new module.

After terminating that loader with sudo, rmmod and insertion of the newly
built `ff68a1d4` module succeeded. The new policy then reached the actual
verifier but failed with EACCES after 27 instructions: reading
`ctx->candidates[ctx->request.stock_index].cookie` uses a variable offset
into the trusted BTF context. The compiler-only success above does not
establish attachability. The implementation owner received this concrete
diagnostic to repair BPF candidate access without moving policy into the
kernel or relaxing the verifier.

Raw loader logs are retained in
`../raw/kv-reclaim-load-575-20260907-01/`. The existing storage policy was
reattached successfully to the new module (`attached`); loader PID 1744933
was live at handoff. The new module remains loaded, the experiment locks
are released, and the saved pre-change module remains at
`/var/tmp/gds-restore-before-stale-20260907.AWgdmi/nvidia-uvm.ko`.
No new KV reclaim policy is attached and no serving performance was measured.

Source clarification for that actual rejection: Linux v6.15
[`check_ptr_to_btf_access`](https://github.com/torvalds/linux/blob/v6.15/kernel/bpf/verifier.c#L6722)
rejects non-constant or nonzero `var_off` before inspecting BTF fields.
Bounding a runtime candidate index therefore does not by itself permit
direct indexing into this typed context. The implementation owner was
directed to use constant-offset reads or a suitable bounded snapshot, while
keeping the matched policy algorithm and the verifier unchanged. The
in-progress index-argument edit has not been counted as a successful repair
or retried against the live module.

The subsequent per-CPU snapshot implementation builds through the new
Makefile targets for the BPF object, native library and loader. Root fixed
two trivial integration errors before building: copying the request struct
without an invalid unary dereference, and specifying the snapshot map key.
Actual loading then fails at a different instruction: the request struct
copy becomes a 64-bit read spanning the 32-bit `stock_index` and `pad0`
fields, which the typed-access verifier rejects after 12 instructions.
The raw `kv-reclaim-snapshot-load-20260907.log` is retained beside the earlier
attempts. The owner is changing the snapshot to explicit fixed-width field
reads; neither compiled snapshot version is counted as attached. Existing
storage-policy loader PID 1744933 remains live, with no driver reload in
this follow-up.

The fixed-width snapshot follow-up builds with the same three Makefile
targets and advances beyond both prior BTF access errors. Actual loading
on the already-loaded module now fails with E2BIG: the verifier processes
1,000,001 instructions against its 1,000,000 limit, with 21,322 total states.
The complete 229,733-byte `kv-reclaim-fieldwidth-load-20260907.log` is
retained in the same raw directory. No KV reclaim policy was attached.
The shared selector repeatedly scans the worst priority class within its
candidate loop and repeats recovery-cost comparisons; the implementation
owner is removing redundant computation without changing the algorithm or
the kernel limit. This is an actual loading failure, not a newly imposed
experiment prerequisite. No performance sample is claimed from this attempt.

The runner's Qwen Next session returned an actual API 524 error. It was
resumed with local GLM; its subsequent empty terminal response was not
treated as task completion and the same session was continued. The selector
and serving-adapter owners remain active, with at most three OpenCode
sessions. No session was stopped because of silence, and no paper files were
modified.

## Factored policy attaches on the real driver

The shared selector now computes the worst priority class once per decision,
retains the current best recovery cost, and compares each candidate once.
Its BPF build uses out-of-line policy functions; native and BPF still share
the same cost, priority and tie rules. Root rebuilt the BPF object, native
library and loader using their Makefile targets. Actual loading on the
already-loaded `ff68a1d4` module succeeds: the retained
`kv-reclaim-factored-load-20260907.log` reports `attached`, and loader PID
1783868 was confirmed live. Neither a driver reload nor a kernel verifier
limit change was needed. The previous failed attempts remain retained.

Root also bounded the native invalid-count fallback's cookie access to the
eight-entry candidate capacity; this does not change valid-input selection.
This step establishes build and attachment, not a serving performance result.
The LMCache adapter and stock/native/BPF performance runner remain in their
existing local OpenCode sessions. No paper or historical measurements change.

## Serving adapter and startup integration

The serving adapter now connects the existing scheduler victim callback to
the matched native/BPF selector and enables the real backing registry after
LMCache cache registration. A pending full-recompute choice bypasses cache
lookup with zero external tokens; a disk-prefix choice retains the original
LMCache lookup and restore path. The earlier draft that suppressed both
routes is fixed. Recovery records distinguish admission from completed I/O.
The adapter uses the installed connector implementation and derives block
bytes from its actual single-group cache allocation rather than a scheduler
spec type that vLLM replaces during initialization.

Root applied the six-line opt-in bootstrap addition to the existing
`bootstrap/sitecustomize.py`; its corrected patch artifact records an
already-applied change. Empty enabled frontend processes now skip writing
shutdown diagnostics, preserving the EngineCore process's populated record.
Ordinary Python compilation of the adapter and startup module passes.
These are source integration results only: no server has yet run with this
adapter and no new reclaim-policy performance number exists. The separate
local runner session remains active; previous performance cells are not
repeated or reclassified by this integration step.

The serving-adapter session subsequently completed normally, followed by the
selector session; both implementations were already committed. The selector
handoff changes only the verifier-linkage explanation, not the attached
algorithm. The runner session reached an actual `finish=length` response at
85,276 input tokens without producing its file, and was continued in the
same session. Its first increment now exists at `run_gds_kv_reclaim.py`, but
is not yet a complete executable campaign.

To parallelize the remaining implementation without overlapping file writes,
a new local Qwen 27B session owns only `kv_reclaim_calibration.py`; the runner
session retains its runner and contract note. The helper's agreed
`run_calibration` API measures actual sequential recompute TTFT per supplied
prompt token, retains raw requests and server exit, and returns one observed
price shared by the campaign arms. No calibration or new serving cell has
run yet. The previous sessions were not interrupted for silence and no
fourth concurrent session was started.

## First real recompute calibration

`kv_reclaim_calibration.py` is implemented and its local session completed
normally. Root ran it once on the RTX 5090 / 575 driver with the actual
1536/1024-token warm arrays, the same model, two sequences and 384 MiB KV
allocation. All eight requests returned HTTP 200 and 16 output tokens each;
the server returned 0, with no helper cleanup errors. Its startup log reports
an actual 4,096-token GPU KV pool. The retained raw directory is
`../raw/kv-reclaim-recompute-calibration-575-20260907-01/`.

The rounded median observed TTFT per input token is 62,502 ns/token. This is
the shared native/BPF recovery-price proxy, not a pure GPU compute cost.
The first request's 535.475 ms TTFT is retained alongside the seven roughly
64--67 ms requests; no sample was dropped. Across this sequential short-output
phase, 128 generated tokens take 4.360114 s (29.357033 token/s), excluding
server startup and teardown. This calibration is not the concurrent long-output
stock/native/BPF performance comparison. The startup/teardown log also retains
vLLM's forced EngineCore termination and semaphore warning despite the top-level
exit code 0; no clean lifecycle or output-equivalence claim is made.

The first direct stock `run_cell` invocation in campaign `gds-kv-reclaim-575-20260907-01`
fails before starting a server with `RecursionError`: its environment wrapper
calls the same overridden environment function recursively. Root changed that
one call to the existing captured `campaign_base._BASE_SERVER_ENVIRONMENT`.
The failed result remains retained, with no performance sample. A fresh first
stock attempt in campaign `gds-kv-reclaim-575-20260907-02` is now running,
reusing the completed calibration. The separate runner session continues its
CLI and campaign orchestration; no completed calibration or prior workload
cell was repeated.
