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

## Live pressured stock cell and continuation work

The runner and CLI are complete and published in `fd334cd9`; the actual
first stock cell in `gds-kv-reclaim-575-20260907-02` remains running as of
22:58 UTC. Its earlier snapshot reported 1,648 generated tokens and zero
counted preemptions; the later retained `live-metrics-02.txt` reports 1,888
generated tokens and 7,295 counted preemptions, with zero running and four
capacity-waiting requests at the instant sampled. Eight length-completed
requests correspond to the cold population; no completed warm-cell throughput
is available. These cumulative, periodically published metrics are not a
complete scheduler trace and cannot establish continuous lack of progress.

The bounded server-log tail now shows repeated disk-prefix restoration for
requests p0 and p4 at total lengths 2,432 and 1,648, respectively; this differs
from the earlier p0/p1 pair at 2,304/1,776. Thus the run has made some progress
and changed its active request composition, despite repeated restore cycles.
The GLM local session is diagnosing the actual scheduler/connector source
path, rather than treating a baseline stall as a policy improvement. The
original run is not stopped or assigned a new wall-clock timeout.

A separate Qwen Next local session owns a bounded runner improvement:
explicit continuation of existing campaign output, reuse of completed
per-cell results (including failed attempts), and per-request result
checkpoints. An unfinished nonempty cell must not be overwritten or repeated
automatically. This implementation is still in progress; it does not change
the already-running Python process or add new performance admission checks.
There are two active local sessions, below the three-session limit. The
completed calibration is reused, and no new native/BPF reclaim performance
result or transparent UVM-to-disk paging implementation is claimed.

## First pressured stock cell completed with request failures

The stock cell ended at 23:04 UTC through the existing runner lifecycle;
root did not interrupt it. All eight cold population requests succeeded.
Only two of eight warm requests completed, each generating 1,024 tokens;
the other six records contain `TimeoutError: timed out` from the existing
600-second urllib socket timeout. The warm phase lasted 1,247.540913 seconds.
Its 2,048 completed output tokens yield 1.641630 token/s of completed-output
goodput, not successful throughput for all eight requests. Successful-request
TTFT median is 10,451.494 ms and excludes the failed requests, whose partial
response fields were not retained by this runner version. No zero-latency
or zero-generation value is inferred for those missing fields.

The complete `result.json` (55,040 bytes) and `server.log` (32,519,959 bytes)
are retained under `../raw/gds-kv-reclaim-575-20260907-02/block-00/position-0-stock/`.
The final log shows generation resuming at about 39.6 token/s with one
running request and none waiting after other clients timed out. The server
returns zero, but that does not erase the six request failures. Likewise,
the runner's zero preemption-message matches are not zero scheduler
preemptions: the retained live metrics explicitly observed 7,295.

The matching native cell (`block-00/position-1-native`) has now started with
the same capacity, prompts, arrival order, generation limit, transport and
62,502 ns/token calibration. It is a real performance attempt, not a claim
that the baseline pathology is resolved or that the new policy is faster.
GLM continues scheduler/connector diagnosis; Qwen 27B independently examines
allocation/reservation arithmetic, while Qwen Next implements continuation
and request checkpointing. All three local sessions remain within the limit.

## First pressured native cell completed with request failures

The native cell finished through the existing lifecycle at 23:17 UTC.
All eight cold requests succeeded; warm requests p0, p1, p6 and p7 each
completed 1,024 output tokens, while p2 through p5 timed out. The warm phase
lasted 681.690323 seconds, yielding 6.008593 token/s of completed-output
goodput and a successful-request TTFT median of 7,997.847 ms. The full result
and server log (94,825 and 15,399,783 bytes, respectively) are retained under
`../raw/gds-kv-reclaim-575-20260907-02/block-00/position-1-native/`.

Native also entered the repeated disk-prefix restoration cycle before the
timed-out requests left. Its higher completed-output rate than this stock
attempt is not a stable policy-effect estimate: these are single runs with
different request failures, and neither completes the workload. The native
adapter was requested, but its expected shutdown diagnostics did not arrive;
the runner records that absence rather than inventing decision or recovery
counts. Server return code zero does not establish eight successful requests
or clean EngineCore teardown.

The matching BPF cell (`block-00/position-2-bpf`) has started with unchanged
workload and transport settings. Local source diagnosis and runner continuation
work remain active. No completed cell or calibration is repeated, and the
current failed stock/native records will not be overwritten by a later fix.

The Qwen Next continuation task subsequently ended with an actual HTTP 524
API error after automatic retries, without writing a source change. Root
confirmed its CLI handle had exited and resumed the same OpenCode session
using GLM. This is recovery from a terminal provider error, not termination
for silence or a new short timeout. The allocator's Qwen 27B session and the
separate GLM diagnosis remain live; the resumed writer is the third session.

## Scheduling-overlap discriminator

The current BPF cell also exhibits repeated prefix restores. Before expanding
the same troubled path to more blocks, root will measure one stock control
with the source-native `--no-async-scheduling` option. This changes only vLLM
schedule/compute overlap: 384 MiB KV, two running sequences, four HTTP
workers, the same eight 1536/1024 prompts, 1024 output tokens, arrival order,
GDS transport and calibration remain fixed. This is a separate diagnostic
performance ablation, not a replacement baseline or a claimed fix.

The source motivates this discriminator: `config/vllm.py` normally enables
async scheduling and gives it two concurrent batches; scheduler initialization
enables deferred KV block freeing for overlapping batches with a consumer
connector. `_free_request_blocks()` may queue a future free while the running
allocation-failure loop immediately tries another victim. These conditions
suggest a possible repeated-admission interaction but do not prove the live
cause. Disabling overlap also changes other scheduling behavior, so even a
positive control cannot alone identify a specific faulty line.

The control is queued behind the existing GPU and struct-ops locks and writes
only to `../raw/gds-kv-reclaim-scheduling-ablation-575-20260907-01/stock-sync/`.
Root uses the existing `run_cell` with `arm="stock", block=0, position=0`;
the process-local `ops.server_argv` wrapper appends `--no-async-scheduling`
to its existing argument list, which the normal result record retains.
No project source, driver or running server is changed by that invocation.
If it removes the repeated-restore behavior, investigate the overlap path
before attributing gains to a BPF policy; if not, continue the admission and
capacity investigation. All failed original cells remain published.

## Initial three-arm pressure block finished

The BPF cell finished through the existing runner at 23:30 UTC. Like native,
it completed warm requests p0, p1, p6 and p7 (1,024 output tokens each),
while p2 through p5 timed out. Its 680.836021-second warm phase yields
6.016133 token/s of completed-output goodput. Successful-request TTFT median
is 7,885.633 ms. The BPF result and server log (94,737 and 15,445,893 bytes)
are retained at `../raw/gds-kv-reclaim-575-20260907-02/block-00/position-2-bpf/`.
Its expected adapter exit diagnostics are absent, as for native.

| First block only | Stock | Native | BPF |
| --- | ---: | ---: | ---: |
| Completed warm requests / attempted | 2 / 8 | 4 / 8 | 4 / 8 |
| Warm request timeouts | 6 | 4 | 4 |
| Warm elapsed seconds | 1247.541 | 681.690 | 680.836 |
| Completed-output goodput, token/s | 1.641630 | 6.008593 | 6.016133 |
| Successful-request TTFT median, ms | 10451.494 | 7997.847 | 7885.633 |

All eight cold requests succeeded in each arm. This is one pressure block
with request failures, not the planned five-block successful performance
comparison. Native and BPF have the same failed request indices in this
block, but their similar goodput does not bound general mechanism overhead
or establish decision equivalence. The successful-request latency medians
must not be interpreted as all-request latency. Historical LMCache results
remain separate and unchanged.

The queued stock scheduling-overlap ablation acquired the GPU locks and
started automatically after BPF exited. Its server receives
`--no-async-scheduling`; the runner source was still unchanged at launch.
It has begun real cold population and warm serving, with no final result yet.

## Scheduling-overlap control completed all requests

The stock control with `--no-async-scheduling` completes all eight warm
requests, each with 1,024 generated tokens, and has no request errors.
Its 8,192 output tokens take 114.115222 seconds: 71.787093 token/s. All eight
cold requests also succeed; the server returns zero. The actual command
retains the added flag, with unchanged model, 384 MiB KV, two running
sequences, four HTTP workers, 250 ms stagger, prompts, generation bound,
GDS transport and price proxy. Runner source had not changed.

Raw `result.json` and `server.log` (171,259 and 212,503 bytes) are retained
under `../raw/gds-kv-reclaim-scheduling-ablation-575-20260907-01/stock-sync/`.
Warm TTFT median is 25,599.252 ms across all eight successful requests;
unlike the failed first-block medians, this includes the entire workload.
Comparing those medians without their different completion sets would be
misleading. Ordinary shutdown still contains the previously documented
EngineCore termination/semaphore warning; request completion is not a claim
about clean process teardown.

This single-variable control supports dependence on vLLM scheduling overlap
in this workload, but does not isolate the exact faulty line or prove a
general repair. It specifically contradicts interpreting the failed
stock-to-native/BPF comparison as an established storage-policy gain: an
existing stock runtime option completes the workload without either policy.
It does not disable gpubpf's asynchronous resource-state model or turn the
cuFile backend into a different transport.

Root has started native with the same non-overlapped configuration, followed
by the matching BPF comparison; no old completed cell is rerun. In parallel,
the GLM scheduler diagnosis is now tasked with a minimal async-preserving
compatibility patch, confined to the existing preemption-seam artifact and
its note. Root will apply any accepted patch only after active GPU work
finishes. The working non-overlapped comparison does not close or replace
repair of the original overlapped path.

## Matching non-overlapped native result

Native with the same `--no-async-scheduling` option completes all eight
warm requests and all eight cold requests without request errors. Its
8,192 warm output tokens take 121.860751 seconds (67.224270 token/s);
warm TTFT median is 27,454.104 ms across all eight requests. The raw result
and server log (171,904 and 219,556 bytes) are retained under
`../raw/gds-kv-reclaim-scheduling-ablation-575-20260907-01/native-sync/`.
The expected adapter exit diagnostics are still absent and recorded as such.

This native sample is slower than the stock control's 71.787093 token/s.
Neither a policy gain nor a stable regression estimate follows from these
single measurements; the adverse observation is preserved. Both complete
the workload, unlike the earlier overlapped cells. The matching BPF cell
is now running at the same root's `bpf-sync/` directory. Source for the
runner remained unchanged at both launches; the local continuation writer
and original async-path repair tasks remain active.
