# FineMoE preparation: original-model reference and retained failures

The original-model GPU reference has passed. FineMoE offload-policy numerical
correctness and performance remain unestablished. Earlier attempts below are
retained chronologically; the latest successful stage is at the end.

`raw/golden-v1/stage/worker.log` records the original full BF16 Qwen checkpoint
failing while loading its seventh of eight shards. The worker exited 1 before
any generation: **0 of 73 requests and 0 of 9 repeat checks completed**.
The failing allocation requested 20 MiB with 14.62 MiB free; PyTorch reported
25.77 GiB allocated and 4.97 GiB reserved but unallocated. That large unused
reservation is a fragmentation/reservation-pressure clue, not proof of the
precise cause or proof that the full 26.67 GiB checkpoint cannot fit.

The controller's `stage/result.json` preserves the failed status and confirms
clean teardown: no cleanup error, GPU back to 15 MiB with no compute process,
UVM references 0, empty struct-ops state, and no Xid/kernel abnormality. Its
44 telemetry samples show a 32,095 MiB peak and no disallowed throttling.
`campaign.json`, `campaign-failure.json`, launch/environment, raw stdout, and
telemetry remain in the original directory; nothing is relabeled as a valid cell.

The approved next attempt uses the canonical PyTorch environment variable
`PYTORCH_ALLOC_CONF=expandable_segments:True`, uniformly for golden, history,
preflight, and every formal arm. The controller removes an inherited legacy
`PYTORCH_CUDA_ALLOC_CONF` and records only its removed name, not its value.
PyTorch documents this allocator option as experimental; it may reduce some
allocation fragmentation but does not guarantee this workload will fit.
See [PyTorch CUDA environment variables](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)
and [CUDA memory management](https://docs.pytorch.org/docs/main/notes/cuda.html).

The retry will use fresh `raw/golden-v2`. Checkpoint revision, BF16 precision,
the full 24-layer/60-expert model, frozen MT-Bench 64/8/1 cohort, 0.5 offload pool
budget, and repeat-derived numerical-tolerance rule remain unchanged. No extension
rebuild or GPU retry was performed as part of this environment-only repair.

## Second load: memory hurdle passed, native SIGILL remains

After the allocator-only change, `raw/golden-v2/stage/worker.log` confirms all
eight shards loaded. Peak sampled GPU memory was 27,925 MiB, below v1's peak.
The process then terminated with return code −4 (`SIGILL`) before any completed
request: again 0/73 requests and 0/9 repeats. There is no Python traceback, so
the specific native instruction/library is not established by this log.
This is a different failure from v1's OOM, not a numerical result or evidence
that a particular GPU policy caused the failure.

The v2 controller again recorded clean teardown: GPU 15 MiB, no compute process,
UVM references 0, empty struct-ops state, no Xid/kernel abnormality, and no cleanup
error. All 50 telemetry samples passed the existing throttling gate.

The next step is a single same-model native-stack diagnostic in fresh
`raw/golden-sigill-gdb-01`, using `compare.py --mode golden --native-backtrace`.
GDB runs in batch with init files, auto-load and debuginfod disabled; raw stdout
includes `bt 30` and eight instructions at the stopped PC, followed by explicit
inferior termination. Existing leases, telemetry and owned-process cleanup remain
active. Only this diagnostic enables Python faulthandler. Its campaign is marked
`diagnostic=true` and is rejected as a golden/history/preflight reference even
if execution unexpectedly succeeds. A normal no-debugger golden must succeed
before later stages; no model, precision, cohort, budget or tolerance changes
are part of this diagnostic.

## Native-stack observation and scoped compatibility retry

The diagnostic stopped at its first SIGSEGV inside `libcuda.so.575.57.08`, through
`cuModuleLoadData` and Triton's `cuda_utils` binary loader. GDB then explicitly
killed its inferior; cleanup passed. Because a debugger can stop before a
driver's signal handler runs, this observation does **not** establish the cause
of v2's un-debugged fatal SIGILL.

The recently generated kernel name was `_bmm_outer_product_kernel`, targeting
CUDA SM120. Installed Torch source registers this as an `aten::bmm` override for
outer-product shapes; Qwen's RoPE implementation uses precisely that matrix
shape. RoPE bmm is therefore a candidate, not a confirmed faulting operator.
The observed toolchain is Torch 2.13.0+cu129, Triton 3.7.1 and its bundled ptxas
12.8.93, with userspace libcuda 575.57.08. Cache directory identifiers are not
used as evidence or provenance.

The approved normal `raw/golden-v3` retry uniformly sets
`TORCH_DISABLE_NATIVE_JIT=1` for every stage and arm. This uses PyTorch's existing
compatibility switch to disable automatic Triton/DSL operator overrides without
editing Torch, Qwen's mathematics, checkpoint, precision, cohort, cache budget
or numerical-tolerance rule. Worker startup verifies the official
`torch._native.common_utils.check_native_jit_disabled()` predicate and records
`runtime_versions.torch_native_dsl_jit_disabled=true` in golden/history/cell raw
results. The separate actual uBPF selector JIT is unaffected and retains its
exact oracle and JIT-call engagement gates. This is a proposed compatibility
remedy, not a verified root-cause claim; full normal 73+9 validation is still
required. No GPU run or extension rebuild was performed during this change.

## Normal v3 completed; final reference will retain both arrays

`raw/golden-v3` completed normally (exit 0, no debugger): 73 original requests
produced 1,168 generated tokens, and nine repeat checks produced another 144.
All nine repeat token sequences matched; their reported maximum absolute logit
error and the frozen absolute tolerance were both 0.0. The persisted original
nine float32 arrays contain 21,878,784 finite values, shape `(16, 1, 151936)` per
request, and total 87,516,288 file bytes. Independent read-only checks verified
all input IDs and request order, token counts/ranges, raw token timestamps,
original-array shape/finiteness, repeat-record counts/tokens, runtime inventory,
reported tolerance consistency, telemetry and clean teardown. No performance
claim is made: another task was compiling on separate CPU cores during this
correctness preparation.

Observed runtime versions were Python 3.12.3, Torch 2.13.0+cu129 (CUDA 12.9),
Transformers 4.49.0 and NumPy 2.5.2, with the native-DSL-disabled predicate true.
The 961 telemetry samples peaked at 28,041 MiB and 53 C with no disallowed
throttling; cleanup returned the GPU to 15 MiB, no compute process, UVM 0 and no
Xid/kernel abnormality. This successful compatibility run still does not prove
the precise cause of the earlier signal.

Limitation: v3 saved the original logits but only scalar errors for the repeat
logits, so an offline reader cannot independently recompute those nine errors
from two arrays. V3 remains unchanged as valid normal preparation. The only next
worker change writes `question-ID-repeat-logits.npy` and records its filename
in each repeat row. No generation, random seed, cohort, mathematics, tolerance
rule or budget changes. A fresh full normal `raw/golden-v4` will be the final
reference after both original and repeated arrays pass offline recomputation.

## Final normal v4: independently recomputed reference passed

`raw/golden-v4` completed normally (exit 0, no debugger): all 73 original
requests and nine repeats passed, producing 1,168 + 144 generated tokens.
The 18 persisted float32 arrays have shape `(16, 1, 151936)` each, contain
43,757,568 finite values, and total 175,032,576 file bytes. An independent
CPU-only audit loaded every original/repeat pair: all values were exactly equal,
and all nine recomputed maximum absolute errors were **0.0**. The frozen
absolute tolerance is therefore 0.0; it is not changed for the offload arms.

The audit also verified frozen input IDs, request/log order, 16 generated token
IDs per request, token timestamps and derived TTFT/TPOT, repeat token equality,
the unchanged 42-file runtime inventory, and the same runtime versions/native
DSL compatibility predicate recorded above. All 961 raw telemetry samples had
inactive throttle flags; peaks were 28,041 MiB and 50 C. Cleanup returned to
15 MiB, no compute process, UVM 0, empty struct-ops state and no Xid/kernel
abnormality. No performance claim is made from this reference preparation.

V4 is the final reference for the next 64-request history stage and the four
numerical canaries. History checks exact original-HF token IDs; each canary also
checks all logits for its disjoint warmup and eight held-out requests against
the same fixed tolerance. No actual FineMoE offload arm has passed those gates
yet, and no four-arm performance block is complete. Earlier failures and v3
remain unchanged.
