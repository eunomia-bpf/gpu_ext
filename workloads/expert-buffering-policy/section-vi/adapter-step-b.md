# Step B: private whole-expert adapter wiring

2026-09-03: source preparation and CPU checks passed. The subsequent
[private offloader build/import](adapter-build-01.md) also passes; **no GPU
request or Section VI performance cell has run**.
Step A is committed as `f5af919`; this step does not change its accepted state
interface or the completed FineMoE source/runtime.
Root reviewed the live contact points and independently reran the 61 control
checks and eight source/cleanup tests successfully before committing this step.

## What is now wired

`adapter-source.patch` modifies eight physical paths in a new private source
copy. `adapter_live.inc` supplies the worker implementation:

1. Qwen counts the actual positive `expert_mask` assignments, starts the layer
   epoch before its existing increasing-ID expert loop, and ends it afterward.
   The loop's computation is AST-identical to the frozen implementation.
   Prediction and speculative sparse fetch are disabled equally for all arms.
2. Existing `(model layer, expert)` representative tensor IDs resolve to complete
   topology nodes. Model layers are not confused with topology stage numbers.
   FIFO, native Section VI, and actual host-uBPF share this interface/executor.
3. The real copy worker snapshots current epoch/residency and holds eligible
   victim locks while native or JIT chooses. Locked/executing nodes are excluded.
   The selected whole node is evicted directly: no second legacy eviction rule.
   Both K entries and the existing strict byte/pool capacity remain enforced.
4. Inside the existing `CompleteDemand` critical section, successful checked
   whole-node H2D completion precedes `State::Admitted`, which precedes the
   existing ready notification. Failed/stale copies do not advance LIFO order.

The adapter rejects overlapping layer epochs, unknown/replaced nodes, stale
tasks and physical/metadata disagreement. It supports this serial workload,
not a general concurrent-request scheduler. A failure aborts the cell instead
of silently falling back. Fake tests exercise this code but do not establish
CUDA completion, physical eviction safety, logits, or live engagement.

## Checks and retained failures

- [`control-final.log`](adapter-cpu-02/control-final.log): 61 checks through the
  actual `adapter_live.inc`, real native/uBPF selector and the existing
  `finemoe_runtime_safety.h`, with **fake** device/node/copy operations. Includes
  three arms, residency drift, lock exclusion, stale epoch, byte exhaustion,
  failed copy, and admission-before-ready ordering. The initial compile failure
  is preserved in `control.log`; `control-02.log` is the first passing run.
- [`stage-02.log`](adapter-cpu-02/stage-02.log): eight-file zero-fuzz dry run and
  actual patch application into fresh `build/stage-check-02`. The first copy
  failed on upstream's non-source `core/core -> core/` self-loop; its partial
  tree and `stage.log` remain. Only that exact loop is excluded. Legitimate
  `finemoe/ops/{core,op_builder}` relative aliases stay inside the private copy;
  the builder is patched once, through its physical `op_builder/` path.
- [`source-check.log`](adapter-cpu-02/source-check.log): eight tests, including
  Python AST/placement checks, unchanged expert-loop AST, private package
  aliases, and actual bounded CPU process-group cleanup with a running or
  already-exited leader. No torch or CUDA module is imported by these tests.

Recheck from the repository root:

```sh
timeout 30s taskset -c 17 make -C workloads/expert-buffering-policy/section-vi -f adapter.mk -j1 test-adapter-control
timeout 30s taskset -c 17 /usr/bin/python3 -B workloads/expert-buffering-policy/section-vi/test_adapter_source.py --source workloads/expert-buffering-policy/section-vi/build/stage-check-02
```

## Later root-only private build

The subsequent build used `build/stage-check-02`, which now contains its
offloader binary. To stage another attempt, select a **fresh** directory:

```sh
taskset -c 17 /usr/bin/python3 -B workloads/expert-buffering-policy/section-vi/prepare_adapter.py --stage workloads/expert-buffering-policy/section-vi/build/offloader-01
timeout 1200s taskset -c 17 workloads/finemoe/.venv/bin/python -B workloads/expert-buffering-policy/section-vi/build_adapter.py --source workloads/expert-buffering-policy/section-vi/build/offloader-01 --log workloads/expert-buffering-policy/section-vi/adapter-cpu-02/offloader-build-01.log
```

This is the existing FineMoE Python 3.12 environment, GCC/G++ 13 and CUDA 12.9,
with `MAX_JOBS=1`, CPU 17 and CUDA devices hidden. It compiles the original 18
C++ translation units plus `adapter_state.cpp`; `PrefetchBuilder` uses
`CppExtension`, not a CUDA/nvcc source list. The original successful build05
did not set an architecture variable. The new entry explicitly sets
`TORCH_CUDA_ARCH_LIST=12.0` for this sm_120 target, not as a claim that this
flag was used in the old build. Budget up to 20 minutes; this is an operational
timeout, not a measured build-duration estimate.

The only external source include directory is the unchanged
`workloads/finemoe/`, for `finemoe_runtime_safety.h` and
`finemoe_copy_ledger.h`. The adapter's state/policy/include files are copied
under private `core/eb_section_vi/`; no frozen source is patched in place.
The wrapper owns a separate process session and checks survivors even after
its leader exits. Signal/error cleanup must empty that group. A successful
build must additionally produce exactly one fresh, nonempty `.so`, import
that exact extension path, expose revision `section-vi-private-adapter-v1` and
the snapshot API, and leave `torch.cuda.is_initialized()` false. The log records
its path and size. These build/import gates now pass as recorded separately.

After build review, the root must adapt the existing FineMoE original-HF
token/logit comparison to the three arms, using the retained exact tolerance.
It must establish real evictions, actual JIT decisions and whole-expert copy
accounting before freezing K or running the proposed 15 new cells. No old
FineMoE/page-profile cell counts toward this Section VI experiment.

## OpenCode draft and human/agent review boundary

The user-requested OpenCode CLI ran with its default model and
`snapshot:false`. Run 01 was interrupted when read-only exploration exceeded
the requested files; its JSONL/stderr and prompt remain. Run 02 could read
only the two attached bounded prompt/context files, exited 0, and its complete
final report is preserved in [`opencode-adapter-final.md`](opencode-adapter-final.md)
alongside the original JSONL/stderr.

The final was a draft, not accepted code or validation. Review rejected its
unowned-mutex unlock, invented node APIs, topology/model-layer confusion,
admission after ready notification, missing status branches, and second legacy
victim selection. The reviewed implementation above supplies concrete types,
locking and copy-completion ordering instead. OpenCode did not write source,
execute a shell/build, import CUDA, or launch a workload.
