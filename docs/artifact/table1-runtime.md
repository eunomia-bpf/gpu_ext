# Table 1 (RTX 5090) runtime dependency map

Navigation for reproducing the current RTX 5090 Table 1 prefill token/s loss
results. Companion to [reanalysis.md](reanalysis.md) (CPU-only statistics) and
[ARTIFACT.md](../../ARTIFACT.md). It maps component ownership, build/run
entrypoints, measured shapes, recorded revisions, and fresh-checkout gaps. This
document itself claims no new GPU run, measurement, or experiment; its build
references are retained build evidence, not Table 1 measurements: the
[component-only build
record](../../workloads/llama.cpp/observability_overhead/revision-rq4/raw/table1-component-build-20260909.jqAD9c/README.md)
and the [fresh runtime build
record](../../workloads/llama.cpp/observability_overhead/revision-rq4/raw/runtime-published-build-20260909.IktLa7/README.md),
both linked below.

## Measured quantity (do not extend)

- Metric: llama.cpp pp512 prefill throughput (token/s) from
  `llama-bench -p 512 -n 0 -r 1 -ngl 99 -o json`, warmup on,
  `GGML_CUDA_DISABLE_GRAPHS=1`, clients pinned to CPUs 8–15. Overhead is
  `100*(baseline-tool)/baseline` within a block; headline = arithmetic mean
  of the paired percentages. Prefill timing excludes probe startup, JIT, and
  final collection.
- Stack: RTX 5090 (sm_120), driver 575.57.08, Linux 6.15.11, CUDA 12.9.

## Component ownership

| Component | Owner | Tracked location / revision |
| --- | --- | --- |
| Perf runner, 7 arms × 10 blocks | gpu_ext | [run_table1_perf.py](../../workloads/llama.cpp/observability_overhead/revision-rq4/run_table1_perf.py) |
| Core helpers (paths, tool specs, env) | gpu_ext | [run_observability_overhead.py](../../workloads/llama.cpp/observability_overhead/run_observability_overhead.py) |
| Reused command helpers | gpu_ext | [run_revision_rq4.py](../../workloads/llama.cpp/observability_overhead/revision-rq4/run_revision_rq4.py) supplies `private_probe` and `run_bench`; `run_arm_cell` is defined in the perf runner. The matched runner's lease/safety/verifier machinery is not used by the 06 perf cells. |
| gpubpf tool sources | bpftime (published) | [eunomia-bpf/bpftime](https://github.com/eunomia-bpf/bpftime) `example/gpu/{kernelretsnoop,threadhist,launchlate}`; copied + patched per run (SEC rewrite, [kernelretsnoop-phase-capacity.patch](../../workloads/llama.cpp/observability_overhead/revision-rq4/kernelretsnoop-phase-capacity.patch) for kernelretsnoop, Makefile include rewrite) |
| GPU runtime (agent, syscall-server, PTX passes, map 1503) | bpftime (published repo, local build untracked) | branch [revision/table1-host-plt-fix](https://github.com/eunomia-bpf/bpftime/tree/revision/table1-host-plt-fix); local READONLY checkout `/home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt` plus **untracked** build tree `build-table1-575-warp` |
| Fresh runtime build entrypoint | gpu_ext | [build_table1_runtime.sh](../../scripts/artifact/build_table1_runtime.sh) (first fresh build completed 2026-09-09 with exit 0; runtime execution remains untested; see "Fresh runtime build") |
| NVBit arm | gpu_ext adapters + NVBit 1.8 core | adapter [nvbit_adapters/observability](../../workloads/llama.cpp/observability_overhead/revision-rq4/nvbit_adapters/README.md) (tracked); NVBit 1.8 x86_64 release under `revision-rq4/deps/nvbit_release_x86_64` (gitignored; obtain the official release separately) |
| onevalue GPU-array candidate | gpu_ext | [onevalue-array-candidate](../../workloads/llama.cpp/observability_overhead/revision-rq4/onevalue-array-candidate/README.md) @ `99b66423` (record `7911b79a`); built into a temporary staging dir |
| llama.cpp binary + model | submodule + local build | `workloads/llama.cpp/llama.cpp` (eunomia-bpf fork); the measured commit is recorded in every cell's nested stdout JSON (`build_commit 26836b27` / `build_number 7102`, resolving to `26836b27ae1ec9d6e94c6b56306cca75c7e86814`), see [table1-llama-source.md](table1-llama-source.md); `build-ptx-1b/` tree and `models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (668 MB) not in git |

## Actual runner path and build entrypoints

- 70-cell Table 1 ([results-table1-warp-plt-575-06](../../workloads/llama.cpp/observability_overhead/revision-rq4/results-table1-warp-plt-575-06/README.md)):
  `run_table1_perf.py` rotates 7 arms per block (baseline + gpubpf/nvbit ×
  {kernelretsnoop, threadhist, launchlate}), 10 blocks, one attempt per cell;
  writes `{arm}_run_{block:02d}/`, `cells.json`, `summary.json`.
- Recorded launch shapes (from `cells.json` / `probe-execution.json`):
  - client: `taskset -c 8-15 /usr/bin/env [LD_PRELOAD=.../build-table1-575-warp/runtime/agent/libbpftime-agent.so] .../build-ptx-1b/bin/llama-bench -m <model> -r 1 -o json -p 512 -n 0 -ngl 99`
  - gpubpf loader: `taskset -c 8-15 /usr/bin/env LD_PRELOAD=.../runtime/syscall-server/libbpftime-syscall-server.so <tool_build>/<tool> [uprobe binary + symbol hint]`, one private `/dev/shm` segment per cell
  - nvbit client: `LD_PRELOAD=<nvbit build>/observability.so` plus `OBS_MODE=<tool>`, `OBS_TARGET_SYMBOL=<mangled rope_norm>`, `OBS_GPU_THREAD_COUNT=<slots>`
- Only gpubpf arms (and the onevalue candidate) preload the
  `build-table1-575-warp` runtime; the baseline preloads nothing, and NVBit
  arms load only their built `observability.so`.
- gpubpf tool build: copy `bpftime/example/gpu/<tool>` to
  `<output>/gpubpf_tool_build/<tool>`, rewrite the SEC target and Makefile
  `runtime/include` to the bpftime root, then `make` per tool.
- NVBit build: `make CXX=g++ NVBIT_ROOT=<deps/nvbit_release_x86_64>
  ARCH=sm_120` on a copy of `nvbit_adapters/observability`
  (`<output>/nvbit_tool_build/`); a logged build is retained in
  [retained-runner-worktree-20260907/build_nvbit.log](../../workloads/llama.cpp/observability_overhead/revision-rq4/retained-runner-worktree-20260907/build_nvbit.log).
- onevalue build: stage the candidate three levels below a bpftime root (the
  run used `/var/tmp/kernelretsnoop-onevalue-build.x1aICR/example/gpu/onevalue-array`
  with `third_party`/`runtime` linked to the hostfix-plt tree), rewrite the
  SEC target, `make -j8 CUDA_HOME=/usr/local/cuda-12.9
  BPFTOOL=/usr/local/sbin/bpftool`, and symlink `kernelretsnoop ->
  onevalue-array` so the unmodified runner can use it.
- Component-only preparation (no probes, no benchmarks). Run from the repo
  root, replacing the quoted placeholders:

  ```sh
  python3 workloads/llama.cpp/observability_overhead/revision-rq4/run_table1_perf.py \
      --build-only \
      --output-dir "NEW_OUTPUT_DIR" \
      --bpftime-root "BPFTIME_CHECKOUT" \
      --bpftime-build-dir "BPFTIME_CHECKOUT/build-table1-575-warp" \
      --nvbit-root "NVBIT_1_8_RELEASE_ROOT"
  ```

  `--nvbit-root` is optional (default: the pinned release under
  `revision-rq4/deps`). The command builds the three gpubpf example tools and
  the NVBit `observability.so` into `NEW_OUTPUT_DIR`, records a compact
  `NEW_OUTPUT_DIR/prepared-tools.json` (resolved tool paths, mode, roots),
  and returns before any probe loader, syscall server, or benchmark; it
  refuses an existing nonempty `NEW_OUTPUT_DIR` and deletes nothing. The first
  actual run is retained in
  [raw/table1-component-build-20260909.jqAD9c](../../workloads/llama.cpp/observability_overhead/revision-rq4/raw/table1-component-build-20260909.jqAD9c/README.md):
  all four components built on this host (exit zero) against the **existing**
  `build-table1-575-warp` tree and local NVBit 1.8 release. That record is
  existing-host component-build evidence, not a full fresh bpftime runtime or
  llama.cpp rebuild. Remaining gap: the command does **not** build the
  bpftime runtime (agent/syscall-server) or the llama.cpp binary/model.
- The `build-table1-575-warp` tree is Debug with `BPFTIME_ENABLE_CUDA_ATTACH=ON`,
  `BPFTIME_LLVM_JIT=ON`, `ENABLE_EBPF_VERIFIER=ON`, `BPFTIME_UBPF_JIT=ON`,
  CUDA 12.9, LLVM 15. Measured envs: kernelretsnoop timing 524288 slots ×
  44 entries, `BPFTIME_SHM_MEMORY_MB=1000`; threadhist 1048576 entries,
  SHM 200; launchlate 22528 entries.

## Fresh runtime build (documented procedure; first build completed)

[build_table1_runtime.sh](../../scripts/artifact/build_table1_runtime.sh)
builds the two runtime shared libraries the Table 1 gpubpf arms preload into a
brand-new output tree from the published source revision, with the documented
runtime configuration. It writes only under its output directory, refuses an
existing nonempty one, never touches the recorded checkout
(`bpftime-table1-hostfix-plt`) or the measured build tree
(`build-table1-575-warp`), and deletes nothing. From the repo root, replacing
the quoted placeholders:

```sh
bash scripts/artifact/build_table1_runtime.sh \
    --output-dir "FRESH_BUILD_OUTPUT" \
    --source-rev "eef8a51abaf2ca1f0cdca9f2425af3bd535da1b7" \
    --cuda-home "/usr/local/cuda-12.9" \
    --llvm-cmake-dir "/usr/lib/llvm-15/cmake" \
    --jobs "2"
```

(`--source-rev`, `--cuda-home`, `--llvm-cmake-dir`, and the compiler defaults
are the recorded values; relative paths are accepted and resolved to
absolute. `--bpftime-url` defaults to the public `eunomia-bpf/bpftime`
repository.)

- Source: `revision/table1-host-plt-fix` at the explicit revision above
  (branch tip verified 2026-09-09 via `git ls-remote`; not proven identical to
  the untracked tree that produced the measured build). The script clones a
  fresh checkout and detaches at that revision.
- Submodules: `git submodule update --init --recursive` checks out the
  gitlinks recorded in that revision — including the nested uBPF dependency
  `third_party/ubpf` @ `39872ad4` and the libbpf nested in
  `third_party/bpftool` @ `3f077472` (the same pins
  [preparation.json](../../workloads/llama.cpp/observability_overhead/revision-rq4/runtime-575/preparation.json)
  recorded for the measured preparation) — plus `vm/llvm-jit` @ `f66cafa` and
  the remaining gitlinks. The resulting commit IDs are recorded in the log
  and report; they are not gated.
- Configuration (from the recorded
  `build-table1-575-warp/CMakeCache.txt`): `CMAKE_BUILD_TYPE=Debug`,
  `BPFTIME_ENABLE_CUDA_ATTACH=ON` + `BPFTIME_CUDA_ROOT`,
  `BPFTIME_LLVM_JIT=ON`, `BPFTIME_UBPF_JIT=ON`, `ENABLE_EBPF_VERIFIER=ON`,
  `LLVM_DIR=/usr/lib/llvm-15/cmake` (LLVM 15), compilers `/usr/bin/cc` +
  `/usr/bin/c++` (GCC 13 host). The build report records resolved paths;
  this does not assert identity with every historical cache entry.
- Targets: only `bpftime-agent` and `bpftime-syscall-server` — the two
  libraries Table 1 preloads, together with their build dependencies.
- Outputs: `<out>/bpftime-source/`, `<out>/build/` (artifacts at in-tree
  `runtime/agent/libbpftime-agent.so` and
  `runtime/syscall-server/libbpftime-syscall-server.so` — the layout
  `run_table1_perf.py --bpftime-build-dir` expects), `<out>/build.log`
  (every command and its output), and `<out>/build-report.json` (checked-out
  commit IDs, configuration, artifact paths and sizes; emitted with `jq`, no
  file hashes).
- Host prerequisites: `git`, `cmake`, `jq`, CUDA 12.9, an LLVM 15 CMake
  config, and the package set from the published `installation.md` in the
  bpftime repo (boost, libelf, zlib, yaml-cpp, clang, pkg-config, GCC ≥ 12).
  Command/path checks report missing prerequisites; additional dependency
  errors surface in the configure/build log.
- Status: **build completed**, 2026-09-09, 11:50:45–11:53:47 PDT, exit zero.
  Agent and syscall-server sizes are 277,369,912 and 272,480,568 bytes.
  The fresh source checkout stayed clean. The
  [successful run record](../../workloads/llama.cpp/observability_overhead/revision-rq4/raw/runtime-published-build-20260909.IktLa7/README.md)
  retains the script, command, logs and build report. Runtime execution
  with these new libraries remains untested; no historical measurement
  was repeated or replaced.

## User-configurable paths and knobs

`run_table1_perf.py` flags; defaults are stale host paths that the recorded
runs overrode. `--model` (default the TinyLlama gguf), `--llama-bench`
(default `build-ptx-1b/bin/llama-bench`, falling back to `build/`),
`--bpftime-root` (actual: the hostfix-plt tree), `--bpftime-build-dir`
(actual: `build-table1-575-warp`; default `$BPFTIME_BUILD_DIR` or
`<root>/build`), `--target-symbol` (default mangled `rope_norm` exit symbol),
`--uprobe-binary` (recorded launchlate uprobe:
`build-ptx-1b/bin/libggml-cuda.so.0.9.4`), `--uprobe-symbol-hint`,
`--blocks 10`, `--probe-startup-s` (default 3; onevalue runs used 20 after
the cuInit-recursion fix), `--gpu-thread-count` (default 22528),
`--threadhist-gpu-thread-count 1048576`, `--n-gpu-layers 99`, `--uvm`,
`--no-warmup`, `--build-only` (stop after the component build; see the
entrypoint above), and `--nvbit-root` (NVBit release root override; default
`revision-rq4/deps/nvbit_release_x86_64`). The model is a download, not in git.

## Measured shape and statistics (preserved)

- Table 1-06, 70 cells, all rc 0 ([report](../../workloads/llama.cpp/observability_overhead/revision-rq4/results-table1-warp-plt-575-06/README.md)):
  baseline mean 37586.322536 tok/s; gpubpf kernelretsnoop 3493.6654284
  (90.7050859%), threadhist 36471.0324999 (2.9653081%), launchlate
  37502.5450369 (0.2208152%); NVBit kernelretsnoop 142.4363463
  (99.6210304%), threadhist 33694.9158263 (10.3501103%), launchlate
  34279.7492721 (8.7959164%). Only the 10 gpubpf_launchlate cells were
  refilled after the uprobe binary path fix.
- onevalue GPU-array, 20 cells ([report](../../workloads/llama.cpp/observability_overhead/revision-rq4/results-onevalue-array-bootstrap-575-20260907/README.md)):
  ten-pair mean 5.400619% (median 5.391675%, range 4.017536–6.704966%);
  baseline 38156.366833 / array 36095.211373 tok/s. First five pairs:
  baseline 37979.256081 / array 35861.535104 tok/s, **5.572554%** — the
  paper bar; a retained subset, not a second experiment. Per run: 720896
  32-byte events, 16384 active warps, 0 reported overflow; 23199768-byte
  value; final bulk lookup mean 10.379343 ms (5 pairs) / 10.597817 ms
  (10 pairs), outside the prefill timing window.
- P40 historical: gpubpf/NVBit 8/85%, 3/87%, 14/93%
  ([p40-submitted-table1.json](../../workloads/llama.cpp/observability_overhead/p40-submitted-table1.json)).
- The full-record SoA device-buffer experiments are separate and are not
  Table 1 replacements.

## Recorded source revisions

- gpu_ext: onevalue candidate `99b66423` (build/measurement record),
  five-pair record `7911b79a`. The 06 `cells.json` embeds no gpu_ext source
  revision; it records absolute launch paths and the llama build version.
- bpftime: the branch tip was verified 2026-09-09 by `git ls-remote` of the
  public repo as `eef8a51abaf2ca1f0cdca9f2425af3bd535da1b7`. That is the
  current branch HEAD, **not** evidence that the historically measured build
  used this exact commit; the measured build is the untracked local tree.
  The recorded overlays ([runtime-575.patch](../../workloads/llama.cpp/observability_overhead/revision-rq4/runtime-575/runtime-575.patch),
  [late-bootstrap-target-filter.patch](../../workloads/llama.cpp/observability_overhead/revision-rq4/runtime-575/late-bootstrap-target-filter.patch)
  on base `d6316fa`) predate the later warp-coalesced / PLT-attach /
  array-related commits on the branch and do not fully describe the
  measured tree. The current local `vm/llvm-jit` revision is `f66cafa`.
- llama.cpp: all 70 original and 20 GPU-array benchmark records report
  `build_commit=26836b27`, `build_number=7102` in nested stdout JSON;
  the full source revision is `26836b27ae1ec9d6e94c6b56306cca75c7e86814`.
  See the [source-version inventory](table1-llama-source.md). This identifies
  the reported Git version, not every build flag or uncommitted edit.
- NVBit: official 1.8 x86_64 release; no revision recorded beyond the name.

## CPU figure reproduction (working) versus full runtime

- CPU-only, verified on a fresh depth-1 clone (`d8f78200`, 2026-09-09):
  after `git submodule update --init --depth 1 -- docs/paper` (pinned
  `62f5ed1`), run `python3 -B scripts/artifact/reproduce_figures.py --out-dir NEW_DIR`.
  The [wrapper](../../scripts/artifact/reproduce_figures.py) reconstructs the paper-selected five-block subset
  from the onevalue `cells.json`, runs the unchanged
  [plot_obs_with_array.py](../paper/tex-revision/img/results-raw/revision/plot_obs_with_array.py)
  and `plot_port_panels.py`, and writes only to `NEW_DIR`; no GPU, model
  cache, or workload build. Tested with Python 3.12.3, Matplotlib 3.6.3,
  NumPy 1.26.4. Paper data input:
  [obs-with-array-data.json](../paper/tex-revision/img/results-raw/revision/obs-with-array-data.json).
- The 70-cell bars predate the wrapper:
  [plot_obs_overhead_bars.py](../../workloads/llama.cpp/observability_overhead/plot_obs_overhead_bars.py)
  reads `p40-submitted-table1.json` plus the 06 `cells.json`.
- Full runtime reproduction (fresh OS/GPU run from a fresh checkout) has
  **not** been established; nothing above claims that status.

## Fresh-checkout gaps (established)

1. **Fresh runtime execution remains open**: the public-source bpftime
   library build now succeeds using the procedure above. Installation on
   a fresh OS and probe/benchmark execution with those newly built libraries
   have not been established. The historical measured runtime remains a
   separate untracked build; a new successful build does not rewrite its history.
2. **NVBit core not vendored here**: `revision-rq4/deps/nvbit_release_x86_64`
   is gitignored; obtain the official NVBit 1.8 x86_64 release separately.
3. **Workload binary + model not published**: `build-ptx-1b/` (cmake flags
   in the [harness README](../../workloads/llama.cpp/observability_overhead/README.md))
   and the 668 MB TinyLlama gguf remain regenerable/downloaded inputs outside
   Git. The reported llama.cpp version is now recovered; its fresh build
   and model-preparation procedure are separate work, not a source-version gap.
4. **Ephemeral staging paths**: the runner worktree
   `/home/yunwei37/workspace/gpu/gpu_ext-table1-runner` (06 cwd; partial log
   mirror under `retained-runner-worktree-20260907/`) and
   `/var/tmp/kernelretsnoop-onevalue-build.x1aICR/` (onevalue build, no
   build log published) do not exist on a fresh host; both are regenerable
   from items 1–3.
5. **Host prerequisites**: RTX 5090 sm_120, driver 575.57.08, clang with a
   BPF target, bpftool, `taskset`, `cuobjdump`, `CUDA_HOME=/usr/local/cuda-12.9`;
   the matched runner additionally expects the two `/tmp/gpubpf-revision-*`
   lease files to exist as regular files (the perf runner does not take them).

## Retained source / raw records

- [06 campaign](../../workloads/llama.cpp/observability_overhead/revision-rq4/results-table1-warp-plt-575-06/README.md) (`cells.json`, `summary.json`, per-cell `probe-execution.json`)
- [component-only build record](../../workloads/llama.cpp/observability_overhead/revision-rq4/raw/table1-component-build-20260909.jqAD9c/README.md) (`build.sh`, per-component build logs, `prepared-tools.json`, runner source captures)
- [fresh runtime build record](../../workloads/llama.cpp/observability_overhead/revision-rq4/raw/runtime-published-build-20260909.IktLa7/README.md) (public-source checkout, two runtime libraries, commands and logs)
- [onevalue campaign](../../workloads/llama.cpp/observability_overhead/revision-rq4/results-onevalue-array-bootstrap-575-20260907/README.md) (`cells.json`, `summary.json`, per-cell `probe.log`/`agent.log`)
- [onevalue build/measurement record](../../workloads/llama.cpp/observability_overhead/revision-rq4/onevalue-array-candidate/build-and-measurement.md), [runtime facts](../../workloads/llama.cpp/observability_overhead/revision-rq4/device-array-runtime-notes.md), [NVBit adapters](../../workloads/llama.cpp/observability_overhead/revision-rq4/nvbit_adapters/README.md), [runtime-575 overlays](../../workloads/llama.cpp/observability_overhead/revision-rq4/runtime-575/README.md), [harness README](../../workloads/llama.cpp/observability_overhead/README.md)
