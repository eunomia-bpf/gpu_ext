# Non-manuscript workspace collection — 2026-09-24

This collection preserves local source changes and loose workspace records.
The separately versioned manuscript at `docs/paper/` is excluded, including
its submodule pointer. No GPU measurements were repeated for this cleanup.

## Dependency source recovery

[dependencies.json](dependencies.json) lists the five local checkouts, their
upstream URLs and base commits, and every changed or added source file with
its size. The patches under `dependencies/` are complete working-source
deltas relative to those base commits. They include the new TVM recorder and
MoE-Infinity queue/server files that an ordinary tracked-only diff omits.

These are standalone recovery snapshots. Apply a snapshot to its clean base
revision, not on top of the workload's existing patch series. The original
workload patch series and deployment checkouts remain in place.

For example, after checking out the XSched base commit from the inventory:

```sh
git apply --check /path/to/gpu_ext/docs/workspace-20260924/dependencies/xsched.patch
git apply /path/to/gpu_ext/docs/workspace-20260924/dependencies/xsched.patch
```

All five patches were checked and applied to the affected files read directly
from their base commits in temporary directories. Every resulting modified or
new file matched the local source exactly: 42 files total. This verifies source
recovery, not a new dependency build or runtime result. Local dependency clones
retain their modifications because the current experiment environment uses them.

## Other retained material

- [Loose XSched source](../../workloads/xsched/level2/native/retained-source-20260924/README.md):
  four files formerly at the repository root under `platforms/cuda/`, moved
  into their workload's dated source snapshot without changing their contents.
- [Storage design notes](workspace-files/disk-direction-ideas-20260906.md) and
  [cuFile log](workspace-files/cufile.log): copies of previously unversioned
  workspace-root files. The notes are historical proposals; the log includes
  compatible-mode notices and mount lookup errors, not proof of native GDS.
- Three full-record source directories are retained in the bpftime worktree
  on `revision/automatic-warp-execution`, with per-directory build notes and
  ignore rules for generated outputs. Historical directory names are kept.
- `bpftime/example/gpu/prefetch/.gitignore` excludes only copied build headers,
  objects and generated executables. The standalone libbpf dependency clones
  under the two llvm-jit checkouts use local Git excludes.

## Previously ignored experiment records

[The historical record archive](records/README.md) preserves logs, structured
outputs and analysis files omitted by broad ignore rules. Its inventory gives
each original repository-relative path and size. Build-tree copies, generated
headers, caches and the already published FineMoE split record are excluded.

## Publication

- `bpftime/review/pr-253-fix`: `6f42ce6`, generated-prefetch ignore rules and
  the previously local branch history.
- `bpftime/revision/automatic-warp-execution`: `e27d009`, three retained
  full-record experiment source directories.
- `bpftime/revision/sass-existing-application`: `fd976ea`, existing source
  published under its local branch name.
- `jax-xla-mapping/develop`: `5f37492b0f`, 16 previously local commits.
- Other active non-manuscript worktrees were already published and clean.

## Local-only assets

Downloaded model weights (`../models-kvpr/`), virtual environments, dependency
builds, binaries and caches remain local. The empty `../results/` directory
contains no records. Prior worktree payload archives remain under
`../.worktree-archives/20260907/`; their recovery procedure is documented in
[the earlier cleanup report](../workspace-cleanup-20260907.md). They are large
local backups, not source files to duplicate into Git. Workspace `AGENTS.md`
and local agent configuration stay in place.

The active top-level repositories/worktrees are gpu_ext, bpftime,
bpftime-auto-warp, bpftime-sass-existing-application,
bpftime-table1-hostfix-plt, gpu_ext-kernel-575-gds, jax-xla-mapping and
KVPR-artifact. Clean third-party clones require no new commits. Existing
historical branches are retained; this collection does not merge or delete them.
