# Workspace cleanup and retained measurements — 2026-09-07 UTC

The user requests collection/publication of data and reports, plus removal of
unused worktrees. Active experiments and other tasks' changes are retained.

## Completed cleanup

| Removed worktree under `/home/yunwei37/workspace/gpu/` | Retained Git branch / revision | Recovery |
| --- | --- | --- |
| `gpu_ext-table1-device-array-fast` (about 3.8 GiB) | `revision/table1-device-array-fast`, `1ba5316a` | Recreate from retained branch; tracked tree was clean and there were no unique untracked/ignored files. |
| `gpu_ext-table1-runner` (about 3.9 GiB) | `revision/table1-pure-runner`, `1ba5316a` | Recreate from branch, then restore the local archive below. |

Both were removed with normal `git worktree remove`, without force. No process
working directory referenced either checkout at inspection. Branches were not
deleted. Their completed measurements are not rerun.

The runner's 316 ignored file entries were archived before removal in
`/home/yunwei37/workspace/gpu/.worktree-archives/20260907/table1-runner-ignored.tar.gz`
(about 24 MiB). This preserves old logs, local tool binaries and build outputs.
Its 17 text logs (9,467,985 bytes) are also collected under
[`retained-runner-worktree-20260907`](../workloads/llama.cpp/observability_overhead/revision-rq4/retained-runner-worktree-20260907/).
They accompany the existing `results-table1-warp-plt-575-03` records, not new
performance samples. All previous figures and numbers remain retained.

Recovery, if needed:

```sh
git -C /home/yunwei37/workspace/gpu/gpu_ext worktree add /home/yunwei37/workspace/gpu/gpu_ext-table1-runner revision/table1-pure-runner
tar -xzf /home/yunwei37/workspace/gpu/.worktree-archives/20260907/table1-runner-ignored.tar.gz -C /home/yunwei37/workspace/gpu/gpu_ext-table1-runner
```

## Measurement collection and active work

LMCache live-feedback's complete 15-cell raw data, paired analysis, and
[report](../workloads/lmcache-disk/results-575-gds-mixed-live-feedback-20260907.md)
are pushed in main `674bf3d2` and development `5ef72ba5`. The adverse BPF/native
comparison is preserved. Regenerable KV payloads remain local, separate from
published measurement records.

The next opt-in executor waits for demand completion or delay-budget expiry
instead of periodically re-evaluating unchanged pending-demand state. It is
implementation work, not yet a measured improvement. Local OpenCode performs
the code changes and independent bottleneck analysis; the root collects,
measures, interprets, commits and pushes. No session is stopped for silence,
and at most three local sessions run concurrently.

Active main, LMCache GDS development, current 575 GDS driver, and bpftime
hostfix/PLT runtime worktrees remain in place. The unrelated bpftime main
review branch and Faiss changes are not touched. Older bpftime-575 and LMCache
page-governor checkouts contain local build/untracked files; their full payloads
are being archived before deciding their removal, not blindly deleted.
