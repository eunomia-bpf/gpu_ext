# Disposable KV-cache cleanup, 2026-09-09 PDT

Completed results, logs, configuration and analysis were committed and pushed
in b8daf444 before removal. No vLLM/serving/XSched workers or compiler process
was running at the owner check. Both shared experiment leases serialize the
removal. The main repository has only its primary worktree; no worktree is
removed in this action.

Exactly the following 15 generated KV cache directories were removed with
rm -r and explicit resolved targets. Each occupied 1208352768 apparent bytes;
total apparent size was 18125291520 bytes (16.880493 GiB). They contained no
Git-tracked files. Only cache directories were deleted, not their sibling
results, server logs, model files, source snapshots or module artifacts.

- cells/block-00/position-0-stock/cache
- cells/block-00/position-1-native/cache
- cells/block-00/position-2-bpf/cache
- cells/block-01/position-0-native/cache
- cells/block-01/position-1-bpf/cache
- cells/block-01/position-2-stock/cache
- cells/block-02/position-0-bpf/cache
- cells/block-02/position-1-stock/cache
- cells/block-02/position-2-native/cache
- cells/block-03/position-0-stock/cache
- cells/block-03/position-1-native/cache
- cells/block-03/position-2-bpf/cache
- cells/block-04/position-0-native/cache
- cells/block-04/position-1-bpf/cache
- cells/block-04/position-2-stock/cache

The deleted cache contents are not backed up in Git; they can be regenerated
by cold population. Reproducing old cache bytes is not an experiment gate.
Prior failed campaigns and all old published performance numbers remain.
