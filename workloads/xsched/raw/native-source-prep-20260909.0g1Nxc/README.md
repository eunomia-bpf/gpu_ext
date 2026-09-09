# Native XSched: eight-patch source preparation

2026-09-09, 11:43:25 PDT: root ran the candidate preparation script using
relative source/output paths. It archived XSched Git revision
`f49289f0220931df78de948ed841ecbaf960a919` and applied all eight patches with
exit zero. It did not read dirty checkout files as source, build a binary,
load a driver or repeat a GPU cell.

[prepare.sh](prepare.sh) and [prepare.log](prepare.log) record the command
and output. The candidate script, changed meta-extension patch, final
Level-2 source-sync patch and eight per-patch logs are retained here.
The first six patches come from `workloads/xsched/level2/` at main base
`1cc9b966`. These captures describe this preparation stage; they are not
a standalone installer. The 22,650,880-byte upstream tar and extracted
source stay in the temporary build tree, outside Git.

All seven files compared by the script matched the successful native HAL
source: the two new shim files, module interception, shim dispatch,
kernel-command marshalling, Level-2 instrumentation and the CUDA queue.
Patch-context fuzz notices for original-entry control and launch-error
propagation remain visible in the logs. No rejects were produced.

## What remains before a fresh build

A broader [source comparison](source-differences.txt), excluding third-party
dependencies and generated logs, found two further differences in the
successful source: `preempt/.../async_xqueue.h` and `async_xqueue.cpp`.
They add two existing suspend/resume counters, increment them, and print
their values on queue destruction. They were present in the successful
run and must be represented explicitly when reproducing that source;
the seven-file match does not establish whole-tree parity.

Also, `git archive` does not populate Git submodules. In particular, the
prepared CLI11, cpp-httplib and jsoncpp directories lack the sources present
in the successful build. Fetching their pinned Git revisions is a remaining
build step, not permission to borrow the old compiled dependencies.

Both findings were sent to the local implementation task to complete the
recipe. This is source-preparation evidence, not a new native performance
result. The earlier [successful native cell](../level2-native-retabs-20260909.oJDCbU/README.md)
and all historical failed attempts remain unchanged.
