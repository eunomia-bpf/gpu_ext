# Same-object Table1 runner: source checkpoint

Local Qwen implemented the opt-in `--auto-warp-three-arm` mode in the
existing `run_table1_perf.py`. It selects one tool, builds its original
example source once without the legacy manual-warp capacity patch, and
uses baseline/auto_warp_off/auto_warp_on with per-cell environment settings.
The old seven-arm mode remains the default. Root mechanically normalized
erroneous leading indentation in the generated patch; no algorithm was
rewritten for that repair.

`python3 -m py_compile` exits zero. The two retained JSON outputs come
directly from `--dry-run` with and without `--auto-warp-three-arm`:
30 and70 planned cells respectively, ten rotating blocks. No build, probe
attachment, GPU operation or performance measurement occurred. These are
schedules, not successful measured cells.

Two source-integration items remain with the same OpenCode owner: relocate
the copied kernelretsnoop Makefile's runtime include without invoking the
manual-warp patch, and record aligned-word transport selection independently
of whole-program leader admission. The ordinary tool/runtime builds and
same-object performance campaign remain pending. The GPU is assigned to
the other task's Fig14 scan; no completed historical experiment is rerun.

Source inspection identified another required integration fix: the reused
`private_probe` supplies the legacy 1000 MiB shared-memory allocation and
44-entry environment setting, but the unpatched kernelretsnoop object has
80-byte records and 256 entries. Its loader does not consume that entry-count
environment setting. The runtime allocates `slots * (24 + 88 * 256) + 32`
bytes. Even retaining the inherited pp512 slot count of 524288 requires
11823743008 bytes (about 11.012 GiB), not the legacy 935329824 bytes.
This is allocation arithmetic from current source, not a measured failure
or throughput result. The original thread-index geometry and the loader/agent
environment must agree in the new mode; off/on must use the same object and
buffer layout. This fix must not reintroduce the manual-warp capacity patch
or change the historical seven-arm configuration. The same local runner
implementation session has received the source locations and required fix.

Do not call this checkpoint a complete runnable campaign or a performance
improvement. Existing adverse results and source contracts are unchanged.
