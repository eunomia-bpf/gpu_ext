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

Do not call this checkpoint a complete runnable campaign or a performance
improvement. Existing adverse results and source contracts are unchanged.
