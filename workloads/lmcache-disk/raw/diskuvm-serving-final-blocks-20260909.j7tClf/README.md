# Disk/UVM serving: remaining three blocks running

Started 2026-09-09 03:07:17 PDT after the grouped-SoA performance
campaign released both shared leases. The exact command is in
run-serving.sh: --resume --blocks 5 against the existing
diskuvm-serving-20260909.9uBMtq/cells campaign. It reused all six
completed block-0/block-1 result.json files and starts at block 2.
The existing 62502 ns/token calibration is reused.

Adapter and runner source are ac02f17c; Python syntax compilation
passed before launch. This change only collects existing counters at
the actual EngineCore shutdown and reads per-PID files after exit.
It does not alter the policy or serving hot path. Sources remain
frozen for the running cells.

The first new BPF cell completes with server exit 0 and no HTTP
failures. Its worker-specific diagnostic file is zero bytes. The
server log shows vLLM mode=abort timeout=0s and immediate manager
force-kill during teardown. Thus the shutdown hook is reached but
the counter write can still be interrupted; this revision does not
yet provide complete restore/fallback attribution. Remaining
performance cells continue regardless. No previous timing is
discarded and no completed cell is repeated.

Active GDS/KV loaders are 431770/431771 for this run. The existing
EXIT trap will drain owned references, restore the saved original
UVM module and restart its loaders. Restoration is pending while
this campaign runs, not yet claimed complete. lifecycle.log and
runner.log are live local logs; results continue in the original
campaign directory. No manuscript is modified.
