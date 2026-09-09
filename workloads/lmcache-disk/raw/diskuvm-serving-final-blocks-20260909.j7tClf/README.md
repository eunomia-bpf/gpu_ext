# Disk/UVM serving continuation: stopped after an actual transport defect

## Final status, 2026-09-09 03:17:41 PDT

This continuation completed four new cells: block 2 (all three arms)
and block 3 stock. Including the six earlier cells, ten real serving
measurements completed, each with eight warm requests, 8192 generated
tokens, server exit zero and no recorded HTTP failures. They are
**fallback-path measurements, not evidence of successful disk/UVM
backing restoration**.

| Block | Stock token/s | Native token/s | BPF token/s |
| --- | ---: | ---: | ---: |
| 0, retained | 73.199047 | 66.625019 | 64.477420 |
| 1, retained | 74.594413 | 74.382520 | 69.010868 |
| 2, new | 70.168700 | 66.331103 | 66.911368 |
| 3, new partial block | 73.451493 | not run | not run |

The first complete EngineCore dump, block-2 native, reports prepared=0,
restored=0, error=48 and stock_fallback=244. Block-3 stock independently
reports prepared=0, restored=0, error=48 and stock_fallback=82.
Block-2 BPF/stock diagnostic files are empty because the parent kills
the worker with the default shutdown timeout of zero; those missing
counters are not interpreted as successful restoration.

Root found a direct implementation error in _read_fdinfo_flags:
Python open(..., "re") raises ValueError: invalid mode: 're'.
The exception escapes _open_backing_direct after os.open and before the
descriptor is returned, so the outer preparation fails and also leaks
that descriptor until process exit. This explains the previous
observation of 48 open KV file descriptors; it was not evidence of
48 retained UVM backings. All earlier numerical/raw records remain
unchanged, but their opt-in setting cannot be used to claim that the
new transport ran successfully. The defect is fixed in f6dfb849 using
mode "r", with actual preparation errors now logged. The installed
Python helper reads fdinfo successfully after this correction.

Root sent SIGINT only to the existing performance runner's DeferredStop
handler after confirming this error. It completed the active block-3
stock cell, saved it, then exited between cells before more fallback-only
work was launched. No in-flight client or OpenCode session was killed.
This is an error-driven stop, not a clock/correctness/performance gate.
The remaining old-treatment cells are not claimed complete and will
not be filled using a changed transport implementation.

The lifecycle exits 3 (requested deferred stop), RESTORATION_OK=1.
The saved original UVM is restored, with GDS/KV loaders 452940/452941
both reporting attached. GPU leases are released. All data, including
unfavorable results and partial/empty diagnostic files, are retained.
The repaired path will be measured in a separate campaign so old and
new implementations are not silently combined.

## Initial run record (historical, retained)

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
