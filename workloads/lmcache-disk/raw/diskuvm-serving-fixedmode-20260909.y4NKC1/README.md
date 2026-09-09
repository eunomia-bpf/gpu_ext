# Disk/UVM fdinfo repair bring-up: two completed failed-path cells

## Final status, 2026-09-09 03:35:28 PDT

Two cells complete with server exit zero and no warm HTTP failures:
stock 72.705982 token/s, native 62.886701 token/s. These are debugging
observations, **not a paired implementation-performance result**.
No BPF cell was launched.

Stock reaches the direct-I/O descriptor inspection and incorrectly rejects
flags 02140002: the helper hardcodes mask 020000, whereas installed
os.O_DIRECT is 040000. The actual descriptor includes O_DIRECT. This is
an adapter bug, not absent direct-I/O support.

By the time the deferred stop request was delivered, native had already
started. Root's mask correction reached that new worker during startup;
its later log instead exposes ctypes.ArgumentError: argument 2:
TypeError: wrong type. The two processes therefore do not use identical
preparation code and must not be paired. Root reproduced the ctypes
failure directly: memmove does not accept this bytearray source, while
bytes(buf) works. Commit 8a352587 uses os.O_DIRECT and repairs both ioctl
readback copies. It preserves the same disk/UVM policy and ABI.

The native dump reports prepared=0, restored=0, error=48 and
stock_fallback=244. Stock's per-process dump is absent even with drain
mode; it is not inferred to be successful. Both server logs show the
official mode=drain timeout=30s setting. Thus the normal-exit parameter
was exercised, but is not by itself a promise that every dump arrives.

The runner finishes the active native cell, then honors its existing
DeferredStop handler between cells (exit 3). No active client or local
OpenCode session is terminated by root. The lifecycle restores the saved
UVM and loaders 479952/479953, RESTORATION_OK=1. The next repaired run is
separate: diskuvm-serving-iobuffer-20260909.aodtQv, not a resumption that
mixes these failed treatments with successful transport measurements.
All raw outputs and both original numbers remain.

## Initial start record (historical, retained)

Started 2026-09-09 03:30:08 PDT. This is a new five-block
stock/native/BPF campaign using the same real workload, fixed prompts,
62502 ns/token existing calibration and warm throughput metric. It is
not a resume of the prior fallback measurements: the earlier ten
completed cells remain in diskuvm-serving-20260909.9uBMtq unchanged.

Source f6dfb849 fixes the invalid Python fdinfo open mode and logs actual
preparation failures. Source a26c57a5 forwards the official vLLM
--shutdown-timeout 30 only for disk-UVM runs; the default remains
unchanged for other callers. This permits normal worker teardown
instead of vLLM's default immediate abort. It is not a benchmark
deadline or an OpenCode timeout. Startup, cold population, barriers
and shutdown remain outside the warm throughput timing. Python syntax
compilation passed, and the repaired fdinfo helper successfully reads
the installed Python process's fdinfo.

The previously built fault helper and disk/UVM driver are reused,
with no rebuild or new paper baseline. run-serving.sh records the
exact command and existing cleanup/restoration trap. Both shared
leases cover the campaign. Active GDS/KV loaders are 471883/471884.
The prior UVM module will be restored after the owned servers and
loaders exit; restoration is pending while this run is active.

No performance result or successful restore count is claimed by this
start record. Future result.json and the existing counter/log output
will establish what actually ran. Missing counters are not a gate
that discards a throughput observation; real transport errors will
be explained rather than called successful offload. No manuscript
is edited and no completed old-treatment cell is overwritten.
