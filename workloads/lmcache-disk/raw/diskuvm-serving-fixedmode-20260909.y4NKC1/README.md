# Disk/UVM serving after the fdinfo-mode repair: running

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
