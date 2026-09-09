# Disk-UVM GPU-promotion in LMCache GDS serving — bring-up handoff

2026-09-09 PDT. RTX 5090 / CUDA 12.9 / 575.57.08. Integrates the existing
disk-UVM GPU-promotion primitive into the real LMCache KV path so completed
immutable KV chunks are restored to GPU through driver fault hydration + D2D
instead of the stock GDS read. All work is opt-in; default-off.

## Status update after this handoff

The OOM and incomplete runs described below are historical. The later
physical-reclaim driver repair (`95097e20`) completed all 15 cells in
[pXYN4F](../raw/diskuvm-physical-reclaim-20260909.pXYN4F/RESULTS.md).
The [current analysis](physical-reclaim-performance-analysis-20260909.md)
reports stock/native/BPF warm throughput of 73.574518 / 68.608032 /
65.422796 token/s: successful execution, but no throughput benefit for the
policy on this workload. The old failures and fixes below remain as the
development history; they are not the current completion status.

## Historical state at handoff

- **`aodtQv` ended 04:05:47 after 13 cells** (deferred stop). It was **not
  request-successful**: block0 native/BPF warm phase hit `EngineCore`
  `torch.OutOfMemoryError` (8.38 MiB free; process 30.71 GiB, torch 29.86 GiB,
  during preempted-request recompute) and block2 BPF 500'd. The earlier "all
  8/8192" was wrong; block2 native was failures 0 but output 5120 / observed
  6040, not 8192 (HTTP 200 can end early). Report on actual completed outputs.
- **EFVUGf (`d6be28a3`) reproduced the OOM.** Five observations were saved;
  the run and original-module restoration ended at 05:11:29 PDT. The report
  is [RESULTS.md](../raw/diskuvm-serving-reclaim-20260909.EFVUGf/RESULTS.md),
  published in `40937690`. First block: stock completed
  8192 tokens @ 70.067 t/s, but native and BPF both hit `EngineCore`
  `torch.OutOfMemoryError` (allocating 20 MiB, 6.38 MiB free; `server.log:862`).
  The first `restore` exception on both was CUDA `code 2`
  (`cudaErrorMemoryAllocation`). That return value does not localize the CUDA
  call or prove a sole root cause. The per-range re-OFFLOAD in this build is
  **not sufficient** to stop it (see bug 4).
- The earlier adapter fixes are complete, but the **OOM remains open**.
  Source inspection identifies physical GPU-chunk release as a missing step
  to investigate; migration and PTE unmapping alone do not establish that HBM
  has been returned to the allocator. Accumulation is a suspected contributor,
  not a measured sole cause. The bounded driver repair is **assigned separately**; it
  is not part of this patch.
- **Do not** treat earlier mixed failed-bringup cells as paired treatments; they
  are retained as fallback evidence only.
- Old 10 fallback cells are preserved. Counters are **descriptive, not a
  performance gate**.

## The transport (one paragraph)

On put, the installed `submit_put_task` captures `kv_bytes = memory_obj.get_size()`
before the GDS `ref_count_down`, and chains a prepare step to the immutable
completion callback: open the on-disk KV file (`file_offset=4096`, KV length =
`file_size - 4096`), `cudaMallocManaged` the **exact** `kv_bytes`, verify the
range start is a 2 MiB UVM block boundary, `UVM_DISK_BACKING_REGISTER` the whole
range, issue the real `OFFLOAD` writeback, await `on_disk`, then `SET_GPU_PROMOTION`.
On get, `load_gds` matches the prepared backing by key + `kv_bytes` and restores by
GPU fault hydration (`diskuvm_fault_read` from `libdiskuvm_fault.so`) + D2D into the
destination pointer, then re-offloads the range (intended to demote it; see bug 4
for why this does not yet free the physical HBM). Any miss, misfit, or restore
error falls back to the original GDS read. This is CPU-staged driver fault hydration — not NVMe→GPU P2P and not
automatic memory-pressure offload.

## Bring-up bugs found and fixed

1. `f6dfb849` — `_read_fdinfo_flags` used `open(path, "re")`; installed Python
   raises `ValueError: invalid mode: re`, which escaped `_open_backing_direct`
   after `os.open`, producing 48 failed prepares + 48 leaked FDs (the earlier
   "48 open FDs" observation was this, **not** retained backings). Fixed `re`→`r`.
2. `a26c57a5` — vLLM default `shutdown_timeout=0` let the APIServer force-kill the
   EngineCore mid-dump, leaving a 0-byte `disk-uvm-diagnostics.json.<pid>`.
   `server_argv`/`start_server` now forward an optional official
   `--shutdown-timeout`; `run_cell` passes `30` **only** for `disk_uvm` cells,
   `None` otherwise, so the `EngineCore.shutdown` teardown gets a grace window to
   emit the per-pid dump before the APIServer force-kill. This provides a grace
   window, not a guarantee of every dump or a clean engine exit (a separate OOM
   can still abort the process).
3. `8a352587` — `_is_direct_flag` masked against a hardcoded `0o20000` while this
   host's `os.O_DIRECT` is `0o40000`; real flags `02140002` do include O_DIRECT,
   so the check spuriously reported "not direct" and fell back. Now masks with
   `os.O_DIRECT`. Same commit also fixed the two ctypes `memmove` readback sites to
   pass `bytes(buf)` (a `bytearray` arg raised `TypeError`).
4. `d6be28a3` — attempted per-range residency cleanup. After synchronized D2D,
   re-issue the
   existing per-range `_offload` + `_wait_offload_done` under a per-backing lock;
   the driver's `!write_needed` path skips the durable file write and unmaps the
   CPU + GPU PTEs (`reclaim_ok`). This does not establish physical GPU-chunk
   release, and the subsequent EFVUGf run still OOMs. UVM registration and the
   on-disk copy stay reusable for the next restore.
5. `d6be28a3` — first-exception visibility: the first `restore` (fault/D2D)
   failure and the first re-offload/cleanup failure are each recorded, logged, and
   carried into the diagnostics JSON as `restore_error` / `cleanup_error` instead
   of being swallowed into a bare counter.

`f6dfb849` added logging of the actual prepare failure reason; the current patch
extends that to the read/restore and re-offload/cleanup paths.

## Reading the results

Per-cell, the runner writes `disk_uvm_diagnostics_path`
(`.../disk-uvm-diagnostics.json`); the engine dumps to
`<LMCACHE_DISK_UVM_DIAG_OUT>.<pid>` at `EngineCore.shutdown`. The runner's
`collect_disk_uvm_diagnostics` globs `<path>.*`, prefers a dump with
`retained_total > 0`, and records it under `result.json` → `disk_uvm_diagnostics`
(`expected`, `path`, `arrived`, `files[]`, `payload`).

Counter schema (per store, summed):
- `prepared` — backings successfully registered + offloaded + GPU-promoted.
- `restored` — reads served by the UVM restore path (fault hydration + D2D).
- `stock_fallback` — reads served by the original GDS read (no matching backing,
  or a restore error that fell through).
- `not_fitted` / `not_aligned` — prepare rejected (`kv_bytes` not a whole 2 MiB UVM
  block / managed-range start not 2 MiB block-aligned).
- `error` — a prepare or restore raised. Note a restore error increments **both**
  `error` and `stock_fallback`.

`prepared` > 0 means backings were registered + offloaded + GPU-promoted;
`restored` > 0 means some reads used the UVM restore path. A cell can show
`prepared`/`restored` > 0 **and still OOM or 500**, so counters and throughput
alone are not proof of a healthy transport — before claiming success, check
`warm_phase.failures`, the actual completed output tokens, and the
`restore_error` / `cleanup_error` strings in the collected JSON.

## Limitations

- `kv_bytes` must be a whole 2 MiB UVM block and the managed range start must be
  2 MiB block-aligned, else prepare falls back (`not_fitted`/`not_aligned`).
- `on_disk` is only true after a real OFFLOAD writeback; registration alone does
  not mark pages on-disk.
- The counters describe transport behavior; they are not a throughput gate.
- Throughput here compares the three KV-reclaim arms, all with the same
  opt-in transport; it does not isolate UVM restore speedup over stock GDS.

## Do not

- Re-run calibration or completed cells; reuse `62502 ns/token` from
  `raw/kv-reclaim-recompute-calibration-575-20260907-01/calibration.json`.
- Alter the preserved source snapshot for a completed campaign; complex repairs
  stay delegated and receive a new source revision and run directory.
- Claim transport success from throughput alone; read the per-cell counters.
