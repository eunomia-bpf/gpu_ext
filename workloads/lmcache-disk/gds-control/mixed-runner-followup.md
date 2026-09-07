# Mixed runner: first execution and immediate follow-up

The first real execution is retained at
`../raw/gds-mixed-backend-575-20260906-01/` (three cells, default 4 reads and
6 writes of 24 MiB). FIFO recorded 10 submit decisions; native and BPF each
recorded 4 submit / 6 defer / 0 recompute. Every request completed and each
cell reported no cleanup error. Four reads per cell make the reported
nearest-rank p99 simply the sample maximum; this is first-run evidence.

Root observed these concrete issues in the first implementation. The next
local-model edit should fix them before using the enlarged workload's numbers:

1. Deferred write `submitted_s` is null in the native/BPF raw records. The
   adapter captures `_async_save_bytes_to_disk` before `wrap_submit_timing`
   replaces it. Install the timing wrapper first so both paths call it, and
   use the same monotonic origin for every request field.
2. The current `offer_s` is actual dispatch time, while the workload promises
   scheduled offered arrivals. Preserve scheduled offer and actual dispatch
   separately; report offered-to-completion latency including dispatch delay.
   Prepare all write buffers and finish their initialization before starting
   that common arrival schedule.
3. `t0_wall` currently passes a monotonic performance counter to `localtime`.
   Use wall time for the wall label and monotonic time only for durations.
4. `allocate_object` uses a global default 24 MiB even when `--object-mib`
   changes. Use the requested object size for allocation and metadata.
5. A subsequent five-block run with 64 reads, 96 writes and a 4096 MiB pool
   completed seven cells, then failed the last eight with CUDA OOM. The
   exception reports **28 GiB still allocated by PyTorch**, exactly seven
   pools, despite calling backend.close. The records and runner snapshot are
   in `../raw/gds-mixed-backend-dispatch-575-20260906-five-block/`.
   Prefer a fresh child process per cell to release the CUDA context and
   equalize allocator state; alternatively demonstrate actual pool release
   after dropping adapter/backend cycles. Do not use a smaller workload to
   conceal the retained pools. Preserve this failed campaign and collect a
   new full run after the fix.

The larger one-block run at `../raw/gds-mixed-backend-575-20260906-02/`
completed all 480 requests, with native and BPF each making 64 submit and 96
defer decisions; all submission timestamps are present after the ordering
fix. Its call-to-completion read p99 is FIFO 227.864 ms, native 133.470 ms,
and BPF 133.342 ms. It remains separate first-block evidence.

Keep these first records. Then collect a larger same-workload three-arm
comparison (enough reads that p99 is more informative than a four-sample max),
with bounded pool memory and identical offered traffic/warmup in every arm.
The root runs GPU experiments; local models implement the changes. These are
measurement fixes from actual records, not new clock/correctness gates.
