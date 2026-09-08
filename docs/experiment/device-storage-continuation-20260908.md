# Device/storage continuation, 2026-09-08

This session does not edit the manuscript. Completed measurements and adverse
results remain intact. Non-trivial development runs through local OpenCode;
the root reviews, builds, measures, records, commits and pushes. At most three
local sessions run concurrently, without stopping them for silence.

## Latest result — disk GPU promotion complete at 12:11 PDT

[Five new full-read processes](../../workloads/lmcache-disk/results-disk-uvm-gpu-promotion-20260908.md)
complete and are published in `07141996` (driver `dea1fefc`, client
`e07b4d69`). Repeated GPU-read median is 0.250566 ms versus the earlier
CPU-first 5.407625 ms (21.582x ratio of separate-campaign medians), but
first GPU restoration rises from 170.601143 to 210.570854 ms (+23.429%).
These are primitive stage timings, not end-to-end LMCache policy gains
or randomized paired treatment effects. No earlier sample was repeated.
The saved original UVM and loaders 3407414/3407415 are restored.

The simple client flag/ioctl wiring was completed directly by root under
the user's allowance for bounded glue edits; the driver algorithm and repair
were authored by the local model. The model's client/packaging follow-up is
superseded by the real completed run, not a reason to rerun it. Mode-3
transport measurement and XSched execution remain unfinished. The native
metadata/main checkpoint compiles and reads the real cubins (`7e28c256`),
but still reports encoding pending and emits no runnable native array.

## Earlier implementation checkpoint — 11:50 PDT

Two follow-ups now build and their source commits are pushed:

- Disk GPU promotion: driver `dea1fefc`, with the local model's early-return
  repair integrated. The first UVM-only build failed on core module symbols;
  the established full module build then succeeded. No new module loaded.
  [Build records](../../workloads/lmcache-disk/raw/disk-uvm-gpu-promotion-20260908.bG4MBx/README.md)
  and the existing five-repeat wrapper adapted for the opt-in are retained.
  Qwen continues the existing client switch; measurements are still pending.
- Record-preserving mode-3 transport: bpftime `886b4ca`, both host reader and
  device writer, with regenerated embedded PTX. Agent and syscall-server
  builds succeeded. Main `fe199ede` retains the patch and build log. Qwen
  now prepares five rotating mode-2/mode-3 pairs, reusing the existing
  original per-thread probe, with normal collector completion rather than
  the old short post-client kill deadline. No new baseline/NVBit replay.

GLM's native XSched turn reached its model output-length limit naturally;
the same session resumed with only the incomplete metadata/main checkpoint
in scope. The native prefix/transfer repair follows that checkpoint. No
Level-2 performance measurement or new speedup is claimed. The three local
sessions remain active; no one was stopped for silence. This is implementation
progress, not an additional completed scientific result.

## Completed immediate requests

- [Disk-backed UVM restoration](../../workloads/lmcache-disk/results-disk-uvm-restore-20260908.md):
  five complete full-read runs, published in `cde80eeb`. At 256 MiB,
  median offload/CPU restore/GPU restore is 278.299/169.819/170.601 ms.
  This is a real file-backed same-address primitive, not a new end-to-end
  LMCache comparison or established GPU-direct P2P transfer. Restoration is
  CPU-first; subsequent GPU reads take 5.408 ms versus 0.337 ms initially.
- [Trampoline geometry/work timings](../../microbench/fig15-device/strict-warp-map-scaling/results-geometry-work-20260908.md):
  all 180 measurements published in `a0c5edf6`.
  Automatic execution is about 57–58% slower in the zero-work CTA sweep;
  larger compute reduces the relative difference. No old sample is removed.
- [Actual callback counts](../../microbench/fig15-device/strict-warp-map-scaling/results-observed-counts-20260908.md):
  all 12 separate diagnostics published in `8429a122`. Observed counts fall
  by 32x, which does not imply faster execution. Counts are not substituted
  for the counter-free timing measurements.

## Current GPU campaign

Completed follow-up: [all ten transport pairs](../../workloads/llama.cpp/observability_overhead/revision-rq4/results-original-ring-encoded-20260908.md)
now have numeric throughput and benchmark exit zero. Mean legacy/optimized
throughput is 34.514/194.059 token/s; the paired ratio median is 5.7702x.
Collector shutdown remains adverse (-9 in all twenty cells), and absolute
overhead is still about 99.5% for the optimized arm. The historical in-progress
checkpoint below is superseded for measurement completion, not for those
limitations. No completed cell was repeated.

The existing Table 1 runner compares original kernelretsnoop with legacy
transport versus aligned-word copy plus encoded-tail publication (mode 2).
The BPF object, record payload and 256-entry per-thread capacity are unchanged.
This is a transport optimization, not a new NVBit comparison or permission
to collapse distinct per-thread events into one observation.

The older 90.7051% kernelretsnoop result is a different probe configuration:
its retained cell records report 720,896 events, 16,384 coordinates and
44 entries per thread; its source patch selects one warp leader and emits
three coordinates plus a timestamp. The new original object emits all
thread coordinates/block dimensions and timestamp (80 bytes), with 256
entries per thread. The transport on/off pair preserves that original object;
its speedup must not be applied arithmetically to the older warp-level result.
Neither historical configuration or measurement is removed.

The first attempt completed ten baselines but failed all twenty loader starts
at the runtime's 10 GiB segment ceiling. Runtime `241872b` allows explicitly
requested segments up to 16 GiB, without changing the default. The resume
runs only those twenty failed attached cells, retaining the original ten
baselines. Baselines precede the repaired cells; it is not a fresh fully
interleaved thirty-cell campaign.

Records and resume command:
`workloads/llama.cpp/observability_overhead/revision-rq4/raw/table1-original-ring-encoded-resume-20260908.ceOL6M/`.
The initial attempt remains in sibling
`table1-original-ring-encoded-20260908.iVdbES/`.

Status at this checkpoint: in progress, not a final result. Numeric benchmark
exits and loader teardown errors are recorded separately. The runner's
`absent` setup-marker field must be read with its logging configuration:
`SPDLOG_LEVEL=warn` suppresses the runtime's informational transport marker.
An absent message does not by itself show whether optimization was disabled.
No added clock or logging gate blocks performance collection.

## Local implementation queue — updated 11:26 PDT

1. Qwen 27B, session `ses_f80dc8da2ffev9TpuxcuajuRMa`: the missing mode-3
   device writer, matching the already prepared host transpose patch.
   The read-only analysis session completed; its incorrect whole-program
   warp-leader interpretation was corrected in `0320ea1d`. Helper 25 is
   explicitly ineligible for that transformation. Mode 3 preserves every
   event and changes storage layout, not the BPF policy or sampling rate.
2. GLM, session `ses_f80c49c7cffebjDrdYLKy8X0ZC`: finish the XSched Level-2
   sm_120 native LDC adapter against actual NVCC-produced instructions.
   The equivalent branchless BPF guardian now builds: 49 BPF instructions,
   accepted by the existing exporter, assembled to sm_120 cubin and linked
   into the tool library (`01c4d4ba`). The first width/metadata parser piece
   also compiles (`866f5489`); native encoding integration is still unfinished.
   The isolated tool-actuator HAL now builds and installs (`4617565a`).
   The native prefix has a concrete compiler issue: its unconditional EXIT
   prevents fallthrough into the original kernel. GLM is repairing that
   source/extraction path; the matched native-C/BPF NVBit route is separate.
   Existing Level-1 results do not count as Level-2 results.
3. Qwen 27B, session `ses_f7e5a4134ffeup2HgHIRqLaiho`: prepare an isolated
   opt-in disk-restoration/GPU-promotion patch. CPU fault behavior and the
   measured CPU-first mode stay unchanged. No driver reload or performance
   claim before root integration and a new scoped run. The first candidate
   landed, but root found an existing early return bypasses its disk hydration
   when no resident source exists. The same session is fixing that path and
   adding the opt-in switch to the existing full-read client.

Qwen Next's earlier provider calls ended with actual HTTP 524 errors; the
current fallback therefore uses two Qwen 27B sessions and one GLM session,
not a fourth session. The mode-3 session's latest Next call also ended with
terminal HTTP 524; root resumed that same session on direct Qwen 27B
(`c60193d0`). GLM's LDC generation was resumed after a terminal
length limit, not interrupted for lack of output. No GPU timing is active
at this checkpoint; subsequent builds/runs use the shared experiment locks.
The saved original UVM is still loaded, not the promotion candidate.
Hummingbird host/device work is
still queued, without new implementation or measurement claims.
