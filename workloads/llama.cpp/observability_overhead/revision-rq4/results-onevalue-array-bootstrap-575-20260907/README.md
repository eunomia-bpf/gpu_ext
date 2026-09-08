# kernelretsnoop GPU-local array: five completed paired blocks

RTX 5090, NVIDIA 575.57.08, TinyLlama-1.1B Q4_K_M, llama.cpp pp512,
one repetition with the existing warmup, CUDA graphs disabled, CPU affinity
8–15. Five alternating-order baseline/tool pairs complete all ten benchmark
processes and five collectors with exit status zero. The source and build
are recorded in [the candidate implementation](../onevalue-array-candidate/build-and-measurement.md),
commit `99b66423`.

| Configuration | Mean prefill token/s | Mean paired overhead |
| --- | ---: | ---: |
| New baseline, 5 measurements | 37979.256081 | — |
| GPU-local array and final bulk readback, 5 measurements | 35861.535104 | **5.572554%** |
| Earlier continuous ring collector, 10 measurements | 3493.665428 | 90.705086% |
| Earlier final-only ring collector, 5 measurements | 3493.318028 | 90.811502% |
| Earlier NVBit, 10 measurements | 142.436346 | 99.621030% |

Earlier rows are retained references from their own campaigns and paired
baselines, **not newly interleaved controls**. The complete original
[three-tool Table 1](../results-table1-warp-plt-575-06/README.md), the
[final-only negative result](../results-final-only-575-20260907/README.md),
and all submitted P40 values remain unchanged. The historical P40
kernelretsnoop overhead was 8% for gpubpf and 85% for NVBit on its different
hardware/workload; similarity to that ratio is not an equivalence claim.

| Block | Baseline token/s | GPU-array token/s | Paired overhead | Final lookup ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 38032.055080 | 35945.547270 | 5.486182% | 10.157582 |
| 2 | 38090.425778 | 36001.316861 | 5.484604% | 10.322219 |
| 3 | 37633.571616 | 36121.629145 | 4.017536% | 10.959894 |
| 4 | 38120.819321 | 35768.962126 | 6.169482% | 10.244875 |
| 5 | 38019.408611 | 35470.220118 | 6.704966% | 10.212145 |

Overhead is `100 * (baseline - tool) / baseline` within each block;
the headline is the arithmetic mean of those five percentages. The paired
median is 5.486182%, with range 4.017536%–6.704966%. No samples are filtered.

The campaign uses the same warning-mode runtime configuration as the original
Table 1 run. Raw logs retain verifier warnings about map-pointer bounds; the
runtime executes the program despite those warnings. These are performance
measurements, **not evidence of strict verifier admission**. No admission
check was added or used to discard a measurement.

## What changed

The same warp-leader callback records every coordinate and timestamp into
GPU-local map type 1503 instead of the mapped-host ring. One map entry holds
16384 per-warp append counters and 44 full 32-byte events per warp; direct
GPU pointer stores avoid per-event host transfer. The collector performs one
whole-value lookup after target completion and sums the per-warp counters.
There is no global committed-counter hotspot, sampling, count-only substitute,
new compiler optimization pass, shared runtime edit or driver edit.

Each of the five runs reports **720896 stored events and nonzero timestamps**,
16384 active warp coordinates, and zero reported overflow or coordinate
mismatches. The allocated value is 23199768 bytes. Final host lookup averages
**10.379343 ms**, range 10.157582–10.959894 ms; it includes the existing map
lookup/copy path, not just PCIe DMA time.

The throughput improvement is scoped to prefill, whose timing excludes
startup/JIT and final collection. Delaying collection does not make its cost
disappear. The array is a finite-run buffer sized for this pp512 geometry and
serial target launches, not an unbounded or concurrent-kernel streaming
collector. The existing final-only ring experiment barely changed throughput,
whereas the GPU-local producer version does: these results support this
implementation-level improvement, not a universal overhead guarantee.

## Reproduction and retained attempts

Build and stage the candidate as described in its linked build record,
including the exact original target section and `kernelretsnoop` executable
symlink. Run the following from the gpu_ext root with a fresh `OUTPUT`:

```python
import sys
from pathlib import Path
p = Path('workloads/llama.cpp/observability_overhead/revision-rq4').resolve()
sys.path.insert(0, str(p))
import run_table1_perf as t
args = t.parse_args([
    '--bpftime-root', '/home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt',
    '--bpftime-build-dir', '/home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt/build-table1-575-warp',
    '--blocks', '5', '--probe-startup-s', '20'])
args.timeout_s = None
root = Path('OUTPUT').resolve()
root.mkdir(exist_ok=False)
t.ARMS = ('baseline', 'gpubpf_kernelretsnoop')
dirs = {'kernelretsnoop': Path('/var/tmp/kernelretsnoop-onevalue-build.x1aICR/example/gpu/onevalue-array')}
cells = []
for block in range(1, 6):
    for arm in (t.ARMS if block % 2 else t.ARMS[::-1]):
        record = {'block': block, 'arm': arm,
                  'storage': 'gpu-array-onevalue' if arm != 'baseline' else 'none',
                  'probe_startup_s': 20, 'syscall_server_bootstrap': True}
        try:
            record.update(t.run_arm_cell(arm, block, args, root, dirs, Path('/unused-nvbit')))
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        cells.append(record)
        t.write_records(root, cells)
```

The executed campaign first ran block 1, then resumed blocks 2–5 from its
stored records without repeating block 1. No correctness, precision, retry
or performance-admission gate was used. The old parser's ring-specific fields
remain -1 for this different collector; its actual diagnostic labels and
lookup time are preserved in every `probe.log`, not interpreted as ring data.

Two earlier attempts exposed a recursive CUDA initialization problem. Their
numeric benchmark outputs and collector failures remain in the linked
implementation record and separate raw directories; they are not counted
as instrumented performance. The initialization-order workaround fixes this
without a runtime or driver rebuild. This five-pair campaign is complete;
do not repeat these cells.

## Requested extension, September 8 UTC: queued, not yet measured

The experiment owner has queued only blocks 6–10 behind the active LMCache
jobs under the existing GPU and struct-ops locks, using the same staged
collector and bpftime build above. Existing blocks 1–5 and their raw files
are retained; the full ten-pair result is not available yet. This continues
the baseline/GPU-array comparison, not a new NVBit comparison.

The historical NVBit block-1 record in
`../results-table1-warp-plt-575-06/cells.json` reports 23068672 32-byte events,
whereas this GPU-array implementation records 720896 per-warp events.
The current NVBit `observe_exit` implementation emits a record for every
predicated thread, without a warp-leader filter, and uses a channel receiver;
the array uses warp-leader records and final bulk collection. Those are
different observation/collection configurations. Extending the repetition
count does not resolve this difference, and the old 99.621030% remains a
historical reference, not a newly matched NVBit baseline. A matched-granularity
comparison remains separate, unfinished implementation/measurement work.
