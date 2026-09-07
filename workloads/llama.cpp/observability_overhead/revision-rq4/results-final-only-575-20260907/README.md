# kernelretsnoop final-only collection: five paired blocks

RTX 5090, NVIDIA 575.57.08, TinyLlama-1.1B Q4_K_M, llama.cpp pp512,
one benchmark repetition with the existing warmup. Five alternating-order
baseline/tool pairs completed, all ten benchmark exits zero. This changes
only collector timing: the same ring-buffer producer records every event,
but the collector waits until target completion to drain. It is **not** the
pending GPU-local-array implementation.

| Configuration | Mean prefill token/s | Mean paired overhead |
| --- | ---: | ---: |
| Current baseline, 5 measurements | 38019.177433 | — |
| Final-only collector, 5 measurements | 3493.318028 | 90.811502% |
| Earlier continuous collector, 10 measurements | 3493.665428 | 90.705086% |
| Earlier NVBit, 10 measurements | 142.436346 | 99.621030% |

The last two rows are retained references from
[the complete Table 1 campaign](../results-table1-warp-plt-575-06/README.md),
not newly paired controls. Their baseline was 37586.322536 token/s. Thus the
0.1064 percentage-point overhead difference is not a paired regression
estimate. Absolute instrumented throughput is essentially unchanged across
these campaigns; suppressing continuous collection alone does not resolve
the approximately 91% overhead. The remaining candidate moves producer
storage to a GPU-local array, which this experiment does not test.

Final-only per-block throughput is 3538.163932, 3419.831264, 3597.908536,
3492.070585 and 3418.615824 token/s. Paired overhead is 90.693829%,
90.972601%, 90.510371%, 90.849198% and 91.031509%. Every tool run reports
720896 committed and collected 32-byte events, all in the final drain, with
zero full-ring drops. Collection remains part of total cell wall time;
prefill throughput does not include process setup, JIT compilation or final
drain. The roughly 83–84 s cell wall times include all of those stages and
must not be described as bulk-copy timings. No extra performance-admission
checks, retries, filtering or clock study were used.

## Implementation and reproduction

Source change: [opt-in collector patch](../kernelretsnoop-final-only.patch),
commit `2eb3afe2`. It leaves normal behavior selectable by omitting
`BPFTIME_KERNELRETSNOOP_FINAL_ONLY`. The source was prepared by the existing
`run_revision_rq4.prepare_tool_source` helper, including the existing compact
warp-record patch, from `/home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt`.
The collector was built with `make -j8 kernelretsnoop`. The runtime is the
same tree's `build-table1-575-warp`, not a rebuilt runtime.

The measurement used the existing Table 1 helpers directly, without invoking
their seven-arm campaign. From the gpu_ext root, the following is the
measurement invocation after preparing and building the patched collector
in a fresh output directory (replace `OUTPUT` accordingly):

```python
import os, sys
from pathlib import Path
p = Path('workloads/llama.cpp/observability_overhead/revision-rq4').resolve()
sys.path.insert(0, str(p))
import run_table1_perf as t
args = t.parse_args([
    '--bpftime-root', '/home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt',
    '--bpftime-build-dir', '/home/yunwei37/workspace/gpu/bpftime-table1-hostfix-plt/build-table1-575-warp',
    '--blocks', '5'])
args.timeout_s = None
root = Path('OUTPUT').resolve()
t.ARMS = ('baseline', 'gpubpf_kernelretsnoop')
dirs = {'kernelretsnoop': root / 'gpubpf_tool_build/kernelretsnoop'}
cells = []
for block in range(1, 6):
    for arm in (t.ARMS if block % 2 else t.ARMS[::-1]):
        os.environ.pop('BPFTIME_KERNELRETSNOOP_FINAL_ONLY', None)
        if arm != 'baseline':
            os.environ['BPFTIME_KERNELRETSNOOP_FINAL_ONLY'] = '1'
        record = {'block': block, 'arm': arm,
                  'collection': 'final-only' if arm != 'baseline' else 'none'}
        try:
            record.update(t.run_arm_cell(arm, block, args, root, dirs, Path('/unused-nvbit')))
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        cells.append(record)
        t.write_records(root, cells)
```

[cells.json](cells.json) retains commands, output, exit status, throughput,
collector records and overhead for every measurement;
[summary.json](summary.json) contains the five-pair means. The per-cell logs
and collector execution records are retained alongside them. All earlier
P40/RTX 5090 numbers remain unchanged. Do not rerun this completed comparison.
