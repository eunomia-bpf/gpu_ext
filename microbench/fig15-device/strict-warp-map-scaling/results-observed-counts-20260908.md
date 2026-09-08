# Actual scalar callback counts: completed separate diagnostic

Completed 2026-09-08 on RTX 5090 / driver 575.57.08. Runtime `e61bdb3`,
same original `shared_update` BPF object used in the completed geometry/work
timings. Twelve attached diagnostic processes all finish successfully: six
settings, automatic execution off/on. Each process launches the target twice
with 128 threads/CTA and **zero warmup**.

| Setting (CTAs / arithmetic iterations) | Observed calls, off | Observed calls, on | Reduction |
|---|---:|---:|---:|
| 2 / 0 | 512 | 16 | 32x |
| 4 / 0 | 1024 | 32 | 32x |
| 8 / 0 | 2048 | 64 | 32x |
| 1 / 8 | 256 | 8 | 32x |
| 1 / 32 | 256 | 8 | 32x |
| 1 / 128 | 256 | 8 | 32x |

These are **device counter values actually copied to the host**, not values
computed from launch geometry. In the off arm, a scalar-site increment uses
the original call predicate; in the on arm it uses the selected warp-leader
predicate immediately before the actual call. The BPF object and its
eligibility analysis are unchanged. Native has no attached BPF program, so
its BPF invocation count is not applicable; no fabricated native measurement
is included.

The diagnostic-only live-context readback synchronizes the launched stream
and reads the module counter. Counters are cumulative since module load.
Both runtime/driver launch bookkeeping paths can report the same value:
e.g. off at 2 CTAs records `[256,256,512,512]`. The table uses the **last**
value (512), not the sum of repeated observations. No counter reset happens
between the two launches; each arm uses a fresh process/module.

## Connection to the completed timing experiment

The [180 geometry/work timing cells](results-geometry-work-20260908.md)
remain unchanged and counter-free. This diagnostic adds atomic increments
and host synchronization, so its elapsed times are **not performance data**.
It uses the same six shapes/work settings but a shorter launch schedule.

Observed callback counts scale with CTA count; automatic execution reduces
them by 32x in this benchmark. That reduction does not establish a speedup:
the original counter-free low-work timing cells are about 57–58% slower with
automatic execution. Added arithmetic reduces the relative timing difference
but does not change the observed invocation count at fixed geometry. The
evidence supports neither block-count-independent total work nor a general
block-count-independent total overhead claim.

## Artifacts and scope

Raw output and runner:
`raw/observed-counts-live-20260908.4UfFEw/`. `diagnostic.log` has one JSON
record per cell, including every observed counter value; each cell retains
application/agent/loader logs and process exit records. The runtime build
log is included. Extract the final observations without rerunning the GPU:

```sh
python3 - <<'PY'
import json
from pathlib import Path
p = Path('raw/observed-counts-live-20260908.4UfFEw/diagnostic.log')
for line in p.read_text().splitlines():
    row = json.loads(line)
    print(row['blocks'], row['work'], row['auto'], row['observed_counter_records'][-1])
PY
```

The earlier `raw/observed-counts-20260908.7LoqI4/` attempt remains: runtime
`7cfba18` created the counter but attempted readback only in a module
destructor that the process did not reach. No count was observed there.
Local Qwen authored the counter and live-context follow-up; root integrated,
built, ran and recorded them. All old timing results and failed prefixes
are retained. No manuscript file was changed.
