# fig13-fast analysis: fig13_fast_20260907_005958

Generated: 2026-09-07T08:36:08.043543+00:00

Source: `/home/yunwei37/workspace/gpu/gpu_ext/workloads/fig13-fast/results/fig13_fast_20260907_005958` (20 CSV data rows, all preserved)

Descriptive only: no pass/fail gates, no retries, no raw-row filtering, no invented metrics, no composite scores, no confidence intervals. Missing numbers are never replaced by zero and never turned into speedups.

## Run overview

- mode: `gpu`
- started_utc: `2026-09-07T07:59:58.070911+00:00`
- timeout_s: `0.0`
- kernel: `hotspot`
- size_factor: `0.6`
- iterations: `1`
- mem_policy: `/home/yunwei37/workspace/gpu/gpu_ext/extension/prefetch_eviction_pid -p HIGH_PID -P 20 -l LOW_PID -L 80`
- sched_policy: `/home/yunwei37/workspace/gpu/gpu_ext/extension/gpu_sched_set_timeslices -p uvmbench_high:1000000 -p uvmbench_low:200`
- engagement: `counters recorded as metadata only; never a gate`
- mem_tool: `/home/yunwei37/workspace/gpu/gpu_ext/extension/prefetch_eviction_pid`
- sched_tool: `/home/yunwei37/workspace/gpu/gpu_ext/extension/gpu_sched_set_timeslices`

- blocks present: ['0', '1', '2', '3', '4']
- arms present: ['baseline', 'memory_only', 'sched_only', 'combined']

Wall latency is each tenant's own SIGCONT-to-exit time in seconds; median kernel time is the uvmbench-reported median time in ms parsed from the tenant log.

## Per-arm metric summary

| arm | metric | rows | rc-ok numeric | rc-excluded numeric | value missing | median | min | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | high wall latency (s) | 5 | 5 | 0 | 0 | 56.547662 | 56.282068 | 56.850576 |
| baseline | low wall latency (s) | 5 | 5 | 0 | 0 | 56.595119 | 56.279050 | 56.844765 |
| baseline | high median kernel time (ms) | 5 | 5 | 0 | 0 | 15652 | 15568 | 15667.3 |
| baseline | low median kernel time (ms) | 5 | 5 | 0 | 0 | 15601.7 | 15566.7 | 15611.3 |
| memory_only | high wall latency (s) | 5 | 5 | 0 | 0 | 25.159860 | 24.660404 | 26.571034 |
| memory_only | low wall latency (s) | 5 | 5 | 0 | 0 | 26.852055 | 26.355107 | 28.229401 |
| memory_only | high median kernel time (ms) | 5 | 5 | 0 | 0 | 13581 | 12865.3 | 15059 |
| memory_only | low median kernel time (ms) | 5 | 5 | 0 | 0 | 413.498 | 413.48 | 413.625 |
| sched_only | high wall latency (s) | 5 | 5 | 0 | 0 | 3.938445 | 3.898265 | 3.964001 |
| sched_only | low wall latency (s) | 5 | 5 | 0 | 0 | 6.076045 | 6.018736 | 6.104975 |
| sched_only | high median kernel time (ms) | 5 | 5 | 0 | 0 | 413.441 | 413.281 | 413.562 |
| sched_only | low median kernel time (ms) | 5 | 5 | 0 | 0 | 413.52 | 413.352 | 413.639 |
| combined | high wall latency (s) | 5 | 5 | 0 | 0 | 3.546399 | 3.538059 | 3.562501 |
| combined | low wall latency (s) | 5 | 5 | 0 | 0 | 6.455460 | 6.437236 | 6.463168 |
| combined | high median kernel time (ms) | 5 | 5 | 0 | 0 | 413.408 | 413.32 | 413.587 |
| combined | low median kernel time (ms) | 5 | 5 | 0 | 0 | 413.505 | 413.451 | 413.582 |

performance medians and paired deltas include only row-role records whose tenant rc is 0 (successful completion); records with rc nonzero or missing keep their raw values and are listed as excluded with explicit reasons. rc/timeout/counter fields are classification metadata and never gate the report.

Values missing from a CSV cell are counted as missing and are never zeroed; rc-excluded numeric values keep their raw numbers in the JSON raw lists together with the explicit exclusion reasons.

## Block-paired deltas

Definition: pct = 100*(num/den - 1), computed within each block between the two arms' rows of that block; positive means num took longer than den on that metric

| pair | metric | blocks paired | median pct | min pct | max pct | unpaired blocks |
|---|---|---:|---:|---:|---:|---|
| memory_only/baseline | high wall latency (s) | 5 | -55.67% | -56.39% | -53.08% | none |
| memory_only/baseline | low wall latency (s) | 5 | -52.55% | -53.17% | -50.34% | none |
| memory_only/baseline | high median kernel time (ms) | 5 | -12.85% | -17.88% | -3.27% | none |
| memory_only/baseline | low median kernel time (ms) | 5 | -97.35% | -97.35% | -97.34% | none |
| sched_only/baseline | high wall latency (s) | 5 | -93.07% | -93.11% | -92.96% | none |
| sched_only/baseline | low wall latency (s) | 5 | -89.27% | -89.41% | -89.20% | none |
| sched_only/baseline | high median kernel time (ms) | 5 | -97.36% | -97.36% | -97.34% | none |
| sched_only/baseline | low median kernel time (ms) | 5 | -97.35% | -97.35% | -97.34% | none |
| combined/baseline | high wall latency (s) | 5 | -93.71% | -93.78% | -93.70% | none |
| combined/baseline | low wall latency (s) | 5 | -88.59% | -88.66% | -88.53% | none |
| combined/baseline | high median kernel time (ms) | 5 | -97.36% | -97.36% | -97.34% | none |
| combined/baseline | low median kernel time (ms) | 5 | -97.35% | -97.35% | -97.34% | none |
| combined/memory_only | high wall latency (s) | 5 | -85.93% | -86.68% | -85.55% | none |
| combined/memory_only | low wall latency (s) | 5 | -75.95% | -77.16% | -75.51% | none |
| combined/memory_only | high median kernel time (ms) | 5 | -96.95% | -97.25% | -96.79% | none |
| combined/memory_only | low median kernel time (ms) | 5 | -0.01% | -0.02% | +0.01% | none |
| combined/sched_only | high wall latency (s) | 5 | -10.17% | -10.53% | -8.61% | none |
| combined/sched_only | low wall latency (s) | 5 | +6.30% | +5.44% | +7.15% | none |
| combined/sched_only | high median kernel time (ms) | 5 | +0.01% | -0.02% | +0.01% | none |
| combined/sched_only | low median kernel time (ms) | 5 | -0.00% | -0.02% | +0.02% | none |

### Paired delta raw values

| pair | metric | block | num | den | pct |
|---|---|---:|---:|---:|---:|
| memory_only/baseline | high wall latency (s) | 0 | 25.228055 | 56.438986 | -55.30% |
| memory_only/baseline | high wall latency (s) | 1 | 25.159860 | 56.850576 | -55.74% |
| memory_only/baseline | high wall latency (s) | 2 | 26.571034 | 56.626556 | -53.08% |
| memory_only/baseline | high wall latency (s) | 3 | 24.660404 | 56.547662 | -56.39% |
| memory_only/baseline | high wall latency (s) | 4 | 24.948039 | 56.282068 | -55.67% |
| memory_only/baseline | low wall latency (s) | 0 | 26.917364 | 56.656841 | -52.49% |
| memory_only/baseline | low wall latency (s) | 1 | 26.852055 | 56.595119 | -52.55% |
| memory_only/baseline | low wall latency (s) | 2 | 28.229401 | 56.844765 | -50.34% |
| memory_only/baseline | low wall latency (s) | 3 | 26.355107 | 56.279050 | -53.17% |
| memory_only/baseline | low wall latency (s) | 4 | 26.640458 | 56.537156 | -52.88% |
| memory_only/baseline | high median kernel time (ms) | 0 | 13206.3 | 15662.6 | -15.68% |
| memory_only/baseline | high median kernel time (ms) | 1 | 13581 | 15584.1 | -12.85% |
| memory_only/baseline | high median kernel time (ms) | 2 | 12865.3 | 15667.3 | -17.88% |
| memory_only/baseline | high median kernel time (ms) | 3 | 15059 | 15568 | -3.27% |
| memory_only/baseline | high median kernel time (ms) | 4 | 13852.6 | 15652 | -11.50% |
| memory_only/baseline | low median kernel time (ms) | 0 | 413.49 | 15601.7 | -97.35% |
| memory_only/baseline | low median kernel time (ms) | 1 | 413.609 | 15566.7 | -97.34% |
| memory_only/baseline | low median kernel time (ms) | 2 | 413.498 | 15611.3 | -97.35% |
| memory_only/baseline | low median kernel time (ms) | 3 | 413.625 | 15600.5 | -97.35% |
| memory_only/baseline | low median kernel time (ms) | 4 | 413.48 | 15608.8 | -97.35% |
| sched_only/baseline | high wall latency (s) | 0 | 3.959820 | 56.438986 | -92.98% |
| sched_only/baseline | high wall latency (s) | 1 | 3.938445 | 56.850576 | -93.07% |
| sched_only/baseline | high wall latency (s) | 2 | 3.901500 | 56.626556 | -93.11% |
| sched_only/baseline | high wall latency (s) | 3 | 3.898265 | 56.547662 | -93.11% |
| sched_only/baseline | high wall latency (s) | 4 | 3.964001 | 56.282068 | -92.96% |
| sched_only/baseline | low wall latency (s) | 0 | 6.080187 | 56.656841 | -89.27% |
| sched_only/baseline | low wall latency (s) | 1 | 6.076045 | 56.595119 | -89.26% |
| sched_only/baseline | low wall latency (s) | 2 | 6.018736 | 56.844765 | -89.41% |
| sched_only/baseline | low wall latency (s) | 3 | 6.024867 | 56.279050 | -89.29% |
| sched_only/baseline | low wall latency (s) | 4 | 6.104975 | 56.537156 | -89.20% |
| sched_only/baseline | high median kernel time (ms) | 0 | 413.363 | 15662.6 | -97.36% |
| sched_only/baseline | high median kernel time (ms) | 1 | 413.544 | 15584.1 | -97.35% |
| sched_only/baseline | high median kernel time (ms) | 2 | 413.281 | 15667.3 | -97.36% |
| sched_only/baseline | high median kernel time (ms) | 3 | 413.562 | 15568 | -97.34% |
| sched_only/baseline | high median kernel time (ms) | 4 | 413.441 | 15652 | -97.36% |
| sched_only/baseline | low median kernel time (ms) | 0 | 413.518 | 15601.7 | -97.35% |
| sched_only/baseline | low median kernel time (ms) | 1 | 413.639 | 15566.7 | -97.34% |
| sched_only/baseline | low median kernel time (ms) | 2 | 413.352 | 15611.3 | -97.35% |
| sched_only/baseline | low median kernel time (ms) | 3 | 413.573 | 15600.5 | -97.35% |
| sched_only/baseline | low median kernel time (ms) | 4 | 413.52 | 15608.8 | -97.35% |
| combined/baseline | high wall latency (s) | 0 | 3.549469 | 56.438986 | -93.71% |
| combined/baseline | high wall latency (s) | 1 | 3.538059 | 56.850576 | -93.78% |
| combined/baseline | high wall latency (s) | 2 | 3.539000 | 56.626556 | -93.75% |
| combined/baseline | high wall latency (s) | 3 | 3.562501 | 56.547662 | -93.70% |
| combined/baseline | high wall latency (s) | 4 | 3.546399 | 56.282068 | -93.70% |
| combined/baseline | low wall latency (s) | 0 | 6.463168 | 56.656841 | -88.59% |
| combined/baseline | low wall latency (s) | 1 | 6.456694 | 56.595119 | -88.59% |
| combined/baseline | low wall latency (s) | 2 | 6.448404 | 56.844765 | -88.66% |
| combined/baseline | low wall latency (s) | 3 | 6.455460 | 56.279050 | -88.53% |
| combined/baseline | low wall latency (s) | 4 | 6.437236 | 56.537156 | -88.61% |
| combined/baseline | high median kernel time (ms) | 0 | 413.408 | 15662.6 | -97.36% |
| combined/baseline | high median kernel time (ms) | 1 | 413.587 | 15584.1 | -97.35% |
| combined/baseline | high median kernel time (ms) | 2 | 413.32 | 15667.3 | -97.36% |
| combined/baseline | high median kernel time (ms) | 3 | 413.466 | 15568 | -97.34% |
| combined/baseline | high median kernel time (ms) | 4 | 413.345 | 15652 | -97.36% |
| combined/baseline | low median kernel time (ms) | 0 | 413.453 | 15601.7 | -97.35% |
| combined/baseline | low median kernel time (ms) | 1 | 413.582 | 15566.7 | -97.34% |
| combined/baseline | low median kernel time (ms) | 2 | 413.451 | 15611.3 | -97.35% |
| combined/baseline | low median kernel time (ms) | 3 | 413.558 | 15600.5 | -97.35% |
| combined/baseline | low median kernel time (ms) | 4 | 413.505 | 15608.8 | -97.35% |
| combined/memory_only | high wall latency (s) | 0 | 3.549469 | 25.228055 | -85.93% |
| combined/memory_only | high wall latency (s) | 1 | 3.538059 | 25.159860 | -85.94% |
| combined/memory_only | high wall latency (s) | 2 | 3.539000 | 26.571034 | -86.68% |
| combined/memory_only | high wall latency (s) | 3 | 3.562501 | 24.660404 | -85.55% |
| combined/memory_only | high wall latency (s) | 4 | 3.546399 | 24.948039 | -85.78% |
| combined/memory_only | low wall latency (s) | 0 | 6.463168 | 26.917364 | -75.99% |
| combined/memory_only | low wall latency (s) | 1 | 6.456694 | 26.852055 | -75.95% |
| combined/memory_only | low wall latency (s) | 2 | 6.448404 | 28.229401 | -77.16% |
| combined/memory_only | low wall latency (s) | 3 | 6.455460 | 26.355107 | -75.51% |
| combined/memory_only | low wall latency (s) | 4 | 6.437236 | 26.640458 | -75.84% |
| combined/memory_only | high median kernel time (ms) | 0 | 413.408 | 13206.3 | -96.87% |
| combined/memory_only | high median kernel time (ms) | 1 | 413.587 | 13581 | -96.95% |
| combined/memory_only | high median kernel time (ms) | 2 | 413.32 | 12865.3 | -96.79% |
| combined/memory_only | high median kernel time (ms) | 3 | 413.466 | 15059 | -97.25% |
| combined/memory_only | high median kernel time (ms) | 4 | 413.345 | 13852.6 | -97.02% |
| combined/memory_only | low median kernel time (ms) | 0 | 413.453 | 413.49 | -0.01% |
| combined/memory_only | low median kernel time (ms) | 1 | 413.582 | 413.609 | -0.01% |
| combined/memory_only | low median kernel time (ms) | 2 | 413.451 | 413.498 | -0.01% |
| combined/memory_only | low median kernel time (ms) | 3 | 413.558 | 413.625 | -0.02% |
| combined/memory_only | low median kernel time (ms) | 4 | 413.505 | 413.48 | +0.01% |
| combined/sched_only | high wall latency (s) | 0 | 3.549469 | 3.959820 | -10.36% |
| combined/sched_only | high wall latency (s) | 1 | 3.538059 | 3.938445 | -10.17% |
| combined/sched_only | high wall latency (s) | 2 | 3.539000 | 3.901500 | -9.29% |
| combined/sched_only | high wall latency (s) | 3 | 3.562501 | 3.898265 | -8.61% |
| combined/sched_only | high wall latency (s) | 4 | 3.546399 | 3.964001 | -10.53% |
| combined/sched_only | low wall latency (s) | 0 | 6.463168 | 6.080187 | +6.30% |
| combined/sched_only | low wall latency (s) | 1 | 6.456694 | 6.076045 | +6.26% |
| combined/sched_only | low wall latency (s) | 2 | 6.448404 | 6.018736 | +7.14% |
| combined/sched_only | low wall latency (s) | 3 | 6.455460 | 6.024867 | +7.15% |
| combined/sched_only | low wall latency (s) | 4 | 6.437236 | 6.104975 | +5.44% |
| combined/sched_only | high median kernel time (ms) | 0 | 413.408 | 413.363 | +0.01% |
| combined/sched_only | high median kernel time (ms) | 1 | 413.587 | 413.544 | +0.01% |
| combined/sched_only | high median kernel time (ms) | 2 | 413.32 | 413.281 | +0.01% |
| combined/sched_only | high median kernel time (ms) | 3 | 413.466 | 413.562 | -0.02% |
| combined/sched_only | high median kernel time (ms) | 4 | 413.345 | 413.441 | -0.02% |
| combined/sched_only | low median kernel time (ms) | 0 | 413.453 | 413.518 | -0.02% |
| combined/sched_only | low median kernel time (ms) | 1 | 413.582 | 413.639 | -0.01% |
| combined/sched_only | low median kernel time (ms) | 2 | 413.451 | 413.352 | +0.02% |
| combined/sched_only | low median kernel time (ms) | 3 | 413.558 | 413.573 | -0.00% |
| combined/sched_only | low median kernel time (ms) | 4 | 413.505 | 413.52 | -0.00% |

## Per-row status and failures

| row | block | arm | high_rc | low_rc | high_timeout | low_timeout | rc_sched_tool | rc_mem_tool | notes | meta_notes |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 2 | 0 | baseline | 0 | 0 | 0 | 0 | n/a | n/a | - | - |
| 3 | 0 | memory_only | 0 | 0 | 0 | 0 | n/a | 0 | - | - |
| 4 | 0 | sched_only | 0 | 0 | 0 | 0 | 0 | n/a | - | - |
| 5 | 0 | combined | 0 | 0 | 0 | 0 | 0 | 0 | - | - |
| 6 | 1 | memory_only | 0 | 0 | 0 | 0 | n/a | 0 | - | - |
| 7 | 1 | sched_only | 0 | 0 | 0 | 0 | 0 | n/a | - | - |
| 8 | 1 | combined | 0 | 0 | 0 | 0 | 0 | 0 | - | - |
| 9 | 1 | baseline | 0 | 0 | 0 | 0 | n/a | n/a | - | - |
| 10 | 2 | sched_only | 0 | 0 | 0 | 0 | 0 | n/a | - | - |
| 11 | 2 | combined | 0 | 0 | 0 | 0 | 0 | 0 | - | - |
| 12 | 2 | baseline | 0 | 0 | 0 | 0 | n/a | n/a | - | - |
| 13 | 2 | memory_only | 0 | 0 | 0 | 0 | n/a | 0 | - | - |
| 14 | 3 | combined | 0 | 0 | 0 | 0 | 0 | 0 | - | - |
| 15 | 3 | baseline | 0 | 0 | 0 | 0 | n/a | n/a | - | - |
| 16 | 3 | memory_only | 0 | 0 | 0 | 0 | n/a | 0 | - | - |
| 17 | 3 | sched_only | 0 | 0 | 0 | 0 | 0 | n/a | - | - |
| 18 | 4 | baseline | 0 | 0 | 0 | 0 | n/a | n/a | - | - |
| 19 | 4 | memory_only | 0 | 0 | 0 | 0 | n/a | 0 | - | - |
| 20 | 4 | sched_only | 0 | 0 | 0 | 0 | 0 | n/a | - | - |
| 21 | 4 | combined | 0 | 0 | 0 | 0 | 0 | 0 | - | - |

rc_sched_tool / rc_mem_tool come from meta.json (n/a when the arm does not use that tool or meta.json is absent); non-integer or missing values stay as reported by the harness.

## Policy tool counters (metadata only)

mem: values in CSV meta columns were recorded by run_fig13_fast.py parse_mem_meta via re.search, which captures the FIRST matching line (per-PID entry in the first periodic print of prefetch_eviction_pid), not the final aggregate; they are labeled 'legacy first entry'. 'Final summary' values are parsed independently from the LAST '=== Summary ===' block of mem_tool.log (printed at detach) for final aggregate activated/used/allow/deny. sched: the scheduler prints its statistics once at exit, so legacy CSV metadata equals final values; the last '=== Statistics ===' block is parsed independently for the full final counter set. These provenance classes are kept separate and are never merged.

| tool | rows for tool arms | log absent | log ok but no final block | rows with final block |
|---|---:|---:|---:|---:|
| mem | 10 | 0 | 0 | 10 |
| sched | 10 | 0 | 0 | 10 |

| tool | counter | present | zero | absent in final block |
|---|---|---:|---:|---:|
| mem | total_activated | 10 | 0 | 0 |
| mem | total_used_calls | 10 | 0 | 0 |
| mem | policy_allow_moved | 10 | 0 | 0 |
| mem | policy_deny_not_moved | 10 | 0 | 0 |
| sched | policy_hit | 10 | 0 | 0 |
| sched | policy_miss | 10 | 10 | 0 |
| sched | timeslice_mod | 10 | 0 | 0 |

counter zero/missing reflects what the policy tools recorded (or failed to record); it is an interpretation limitation only, never a performance gate

### Per-row counter provenance

| row | block | arm | mem final summary (LAST block) | mem legacy first entry (CSV) | sched final statistics | sched legacy (CSV) |
|---:|---:|---|---|---|---|---|
| 2 | 0 | baseline | log tool_not_used_by_this_arm | n/a | log tool_not_used_by_this_arm | n/a |
| 3 | 0 | memory_only | act=32012576 used=88366 allow=21390035 deny=10625672 | act=1634629;allow=1451986;deny=182643 | log tool_not_used_by_this_arm | n/a |
| 4 | 0 | sched_only | log tool_not_used_by_this_arm | n/a | hit=12 miss=0 mod=12 | hit=12;miss=0;mod=12 |
| 5 | 0 | combined | act=10596152 used=19260 allow=8052193 deny=2553305 | act=1516559;allow=1349413;deny=167146 | hit=12 miss=0 mod=12 | hit=12;miss=0;mod=12 |
| 6 | 1 | memory_only | act=31675573 used=86974 allow=21197659 deny=10481096 | act=1631279;allow=1449562;deny=181717 | log tool_not_used_by_this_arm | n/a |
| 7 | 1 | sched_only | log tool_not_used_by_this_arm | n/a | hit=12 miss=0 mod=12 | hit=12;miss=0;mod=12 |
| 8 | 1 | combined | act=10586342 used=19260 allow=8046660 deny=2549028 | act=1541079;allow=1370992;deny=170087 | hit=12 miss=0 mod=12 | hit=12;miss=0;mod=12 |
| 9 | 1 | baseline | log tool_not_used_by_this_arm | n/a | log tool_not_used_by_this_arm | n/a |
| 10 | 2 | sched_only | log tool_not_used_by_this_arm | n/a | hit=12 miss=0 mod=12 | hit=12;miss=0;mod=12 |
| 11 | 2 | combined | act=10594462 used=19260 allow=8052063 deny=2551745 | act=1543899;allow=1373763;deny=170136 | hit=12 miss=0 mod=12 | hit=12;miss=0;mod=12 |
| 12 | 2 | baseline | log tool_not_used_by_this_arm | n/a | log tool_not_used_by_this_arm | n/a |
| 13 | 2 | memory_only | act=33576091 used=94277 allow=22345898 deny=11233200 | act=1639629;allow=1457647;deny=181982 | log tool_not_used_by_this_arm | n/a |
| 14 | 3 | combined | act=10536182 used=19260 allow=8005911 deny=2539617 | act=1540309;allow=1370609;deny=169700 | hit=12 miss=0 mod=12 | hit=12;miss=0;mod=12 |
| 15 | 3 | baseline | log tool_not_used_by_this_arm | n/a | log tool_not_used_by_this_arm | n/a |
| 16 | 3 | memory_only | act=31388248 used=85456 allow=21015852 deny=10375615 | act=1614909;allow=1435044;deny=179865 | log tool_not_used_by_this_arm | n/a |
| 17 | 3 | sched_only | log tool_not_used_by_this_arm | n/a | hit=12 miss=0 mod=12 | hit=12;miss=0;mod=12 |
| 18 | 4 | baseline | log tool_not_used_by_this_arm | n/a | log tool_not_used_by_this_arm | n/a |
| 19 | 4 | memory_only | act=31646643 used=86683 allow=21174963 deny=10474831 | act=1604149;allow=1426100;deny=178049 | log tool_not_used_by_this_arm | n/a |
| 20 | 4 | sched_only | log tool_not_used_by_this_arm | n/a | hit=12 miss=0 mod=12 | hit=12;miss=0;mod=12 |
| 21 | 4 | combined | act=10563262 used=19260 allow=8027824 deny=2544784 | act=1538679;allow=1368922;deny=169757 | hit=12 miss=0 mod=12 | hit=12;miss=0;mod=12 |

## Notes token counts

- CSV notes column: {}
- meta.json notes: {}

## Summary counts per arm

| arm | rows | high_rc != 0 | low_rc != 0 | high_rc missing | low_rc missing | high_timeouts | low_timeouts | meta.json absent/unreadable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| memory_only | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| sched_only | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| combined | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Timeouts are CSV high_timeout/low_timeout columns (1 = timed out); meta.json timeout booleans are in the JSON report.

## Interpretation limitations

- performance medians and paired deltas include only row-role records whose tenant rc is 0 (successful completion); records with rc nonzero or missing keep their raw values and are listed as excluded with explicit reasons. rc/timeout/counter fields are classification metadata and never gate the report.
- counter zero/missing reflects what the policy tools recorded (or failed to record); it is an interpretation limitation only, never a performance gate
- mem: values in CSV meta columns were recorded by run_fig13_fast.py parse_mem_meta via re.search, which captures the FIRST matching line (per-PID entry in the first periodic print of prefetch_eviction_pid), not the final aggregate; they are labeled 'legacy first entry'. 'Final summary' values are parsed independently from the LAST '=== Summary ===' block of mem_tool.log (printed at detach) for final aggregate activated/used/allow/deny. sched: the scheduler prints its statistics once at exit, so legacy CSV metadata equals final values; the last '=== Statistics ===' block is parsed independently for the full final counter set. These provenance classes are kept separate and are never merged.
