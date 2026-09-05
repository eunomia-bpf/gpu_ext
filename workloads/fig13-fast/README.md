# Fig.13 fast — four-arm memory/scheduling matrix (minimal runnable)

Fresh, independently timed companion to the historical Fig. 13: one HotSpot
uvmbench workload, two concurrent tenants, four policy arms, five interleaved
blocks.

## Arms

| arm          | memory policy (prefetch_eviction_pid) | sched policy (gpu_sched_set_timeslices) |
|--------------|----------------------------------------|------------------------------------------|
| `baseline`   | none                                   | none                                     |
| `memory_only`| high PID `-P 20`, low PID `-L 80`      | none                                     |
| `sched_only` | none                                   | `uvmbench_high:1000000us`, `uvmbench_low:200us` |
| `combined`   | both                                   | both                                     |

Workload: `microbench/memory/uvmbench --kernel=hotspot --size_factor=0.6
--mode=uvm --iterations=1`, two concurrent tenants (`uvmbench_high`,
`uvmbench_low`, launched via `/tmp` symlinks so the scheduler policy can key
on comm).

## Per-arm ordering

1. Both tenants are SIGSTOPped before exec: both PIDs exist before any policy
   starts and before any tenant CUDA initialization.
2. Sched policy (where present) starts while tenants are stopped, i.e. before
   tenant CUDA initialization.
3. Memory policy (where present) starts after tenant PIDs exist and before
   SIGCONT.
4. Both tenants are resumed; each tenant's completion latency is timed
   independently from its own SIGCONT to its own exit.

Five interleaved blocks: each block runs all four arms back to back with the
arm order rotated (block `b` starts at arm `b`), so every arm occupies every
position across blocks.

## Measurements

- Per-tenant completion latency (s), independently timed.
- Per-tenant uvmbench median time (ms) and bandwidth (GB/s), parsed from the
  tenant log; uvmbench's own `--output` CSV is kept per arm.
- Tool logs: `sched_tool.log`, `mem_tool.log`, plus `events.log`,
  `run.json`, per-arm `meta.json`, and tenant logs.
- Engagement counters (sched `policy_hit/policy_miss/timeslice_mod`; memory
  `Total activated/Policy allow/Policy deny`) are recorded as **metadata
  only and are never a gate**.

No correctness/review/verifier/hash/checksum/digest gate, no retry, no
filtering: failures and raw numbers are preserved (row `notes`, per-arm
`meta.json`, tenant/tool logs). A per-tenant wall-clock timeout (default
600 s) kills a stuck tenant and the resulting numbers/exit codes are kept.

## Layout

```
workloads/fig13-fast/
  run_fig13_fast.py        harness
  stubs/stub_tenant.py     CPU dry-run tenant
  stubs/stub_policy.py     CPU dry-run policy tool
  results/fig13_fast_<ts>[_dryrun]/
    run.json  fig13_fast.csv  events.log
    blockNN_<arm>/{meta.json, tenant_uvmbench_high.log, tenant_uvmbench_low.log,
                   uvmbench_uvmbench_high_results.csv, uvmbench_uvmbench_low_results.csv,
                   sched_tool.log (arms with sched), mem_tool.log (arms with mem)}
```

## Run

CPU dry-run test (no GPU, no BPF):

```
python3 workloads/fig13-fast/run_fig13_fast.py --dry-run
```

GPU run (one exact command; requires the repo's extension tools to be built
and the kernel nvidia driver with the struct_ops hooks):

```
cd /home/yunwei37/workspace/gpu/gpu_ext/workloads/fig13-fast && sudo python3 run_fig13_fast.py --blocks 5
```
