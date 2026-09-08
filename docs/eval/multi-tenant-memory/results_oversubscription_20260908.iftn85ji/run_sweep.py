"""Configure the original two-tenant runner for the authorized five-policy sweep."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import csv
import json
import shutil

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/yunwei37/workspace/gpu/gpu_ext')
source = REPO / 'docs/eval/multi-tenant-memory/run_policy_comparison.py'
spec = importlib.util.spec_from_file_location('original_policy_runner', source)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.COMBINED_ARMS = ('baseline', 'sched_only', 'prefetch_only', 'memory_only', 'combined')
runner.COMBINED_ARM_TOOLS['prefetch_only'] = ('mem',)
original_cell = runner.run_combined_arm

def configured_cell(block, policy, factor, kernel, run_root, link_dir, args, log):
    tool = 'prefetch_pid_tree' if policy == 'prefetch_only' else 'prefetch_eviction_pid'
    args.mem_tool = str(REPO / 'extension' / tool)
    row = original_cell(block, policy, factor, kernel, run_root, link_dir, args, log)
    meta = json.loads((run_root / f'block{block:02d}_{policy}' / 'meta.json').read_text())
    failed = any(str(row[f'{role}_rc']) != '0' for role in ('high', 'low'))
    failed |= any(meta.get(f'rc_{tool}_tool', 0) != 0 for tool in ('sched', 'mem'))
    if failed:
        with (run_root / 'combined_comparison.csv').open('a', newline='') as f:
            csv.DictWriter(f, fieldnames=runner.COMBINED_CSV_COLUMNS).writerow(row)
        raise RuntimeError(f'Failed experiment retained: {kernel}/{factor}/{block}/{policy}')
    return row

runner.run_combined_arm = configured_cell
args = SimpleNamespace(blocks=5,
    mem_tool=str(REPO/'extension/prefetch_eviction_pid'),
    sched_tool=str(REPO/'extension/gpu_sched_set_timeslices'),
    uvmbench=str(REPO/'microbench/memory/uvmbench'),
    tenant_launcher=str(REPO/'workloads/fig13-fast/tenant_launcher.py'))
for ratio in (0.8, 1.0, 1.2, 1.5):
    for kernel in ('hotspot', 'gemm', 'kmeans_sparse'):
        output = ROOT / kernel / f'ratio-{ratio:.1f}'
        if output.exists():
            raise RuntimeError(f'Refusing to repeat existing measurements: {output}')
        print(f'SWEEP_START kernel={kernel} requested_total_ratio={ratio}', flush=True)
        path = runner.run_combined_mode(args, kernel, ratio/2, output)
        print(f'SWEEP_COMPLETE kernel={kernel} ratio={ratio} csv={path}', flush=True)
print('ALL_SWEEP_MEASUREMENTS_COMPLETE', flush=True)
