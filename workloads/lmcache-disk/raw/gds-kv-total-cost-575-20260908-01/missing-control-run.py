import fcntl
import json
from pathlib import Path
import sys
import time

work = Path('/home/yunwei37/workspace/gpu/gpu_ext/workloads/lmcache-disk')
sys.path.insert(0, str(work))
import run_gds_kv_reclaim as r

root = work / 'raw/gds-kv-total-cost-575-20260908-01'
locks = [open(p, 'r+') for p in ('/tmp/gpubpf-revision-gpu0.lock', '/tmp/gpubpf-revision-struct-ops.lock')]
print('Waiting for current campaign to release GPU and struct-ops locks', flush=True)
for lock in locks:
    fcntl.flock(lock, fcntl.LOCK_EX)
campaign = json.loads((root / 'campaign.json').read_text())
if campaign.get('status') != 'complete':
    raise RuntimeError('main campaign did not complete; leave missing control for explicit continuation')
run_dir = root / 'block-01/position-4-native_ratio-restored'
run_dir.mkdir(parents=True, exist_ok=False)
r.ops.atomic_write_text(root / 'missing-control-run.py', Path(__file__).read_text())
r.ops.atomic_write_json(root / 'campaign-before-control-replacement.json', campaign)
original_environment = r.cell_server_environment
def environment(*args, **kwargs):
    env = original_environment(*args, **kwargs)
    env['LMCACHE_KV_RECLAIM_NATIVE_LIB'] = str(work / 'gds-control/kv_reclaim_native.so')
    return env
r.cell_server_environment = environment
old = json.loads((root / 'block-00/position-0-stock/result.json').read_text())
prefixes = r.perf.load_fixed_prompts()['prefixes']
r.ops.atomic_write_json(run_dir / 'configuration.json', {'policy_label': 'native_ratio', 'source_commit': '4381f660', 'native_library': str(work / 'gds-control/kv_reclaim_native.so'), 'reason': 'missing planned control after unselected variant overwrite; measured after remaining blocks, not interleaved with block1', 'bpf_object': None})
print('START missing original-ratio control for block1, position4', flush=True)
result = r.run_cell('native', 1, 4, run_dir, 18080, Path(old['command'][2]), prefixes, r.warm_arrival_order(1, len(prefixes)), '575.57.08', r.DEFAULT_STORE_BARRIER_TIMEOUT_S, 256, 402653184, 1024, 250.0, 4, 62502)
for cell in campaign['cells']:
    if cell['block'] == 1 and cell['position'] == 0:
        cell['requested_policy_label'] = cell['policy_label']
        cell['policy_label'] = 'native_net_unplanned'
        cell['policy_correction'] = str(root / 'block-01/position-0-native_ratio-after-enospc/policy-correction.md')
campaign['cells'].append({'block': 1, 'position': 4, 'policy_label': 'native_ratio', 'result': str(run_dir / 'result.json'), 'warm_phase': result.get('warm_phase'), 'error': result.get('error'), 'late_replacement': True})
campaign['missing_control_completed_ns'] = time.time_ns()
r.ops.atomic_write_json(root / 'campaign.json', campaign)
print(json.dumps({'DONE': 'native_ratio_restored', 'warm_phase': result.get('warm_phase'), 'error': result.get('error')}), flush=True)
