import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

repo = Path('/home/yunwei37/workspace/gpu/gpu_ext')
work = repo / 'workloads/lmcache-disk'
sys.path.insert(0, str(work))
import run_gds_kv_reclaim as r

root = work / 'raw/gds-kv-total-cost-575-20260908-01'
control = work / 'gds-control'
old_command = [str(control / 'kv_reclaim_loader'), str(control / 'kv_reclaim_policy.bpf.o')]
new_command = [old_command[0], str(control / 'kv_reclaim_total_cost.bpf.o')]
labels = ['stock', 'native_ratio', 'native_total', 'bpf_total']
original_environment = r.cell_server_environment
label = None

def environment(*args, **kwargs):
    env = original_environment(*args, **kwargs)
    env.pop('LMCACHE_KV_RECLAIM_NATIVE_LIB', None)
    if label in ('native_ratio', 'native_total'):
        name = 'kv_reclaim_native.so' if label == 'native_ratio' else 'kv_reclaim_total_cost_native.so'
        env['LMCACHE_KV_RECLAIM_NATIVE_LIB'] = str(control / name)
    return env

r.cell_server_environment = environment
locks = [open(path, 'r+') for path in ('/tmp/gpubpf-revision-gpu0.lock', '/tmp/gpubpf-revision-struct-ops.lock')]
for lock in locks:
    fcntl.flock(lock, fcntl.LOCK_EX)
root.mkdir(parents=True, exist_ok=False)
r.ops.atomic_write_text(root / 'run.py', Path(__file__).read_text())
prefixes = r.perf.load_fixed_prompts()['prefixes']
old_result = json.loads((work / 'raw/gds-kv-reclaim-grace-575-20260908-01/native-async/result.json').read_text())
model = Path(old_result['command'][2])
campaign = {'source_commit': '4381f660', 'policy_labels': labels, 'cells': [], 'old_loader_command': old_command, 'new_loader_command': new_command, 'status': 'starting', 'started_ns': time.time_ns()}
r.ops.atomic_write_json(root / 'campaign.json', campaign)
actual = subprocess.check_output(['sudo', '-n', 'cat', '/proc/1783868/cmdline']).decode().rstrip('\0').split('\0')
if actual != old_command:
    raise RuntimeError('old loader PID no longer matches the owned command; no process changed')
subprocess.run(['sudo', '-n', 'kill', '-TERM', '1783868'], check=True)
while Path('/proc/1783868').exists():
    stat = Path('/proc/1783868/stat').read_text().split()
    if stat[2] == 'Z':
        break
    time.sleep(0.1)
loader = None
try:
    with (root / 'total-cost-loader.log').open('x') as stream:
        loader = subprocess.Popen(['sudo', '-n', *new_command], stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    campaign['loader_process_group'] = loader.pid
    while 'attached' not in (root / 'total-cost-loader.log').read_text():
        if loader.poll() is not None:
            raise RuntimeError(f'total-cost loader exited {loader.returncode}; see total-cost-loader.log')
        time.sleep(0.25)
    campaign['status'] = 'running'
    r.ops.atomic_write_json(root / 'campaign.json', campaign)
    for block in range(5):
        order = labels[block % 4:] + labels[:block % 4]
        for position, label in enumerate(order):
            run_dir = root / f'block-{block:02d}' / f'position-{position}-{label}'
            run_dir.mkdir(parents=True, exist_ok=False)
            arm = 'stock' if label == 'stock' else 'bpf' if label == 'bpf_total' else 'native'
            r.ops.atomic_write_json(run_dir / 'configuration.json', {'policy_label': label, 'source_commit': '4381f660', 'bpf_object': new_command[1] if arm == 'bpf' else None, 'native_library': environment(arm, run_dir / 'cache', '575.57.08', 256, 62502, None).get('LMCACHE_KV_RECLAIM_NATIVE_LIB')})
            print(f'START block={block} position={position} policy={label}', flush=True)
            result = r.run_cell(arm, block, position, run_dir, 18080, model, prefixes, r.warm_arrival_order(block, len(prefixes)), '575.57.08', r.DEFAULT_STORE_BARRIER_TIMEOUT_S, 256, 402653184, 1024, 250.0, 4, 62502)
            campaign['cells'].append({'block': block, 'position': position, 'policy_label': label, 'result': str(run_dir / 'result.json'), 'warm_phase': result.get('warm_phase'), 'error': result.get('error')})
            r.ops.atomic_write_json(root / 'campaign.json', campaign)
            print(json.dumps({'DONE': label, 'block': block, 'warm_phase': result.get('warm_phase'), 'error': result.get('error')}), flush=True)
    campaign['status'] = 'complete'
except BaseException as error:
    campaign['status'] = 'error'
    campaign['error'] = f'{type(error).__name__}: {error}'
    raise
finally:
    if loader is not None and loader.poll() is None:
        subprocess.run(['sudo', '-n', 'kill', '-TERM', '--', f'-{loader.pid}'], check=True)
        loader.wait()
    with (root / 'restored-original-loader.log').open('x') as stream:
        restored = subprocess.Popen(['sudo', '-n', *old_command], stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    campaign['restored_loader_process_group'] = restored.pid
    while 'attached' not in (root / 'restored-original-loader.log').read_text():
        if restored.poll() is not None:
            campaign['restore_error'] = f'loader exited {restored.returncode}'
            break
        time.sleep(0.25)
    campaign['finished_ns'] = time.time_ns()
    r.ops.atomic_write_json(root / 'campaign.json', campaign)
