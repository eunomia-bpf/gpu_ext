#!/usr/bin/env python3
"""Thin POD five-arm coordinator; real GPU work requires the shared leases.

The benchmark owns operator numerics/decisions. This file only owns processes,
the private loader segment, the existing safety checks, and the frozen order.
Import and CPU unit tests perform no GPU work.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import stat
import subprocess
import sys
import time

import bench
from prepare_ptx import TUS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'gpreempt'))
import run_three_way as shared

safety = shared.safety
BPFTIME = HERE.parents[2] / 'bpftime/build-cuda-pr503'
AGENT = BPFTIME / 'runtime/agent/libbpftime-agent.so'
SERVER = BPFTIME / 'runtime/syscall-server/libbpftime-syscall-server.so'
PYTHON = HERE.parent / 'moe-infinity/.venv/bin/python'
BRIDGE = HERE / 'build/libpod_launch_bridge.so'
TIMEOUT = 1800
OPERATOR_PHASE_KEYS = ('process_main_ns', 'stdlib_imports_done_ns',
                       'runtime_imports_start_ns', 'runtime_imports_done_ns',
                       'pre_first_diagnostic_ns', 'post_first_sync_ns',
                       'warmup_done_ns', 'steady_complete_ns')


def orders(mode):
    rng = random.Random(20260903)
    result = []
    for block in range(1, 2 if mode == 'preflight' else 6):
        arms = list(bench.ARMS)
        rng.shuffle(arms)
        result.extend({'block': block, 'arm': arm} for arm in arms)
    return result


def environment(arm, extraction, shm=None, loader=False):
    # Deliberately do not inherit another experiment's injection or policy env.
    env = dict(PATH='/usr/local/cuda-12.9/bin:/usr/bin:/bin', LANG='C.UTF-8',
               CUDA_VISIBLE_DEVICES='0', LD_LIBRARY_PATH='/usr/local/cuda-12.9/lib64:/usr/lib/x86_64-linux-gnu',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', SPDLOG_LEVEL='warn')
    if loader or arm == 'pod_bpf':
        if not shm or not shm.startswith('pod_attention_') or '/' in shm:
            raise ValueError('a unique private POD segment is required')
        env['BPFTIME_GLOBAL_SHM_NAME'] = shm
        if loader:
            env['LD_PRELOAD'] = str(SERVER)
        else:
            env.update(LD_PRELOAD=f'{BRIDGE}:{AGENT}', POD_LAUNCH_BRIDGE='bpf',
                BPFTIME_PTXPASS_LIBRARIES=str(HERE / 'build/libpod_ptx_adapter.so'),
                BPFTIME_CUDA_LATE_PTX_DIR=str(extraction / 'device'),
                BPFTIME_CUDA_DEFER_PTX_EXTRACTION='1', BPFTIME_CUDA_DISABLE_CUOBJDUMP='1')
    elif arm == 'pod_cuda':
        env.update(LD_PRELOAD=str(BRIDGE), POD_LAUNCH_BRIDGE='cuda')
    elif arm not in bench.ARMS:
        raise ValueError('unknown POD arm')
    return env


def preparation(extraction):
    value = json.loads((extraction / 'inventory.json').read_text())
    records = value.get('records', [])
    if (value.get('complete') is not True or value.get('cpu_preparation_only') is not True
            or value.get('planned_tus') != list(TUS) or [r['tu'] for r in records] != list(TUS)):
        raise ValueError('missing complete linked four-TU preparation')
    paths = [Path(value['extension']), HERE / 'build/python/flash_attn_og.cpython-312-x86_64-linux-gnu.so',
             HERE / 'build/pod-loader', HERE / 'build/selector.bpf.o', HERE / 'build/selector.bin',
             HERE / 'build/libpod_ptx_adapter.so', BRIDGE, AGENT, SERVER, HERE / 'bench.py',
             HERE / 'run_study.py', extraction / 'inventory.json', extraction / 'exact-kernels.txt']
    representatives = []
    for record in records:
        packets = record['packets']
        if sum(p['official_entries'] for p in packets) != 135 or sum(p['typed_calls'] for p in packets) != 128:
            raise ValueError('incomplete official entry/call inventory')
        for packet in packets:
            if (not 0 < packet['response_json_bytes'] < packet['transport_capacity'] == 64 << 20
                    or Path(packet['filename']).name != packet['filename']):
                raise ValueError('invalid actual PTX transport packet')
            representatives.append(packet['representative'])
            paths.append(extraction / 'device' / packet['filename'])
    names = (extraction / 'exact-kernels.txt').read_text().splitlines()
    if names != representatives or len(names) != 6 or len(set(names)) != 6:
        raise ValueError('exact loader inventory differs from actual six packets')
    return paths


def file_inventory(paths):
    result = {}
    for path in paths:
        info = path.stat()
        if not path.is_file() or not info.st_size:
            raise ValueError('missing runtime file: ' + str(path))
        result[str(path)] = dict(bytes=info.st_size, mtime_ns=info.st_mtime_ns)
    return result


def require_no_build():
    active = []
    for path in Path('/proc').glob('[0-9]*/comm'):
        try:
            name = path.read_text().strip()
        except OSError:
            continue
        if name in ('nvcc', 'cicc', 'ptxas', 'ninja'):
            active.append((int(path.parent.name), name))
    if active:
        raise RuntimeError(f'CPU compilation must finish before GPU measurement: {active}')


def ordered_monotonic_ns(record, keys):
    values = [record.get(key) for key in keys]
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError('missing/non-integer monotonic phase timestamp')
    if values != sorted(values):
        raise ValueError('monotonic phase timestamps are out of order')
    return values


def validate_report(report, arm, block, preflight, phase_study=False):
    expected = bench.shape_order(block, preflight or phase_study)
    if (report.get('complete') is not True or report.get('numeric_protocol') != bench.NUMERIC_PROTOCOL
            or 'error' in report or report.get('arm') != arm
            or report.get('block') != block or report.get('preflight') is not preflight
            or report.get('phase_study', False) is not phase_study
            or report.get('shape_order') != [list(x) for x in expected]
            or [(c['model'], c['decode_batch']) for c in report.get('cells', [])] != expected):
        raise ValueError('incomplete/wrong operator result or shape order')
    if phase_study:
        if set(report.get('phase_timestamps', {})) != set(OPERATOR_PHASE_KEYS):
            raise ValueError('phase study has missing/extra operator timestamps')
        ordered_monotonic_ns(report['phase_timestamps'], OPERATOR_PHASE_KEYS)
    for cell in report['cells']:
        settings = dict(numeric_protocol=bench.NUMERIC_PROTOCOL, kv_heads=bench.MODEL_HEADS[cell['model']], query_heads=32, head_dim=128,
            prefill_batch=1, prefill_length=8192, decode_query_length=1, decode_cache_extent=8192,
            decode_valid_kv=8191, dtype='float16', warmups=10, atol=1e-3, rtol=1e-5,
            seed=20260904 + (0 if cell['model'] == 'llama-3-8b' else 1000) + cell['decode_batch'])
        if any(cell.get(key) != val for key, val in settings.items()):
            raise ValueError('official workload or fixed correctness settings differ')
        samples = cell['samples']
        if len(samples) != (3 if preflight else 100):
            raise ValueError('missing unfiltered operator observations')
        for key, output in (('cuda_ms', 'mean_cuda_ms'), ('host_wall_ms', 'mean_host_wall_ms')):
            values = [s[key] for s in samples]
            if (any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values)
                    or not math.isclose(sum(values) / len(values), cell[output], rel_tol=1e-12)):
                raise ValueError('invalid raw timing or estimator')
        for key in ('max_abs_vs_official', 'official_max_abs_vs_fp32'):
            if not math.isfinite(cell[key]) or cell[key] < 0:
                raise ValueError('missing actual numerical validation')
        characterization = cell.get('fp32_characterization', {})
        if set(characterization) != {'prefill', 'decode'}:
            raise ValueError('both full-shape FP32 characterizations are required')
        for phase, shape in (('prefill', [1, 8192, 32, 128]), ('decode', [cell['decode_batch'], 1, 32, 128])):
            stats = characterization[phase]
            if (stats.get('numeric_protocol') != bench.NUMERIC_PROTOCOL or stats.get('phase') != phase
                    or stats.get('role') != 'characterization_not_cross_precision_pass_gate'
                    or stats.get('finite') is not True or stats.get('shape_checked') is not True
                    or stats.get('mask') != ('causal_prefix' if phase == 'prefill' else 'valid_kv')
                    or stats.get('output_shape') != shape or type(stats.get('checked_elements')) is not int
                    or stats.get('checked_elements') != math.prod(shape)
                    or type(stats.get('exceeding_elements')) is not int
                    or not 0 <= stats['exceeding_elements'] <= stats['checked_elements']
                    or stats.get('atol') != 1e-3 or stats.get('rtol') != 1e-5):
                raise ValueError('incomplete FP32 characterization or changed reference semantics')
            for metric in ('max_abs_error', 'mean_abs_error', 'rms_error'):
                if type(stats.get(metric)) not in (float, int) or not math.isfinite(stats[metric]) or stats[metric] < 0:
                    raise ValueError('invalid full-shape FP32 error statistics')
            if stats['mean_abs_error'] > stats['rms_error'] + 1e-15 or stats['rms_error'] > stats['max_abs_error'] + 1e-15:
                raise ValueError('inconsistent full-shape error statistics')
            expected_dir = bench.fp32_diagnostic_name(cell['model'], cell['decode_batch'], phase) if stats['exceeding_elements'] else None
            if stats.get('diagnostic_directory') != expected_dir:
                raise ValueError('missing/non-unique actual excess-error row diagnostic')
        if cell['official_max_abs_vs_fp32'] != max(x['max_abs_error'] for x in characterization.values()):
            raise ValueError('FP32 summary differs from full-phase characterizations')
        if arm.startswith('pod_'):
            diagnostic = cell['diagnostic']
            mode = {'pod_inline': 0, 'pod_cuda': 1, 'pod_bpf': 2}[arm]
            meta = diagnostic['metadata']
            if cell['fused_params'] != 15 or meta['mode'] != mode or meta['trace'] != 1:
                raise ValueError('wrong original tile selection or device mode')
            audited = bench.audit_decisions(meta, diagnostic['counters'], diagnostic['contexts'],
                                           2 if arm == 'pod_bpf' else 1)
            if audited != diagnostic['audit']:
                raise ValueError('saved atomic/exactly-once audit differs from actual context')
            if arm != 'pod_inline':
                bridge = cell['launch_bridge']
                bench.audit_bridge(bridge['before'], bridge['after'], 11 + len(samples),
                                   meta['smem_bytes'], arm.removeprefix('pod_'))
                if phase_study:
                    launches = bridge.get('first_launches', [])
                    if (len(launches) != 1
                            or bridge['after'].get('first_launches', 0)
                               - bridge['before'].get('first_launches', 0) != 1
                            or launches[0].get('kernel', '').find('true_fused_tb_fwd_kernel') < 0
                            or type(launches[0].get('monotonic_ns')) is not int
                            or not report['phase_timestamps']['pre_first_diagnostic_ns']
                                   <= launches[0]['monotonic_ns']
                                   <= report['phase_timestamps']['post_first_sync_ns']):
                        raise ValueError('fixed-shape first kernel launch is missing or out of phase')
            elif cell['launch_bridge'] is not None:
                raise ValueError('inline baseline unexpectedly used launch injection')
        elif cell['diagnostic'] is not None or cell['launch_bridge'] is not None or cell['fused_params'] is not None:
            raise ValueError('non-fused baseline unexpectedly used POD or injection')
    scans = {f"{cell['model']}:bs{cell['decode_batch']}": cell['fp32_characterization'] for cell in report['cells']}
    if report.get('fp32_characterizations') != scans:
        raise ValueError('checkpointed complete reference scans differ from final cells')
    return report


def segment_identity(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise RuntimeError('owned segment has unexpected type/owner')
    return info.st_dev, info.st_ino, info.st_uid


def remove_owned_segment(path, identity):
    if identity is None:
        return False
    try:
        actual = segment_identity(path)
    except FileNotFoundError:
        return False
    if actual != identity:
        raise RuntimeError('private segment changed identity; refusing removal')
    path.unlink()
    return True


def validate_phase_execution(result, report, arm):
    phase = result.get('phase_timestamps', {})
    common = ('cell_start_ns', 'client_spawn_ns', 'client_exit_ns',
              'cleanup_complete_ns')
    ordered_monotonic_ns(phase, common)
    if arm == 'pod_bpf':
        ordered_monotonic_ns(phase, ('cell_start_ns', 'loader_spawn_ns',
                                    'loader_ready_ns', 'client_spawn_ns'))
    elif phase.get('loader_spawn_ns') is not None or phase.get('loader_ready_ns') is not None:
        raise ValueError('non-BPF phase arm unexpectedly started a policy loader')
    operator = ordered_monotonic_ns(report['phase_timestamps'], OPERATOR_PHASE_KEYS)
    if not phase['client_spawn_ns'] <= operator[0] <= operator[-1] <= phase['client_exit_ns']:
        raise ValueError('operator timestamps escape the fresh client lifetime')
    return result


def run_cell(directory, specification, mode, extraction, runtime_paths, campaign_runtime,
             phase_study=False):
    directory.mkdir()
    arm, block = specification['arm'], specification['block']
    command = ['taskset', '-c', '8-15', str(PYTHON), str(HERE / 'bench.py'), '--arm', arm,
               '--block', str(block), '--output', str(directory / 'operator.json')]
    if mode == 'preflight':
        command.append('--preflight')
    if phase_study:
        command.append('--phase-study')
    name = f'pod_attention_{os.getpid()}_{time.monotonic_ns()}' if arm == 'pod_bpf' else None
    segment = Path('/dev/shm') / name if name else None
    target_env = environment(arm, extraction, name)
    launch_env = dict(target_env)
    preload = launch_env.pop('LD_PRELOAD', None)
    if preload:
        # Pin the wrapper before loading any agent. Preloading taskset would
        # initialize bpftime before its main/affinity/exec, then again in Python.
        command[3:3] = ['/usr/bin/env', 'LD_PRELOAD=' + preload]
    client = loader = telemetry = before = identity = operator_report = None
    streams, cleanup = [], []
    result = dict(status='failed', numeric_protocol=bench.NUMERIC_PROTOCOL, **specification, command=command, timeout_seconds=TIMEOUT,
                  environment=target_env, launch_environment=launch_env, private_segment=name,
                  phase_study=phase_study)
    if phase_study:
        result['phase_timestamps'] = dict(cell_start_ns=time.monotonic_ns(),
                                          loader_spawn_ns=None, loader_ready_ns=None,
                                          client_spawn_ns=None, client_exit_ns=None,
                                          cleanup_complete_ns=None)
    try:
        require_no_build()
        result['runtime_before'] = file_inventory(runtime_paths)
        if result['runtime_before'] != campaign_runtime:
            raise RuntimeError('runtime files changed since the frozen campaign inventory')
        before = safety.safety_snapshot()
        safety.validate_pre_server_safety(before)
        if before['gpu']['driver'] != '575.57.08':
            raise RuntimeError('the prepared 575 driver is required; no driver changes are made')
        result['safety_before'] = before
        telemetry, stream, telemetry_path = safety.start_gpu_telemetry(directory)
        streams.append(stream)
        if name:
            if segment.exists() or segment.is_symlink():
                raise RuntimeError('private segment already exists; refusing loader start')
            log = (directory / 'loader.log').open('x')
            streams.append(log)
            loader_command = [str(HERE / 'build/pod-loader'), str(HERE / 'build/selector.bpf.o'),
                              str(extraction / 'exact-kernels.txt')]
            loader_env = environment(arm, extraction, name, loader=True)
            result.update(loader_command=loader_command, loader_environment=loader_env)
            if phase_study:
                result['phase_timestamps']['loader_spawn_ns'] = time.monotonic_ns()
            loader = subprocess.Popen(loader_command, stdin=subprocess.PIPE, stdout=log,
                stderr=subprocess.STDOUT, env=loader_env, start_new_session=True, cwd=HERE)
            deadline = time.monotonic() + 30
            while 'POD_LOADER_READY kernels=6\n' not in (directory / 'loader.log').read_text():
                if identity is None:
                    try:
                        identity = segment_identity(segment)
                    except FileNotFoundError:
                        pass
                if loader.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError('private BPF loader did not become ready')
                time.sleep(.1)
            if identity is None:
                identity = segment_identity(segment)
            if phase_study:
                result['phase_timestamps']['loader_ready_ns'] = time.monotonic_ns()
        log = (directory / 'client.log').open('x')
        streams.append(log)
        started = time.monotonic()
        if phase_study:
            result['phase_timestamps']['client_spawn_ns'] = time.monotonic_ns()
        client = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
            env=launch_env, start_new_session=True, cwd=HERE)
        while client.poll() is None:
            if time.monotonic() - started >= TIMEOUT:
                raise TimeoutError('owned operator client exceeded startup/runtime bound')
            if loader is not None and loader.poll() is not None:
                raise RuntimeError('private loader exited while CUDA client was alive')
            time.sleep(.2)
        if phase_study:
            result['phase_timestamps']['client_exit_ns'] = time.monotonic_ns()
        result.update(returncode=client.returncode, process_wall_seconds=time.monotonic() - started)
        if client.returncode:
            raise RuntimeError(f'operator client exited {client.returncode}')
        operator_report = json.loads((directory / 'operator.json').read_text())
        validate_report(operator_report, arm, block, mode == 'preflight', phase_study)
        result['status'] = 'passed'
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            shared.stop_owned(client)
        except BaseException as error:
            cleanup.append(str(error))
        client_stopped = client is None or not shared.group_members(client.pid)
        if not client_stopped:
            cleanup.append('CUDA client still alive; cannot safely close its policy loader')
        if loader is not None and client_stopped:
            try:
                # An open stdin owns the policy lifetime, not a guessed sleep.
                loader.stdin.close()
                loader.wait(timeout=10)
                if loader.returncode != 0 or (directory / 'loader.log').read_text().count('POD_LOADER_CLOSED\n') != 1:
                    raise RuntimeError('private loader failed its orderly detach')
            except BaseException as error:
                cleanup.append(str(error))
            try:
                shared.stop_owned(loader)
            except BaseException as error:
                cleanup.append(str(error))
        try:
            if segment is not None:
                if any(p is not None and shared.group_members(p.pid) for p in (client, loader)):
                    raise RuntimeError('owned processes survive; refusing private segment removal')
                if identity is None:
                    if segment.exists() or segment.is_symlink():
                        raise RuntimeError('unidentified private segment survived; refusing removal')
                    result['private_segment_removed'] = False
                else:
                    result['private_segment_removed'] = remove_owned_segment(segment, identity)
                if segment.exists() or segment.is_symlink():
                    raise RuntimeError('private loader segment survived cleanup')
        except BaseException as error:
            cleanup.append(str(error))
        try:
            shared.stop_owned(telemetry)
        except BaseException as error:
            cleanup.append(str(error))
        for stream in streams:
            try:
                stream.close()
            except BaseException as error:
                cleanup.append(str(error))
        try:
            result['runtime_after'] = file_inventory(runtime_paths)
            if result.get('runtime_before') != result['runtime_after']:
                raise RuntimeError('runtime files changed during the cell')
        except BaseException as error:
            cleanup.append(str(error))
        try:
            if before is not None:
                result['safety_after'] = safety.wait_for_post_server_safety(before)
        except BaseException as error:
            cleanup.append(str(error))
        try:
            if telemetry is not None:
                result['telemetry'] = safety.validate_gpu_telemetry(telemetry_path, allow_fixed_power_cap=True)
        except BaseException as error:
            cleanup.append(str(error))
        if phase_study:
            result['phase_timestamps']['cleanup_complete_ns'] = time.monotonic_ns()
            if result.get('status') == 'passed' and not cleanup:
                try:
                    validate_phase_execution(result, operator_report, arm)
                except BaseException as error:
                    cleanup.append(str(error))
        if cleanup:
            result.update(status='failed', cleanup_errors=cleanup)
        safety.atomic_write_json(directory / 'execution.json', result)
        if cleanup:
            raise RuntimeError('; '.join(cleanup))
    return result


def validate_preflight(directory, current_inventory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest.get('complete') is not True or manifest.get('mode') != 'preflight'
            or manifest.get('numeric_protocol') != bench.NUMERIC_PROTOCOL
            or manifest.get('order') != orders('preflight') or manifest.get('runtime') != current_inventory):
        raise ValueError('formal study requires complete preflight of the unchanged runtime')
    for item in orders('preflight'):
        cell = directory / f"block-{item['block']:02d}-{item['arm']}"
        execution = json.loads((cell / 'execution.json').read_text())
        if (execution.get('status') != 'passed' or execution.get('numeric_protocol') != bench.NUMERIC_PROTOCOL
                or execution.get('cleanup_errors')
                or execution.get('runtime_before') != current_inventory
                or execution.get('runtime_after') != current_inventory):
            raise ValueError('preflight contains failed/incomplete cleanup')
        validate_report(json.loads((cell / 'operator.json').read_text()), item['arm'], item['block'], True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'full'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ptx', type=Path, default=HERE / 'build/ptx-runtime-01')
    parser.add_argument('--preflight', type=Path, help='required for full: prior complete five-arm preflight')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite an existing campaign')
    extraction = args.ptx.resolve(strict=True)
    runtime_paths = preparation(extraction)
    runtime = file_inventory(runtime_paths)
    if args.mode == 'full':
        if args.preflight is None:
            parser.error('full requires --preflight from this unchanged runtime')
        validate_preflight(args.preflight, runtime)
    require_no_build()
    output = args.output.resolve()
    output.mkdir(parents=True)
    manifest = dict(complete=False, numeric_protocol=bench.NUMERIC_PROTOCOL, mode=args.mode, order=orders(args.mode), seed=20260903,
                    runtime=runtime, ptx=str(extraction), excluded_from_formal=args.mode == 'preflight',
                    preflight=str(args.preflight.resolve()) if args.preflight is not None else None,
                    completed=[], lease_paths=['/tmp/gpubpf-revision-gpu0.lock',
                                              '/tmp/gpubpf-revision-struct-ops.lock'])
    lease = None
    try:
        # Identical two lock files used by GPreempt, Hummingbird and FineMoE.
        lease = shared.Leases()
        safety.atomic_write_json(output / 'manifest.json', manifest)
        for item in manifest['order']:
            run_cell(output / f"block-{item['block']:02d}-{item['arm']}", item, args.mode,
                     extraction, runtime_paths, runtime)
            manifest['completed'].append(item)
            safety.atomic_write_json(output / 'manifest.json', manifest)
        manifest['complete'] = True
    except BaseException as error:
        manifest['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        safety.atomic_write_json(output / 'manifest.json', manifest)
        if lease is not None:
            lease.close()


if __name__ == '__main__':
    main()
