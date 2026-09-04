#!/usr/bin/env python3
"""POD setup/first-launch/steady three-arm follow-up.

This is an independent, one-shape campaign.  Importing it is CPU-only.  A real
run retains run_study's leases, driver/telemetry checks, numerical oracle,
device decision audit, bridge accounting, and owned-loader cleanup.
"""
import argparse
import json
from pathlib import Path
import random

import bench
import run_study as base

HERE = Path(__file__).resolve().parent
ARMS = ('pod_inline', 'pod_cuda', 'pod_bpf')
PROTOCOL = 'pod-device-setup-phases-v1'
SEED = 20260903
FIXED_SHAPE = ('llama-3-8b', 32)


def orders(mode):
    if mode not in ('preflight', 'full'):
        raise ValueError('unknown phase-study mode')
    rng = random.Random(SEED)
    result = []
    for block in range(1, 2 if mode == 'preflight' else 6):
        arms = list(ARMS)
        rng.shuffle(arms)
        result.extend({'block': block, 'arm': arm} for arm in arms)
    return result


def exact_targets(extraction):
    targets = (extraction / 'exact-kernels.txt').read_text().splitlines()
    if (len(targets) != 6 or len(set(targets)) != 6
            or any('true_fused_tb_fwd_kernel' not in target for target in targets)):
        raise ValueError('phase study requires the frozen six-target POD inventory')
    return targets


def durations(execution, report):
    parent = execution['phase_timestamps']
    child = report['phase_timestamps']
    result = {
        'pre_python_main_ns': child['process_main_ns'] - parent['client_spawn_ns'],
        'stdlib_imports_ns': child['stdlib_imports_done_ns'] - child['process_main_ns'],
        'runtime_imports_ns': child['runtime_imports_done_ns'] - child['runtime_imports_start_ns'],
        'post_import_setup_ns': child['pre_first_diagnostic_ns'] - child['runtime_imports_done_ns'],
        'first_diagnostic_sync_ns': child['post_first_sync_ns'] - child['pre_first_diagnostic_ns'],
        'warmups_ns': child['warmup_done_ns'] - child['post_first_sync_ns'],
        'steady_samples_ns': child['steady_complete_ns'] - child['warmup_done_ns'],
        'post_steady_process_ns': parent['client_exit_ns'] - child['steady_complete_ns'],
        'post_process_cleanup_ns': parent['cleanup_complete_ns'] - parent['client_exit_ns'],
        'whole_cell_ns': parent['cleanup_complete_ns'] - parent['cell_start_ns'],
        'loader_ready_ns': (parent['loader_ready_ns'] - parent['loader_spawn_ns']
                            if execution['arm'] == 'pod_bpf' else None),
    }
    if any(type(value) is not int or value < 0 for value in result.values() if value is not None):
        raise ValueError('negative/non-integer derived phase duration')
    return result


def validate_cell(directory, item, preflight, runtime, targets):
    execution = json.loads((directory / 'execution.json').read_text())
    report = json.loads((directory / 'operator.json').read_text())
    arm, block = item['arm'], item['block']
    if (execution.get('status') != 'passed' or execution.get('phase_study') is not True
            or execution.get('numeric_protocol') != bench.NUMERIC_PROTOCOL
            or execution.get('arm') != arm or execution.get('block') != block
            or execution.get('runtime_before') != runtime or execution.get('runtime_after') != runtime
            or execution.get('cleanup_errors')):
        raise ValueError('phase cell has failed execution/runtime/cleanup evidence')
    base.validate_report(report, arm, block, preflight, phase_study=True)
    base.validate_phase_execution(execution, report, arm)
    if report['shape_order'] != [list(FIXED_SHAPE)] or len(report['cells']) != 1:
        raise ValueError('phase cell changed the one frozen operator shape')

    cell = report['cells'][0]
    launched = [] if arm == 'pod_inline' else cell['launch_bridge'].get('first_launches', [])
    if arm != 'pod_inline':
        if len(launched) != 1 or launched[0]['kernel'] not in targets:
            raise ValueError('actual fixed-shape launch is not one frozen attachment target')
    if arm == 'pod_bpf':
        loader_log = (directory / 'loader.log').read_text()
        expected_inventory = (Path(execution['environment']['BPFTIME_CUDA_LATE_PTX_DIR']).parent
                              / 'exact-kernels.txt').resolve()
        if (loader_log.count('POD_LOADER_READY kernels=6\n') != 1
                or loader_log.count('POD_LOADER_CLOSED\n') != 1
                or execution.get('private_segment_removed') is not True
                or Path(execution['loader_command'][-1]).resolve() != expected_inventory):
            raise ValueError('six-target loader lifetime/cleanup evidence is incomplete')
    elif execution.get('private_segment') is not None:
        raise ValueError('non-BPF phase arm unexpectedly owns a policy segment')

    first_by_name = {record['kernel']: record['monotonic_ns'] for record in launched}
    target_observations = [
        {'kernel': target, 'registered_by_loader': arm == 'pod_bpf',
         'first_launch_ns': first_by_name.get(target)} for target in targets
    ] if arm == 'pod_bpf' else None
    return {
        'complete': True,
        'protocol': PROTOCOL,
        'numeric_protocol': bench.NUMERIC_PROTOCOL,
        'arm': arm,
        'block': block,
        'preflight': preflight,
        'fixed_shape': list(FIXED_SHAPE),
        'operator_timestamps': report['phase_timestamps'],
        'coordinator_timestamps': execution['phase_timestamps'],
        'durations': durations(execution, report),
        'actual_first_launches': launched,
        'attachment_targets': target_observations,
        'registered_target_count': 6 if arm == 'pod_bpf' else None,
        'launched_target_count': len(launched),
        'scope': ('The fixed shape launches one target. Null first_launch_ns values mean '
                  'that this cell did not launch that registered alternative; they are not '
                  'evidence of six-kernel launch coverage.'),
    }


def validate_matched_block(directory, block):
    reports = {
        arm: json.loads((directory / f'block-{block:02d}-{arm}' / 'operator.json').read_text())
        for arm in ARMS
    }
    metadata = {}
    fixed_fields = ('nsmid', 'grid_ctas', 'prefill_blocks', 'decode_blocks',
                    'factor_p', 'factor_d', 'smem_bytes', 'threads', 'fused_op', 'trace')
    for arm, report in reports.items():
        cell = report['cells'][0]
        metadata[arm] = {key: cell['diagnostic']['metadata'][key] for key in fixed_fields}
    if len({json.dumps(value, sort_keys=True) for value in metadata.values()}) != 1:
        raise ValueError('three phase arms did not execute the same fused work shape')
    first = {
        arm: reports[arm]['cells'][0]['launch_bridge']['first_launches'][0]['kernel']
        for arm in ('pod_cuda', 'pod_bpf')
    }
    if len(set(first.values())) != 1:
        raise ValueError('CUDA-control and BPF arms did not launch the same exact kernel')
    return {'block': block, 'metadata': metadata['pod_inline'],
            'adapter_first_kernel': first['pod_cuda']}


def validate_preflight(directory, runtime, targets):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest.get('complete') is not True or manifest.get('protocol') != PROTOCOL
            or manifest.get('numeric_protocol') != bench.NUMERIC_PROTOCOL
            or manifest.get('mode') != 'preflight' or manifest.get('order') != orders('preflight')
            or manifest.get('completed') != orders('preflight')
            or manifest.get('matched_blocks') != [validate_matched_block(directory, 1)]
            or manifest.get('arms') != list(ARMS)
            or manifest.get('fixed_shape') != list(FIXED_SHAPE)
            or manifest.get('warmups') != 10 or manifest.get('samples_per_cell') != 3
            or manifest.get('fresh_process_per_cell') is not True
            or manifest.get('runtime') != runtime or manifest.get('exact_targets') != targets
            or manifest.get('excluded_from_formal') is not True):
        raise ValueError('full phase study requires its complete unchanged-runtime preflight')
    for item in orders('preflight'):
        cell = directory / f"block-{item['block']:02d}-{item['arm']}"
        expected = validate_cell(cell, item, True, runtime, targets)
        if json.loads((cell / 'phase.json').read_text()) != expected:
            raise ValueError('preflight phase summary differs from validated raw records')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'full'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ptx', type=Path, default=HERE / 'build/ptx-runtime-01')
    parser.add_argument('--preflight', type=Path,
                        help='required for full: prior complete three-arm phase preflight')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite an existing phase campaign')

    extraction = args.ptx.resolve(strict=True)
    runtime_paths = base.preparation(extraction) + [Path(__file__).resolve()]
    runtime = base.file_inventory(runtime_paths)
    targets = exact_targets(extraction)
    if args.mode == 'full':
        if args.preflight is None:
            parser.error('full requires --preflight from this unchanged phase runtime')
        validate_preflight(args.preflight, runtime, targets)
    base.require_no_build()

    output = args.output.resolve()
    output.mkdir(parents=True)
    manifest = {
        'complete': False,
        'protocol': PROTOCOL,
        'numeric_protocol': bench.NUMERIC_PROTOCOL,
        'mode': args.mode,
        'order': orders(args.mode),
        'arms': list(ARMS),
        'seed': SEED,
        'fixed_shape': list(FIXED_SHAPE),
        'warmups': 10,
        'samples_per_cell': 3 if args.mode == 'preflight' else 100,
        'fresh_process_per_cell': True,
        'runtime': runtime,
        'ptx': str(extraction),
        'exact_targets': targets,
        'excluded_from_formal': args.mode == 'preflight',
        'preflight': str(args.preflight.resolve()) if args.preflight is not None else None,
        'completed': [],
        'matched_blocks': [],
        'lease_paths': ['/tmp/gpubpf-revision-gpu0.lock',
                        '/tmp/gpubpf-revision-struct-ops.lock'],
    }
    lease = None
    try:
        lease = base.shared.Leases()
        base.safety.atomic_write_json(output / 'manifest.json', manifest)
        for item in manifest['order']:
            directory = output / f"block-{item['block']:02d}-{item['arm']}"
            base.run_cell(directory, item, args.mode, extraction, runtime_paths,
                          runtime, phase_study=True)
            phase = validate_cell(directory, item, args.mode == 'preflight', runtime, targets)
            base.safety.atomic_write_json(directory / 'phase.json', phase)
            manifest['completed'].append(item)
            if sum(done['block'] == item['block'] for done in manifest['completed']) == len(ARMS):
                manifest['matched_blocks'].append(validate_matched_block(output, item['block']))
            base.safety.atomic_write_json(output / 'manifest.json', manifest)
        manifest['complete'] = True
    except BaseException as error:
        manifest['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        base.safety.atomic_write_json(output / 'manifest.json', manifest)
        if lease is not None:
            lease.close()


if __name__ == '__main__':
    main()
