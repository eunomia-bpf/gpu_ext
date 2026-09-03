#!/usr/bin/env python3
"""Strict offline audit and paired analysis of the complete POD operator study.

No GPU, torch import, admission, or producer mean is used for the estimates.
Saved numeric checks are verified as run attestations; this does not rerun
attention or reconstruct all numerical output tensors.
"""
import argparse
import json
import math
from pathlib import Path
import random
import re
import statistics

import bench
import run_study as run

SEED = 20260905
DRAWS = 10000
SHAPES = [(model, batch) for model in bench.MODEL_HEADS for batch in (32, 64, 96, 128, 192)]
COMPARISONS = [('device_bpf_vs_cuda_adapter', 'pod_bpf', 'pod_cuda'),
               ('device_bpf_vs_original_inline', 'pod_bpf', 'pod_inline'),
               ('cuda_adapter_vs_original_inline', 'pod_cuda', 'pod_inline')]
COMPARISONS += [(f'{arm}_vs_{baseline}', arm, baseline)
                for arm in ('pod_inline', 'pod_cuda', 'pod_bpf')
                for baseline in ('official_serial', 'official_streams')]


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    with path.open() as file:
        return json.load(file)


def percentile(values, q):
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_indices():
    rng = random.Random(SEED)
    return [[rng.randrange(5) for _ in range(5)] for _ in range(DRAWS)]


def paired_ratio(numerator, denominator, indices):
    require(len(numerator) == len(denominator) == 5, 'paired statistic requires all five blocks')
    require(all(type(x) in (int, float) and math.isfinite(x) and x > 0
                for x in numerator + denominator), 'nonpositive/nonfinite cell mean')
    logs = [math.log(a / b) for a, b in zip(numerator, denominator)]
    draws = [math.exp(statistics.fmean(logs[i] for i in draw)) for draw in indices]
    return dict(geometric_mean_ratio=math.exp(statistics.fmean(logs)),
                block_ratios=[a / b for a, b in zip(numerator, denominator)],
                confidence_interval_95=[percentile(draws, .025), percentile(draws, .975)],
                blocks=5, lower_is_better=True)


def audit_execution(directory, arm, block, runtime, report):
    execution = read_json(directory / 'execution.json')
    require(execution.get('status') == 'passed' and execution.get('numeric_protocol') == bench.NUMERIC_PROTOCOL
            and not execution.get('error')
            and not execution.get('cleanup_errors') and execution.get('returncode') == 0,
            'failed execution or incomplete cleanup: ' + str(directory))
    require(execution.get('arm') == arm and execution.get('block') == block,
            'execution belongs to another arm/block')
    require(execution.get('runtime_before') == runtime == execution.get('runtime_after'),
            'runtime differs from the fixed campaign inventory')
    env = execution.get('environment', {})
    launch_env = execution.get('launch_environment')
    require(launch_env == {key: value for key, value in env.items() if key != 'LD_PRELOAD'},
            'wrapper inherited injection or target environment was not retained')
    command = execution.get('command', [])
    injection = ['/usr/bin/env', 'LD_PRELOAD=' + env['LD_PRELOAD']] if 'LD_PRELOAD' in env else []
    operator = command[3 + len(injection):]
    require(command[:3] == ['taskset', '-c', '8-15'] and command[3:3 + len(injection)] == injection
            and len(operator) == 8 and Path(operator[1]).name == 'bench.py'
            and operator[2:7] == ['--arm', arm, '--block', str(block), '--output']
            and Path(operator[7]).name == 'operator.json' and Path(operator[7]).parent.name == directory.name,
            'not the formal fixed operator command')
    require(env.get('CUDA_VISIBLE_DEVICES') == '0' and env.get('OMP_NUM_THREADS') == '1'
            and env.get('MKL_NUM_THREADS') == '1' and env.get('OPENBLAS_NUM_THREADS') == '1',
            'client GPU/CPU environment differs')
    preloads = [Path(x).name for x in env.get('LD_PRELOAD', '').split(':') if x]
    bpf_keys = {key for key in env if key.startswith('BPFTIME_')}
    loader_path = directory / 'loader.log'
    segment = execution.get('private_segment')
    if arm == 'pod_bpf':
        require(preloads == ['libpod_launch_bridge.so', 'libbpftime-agent.so']
                and env.get('POD_LAUNCH_BRIDGE') == 'bpf', 'missing real BPF/bridge injection')
        require(isinstance(segment, str) and re.fullmatch(r'pod_attention_[0-9]+_[0-9]+', segment)
                and env.get('BPFTIME_GLOBAL_SHM_NAME') == segment
                and type(execution.get('private_segment_removed')) is bool,
                'private loader lifecycle not recorded')
        require(env.get('BPFTIME_CUDA_DEFER_PTX_EXTRACTION') == '1'
                and env.get('BPFTIME_CUDA_DISABLE_CUOBJDUMP') == '1'
                and Path(env.get('BPFTIME_CUDA_LATE_PTX_DIR', '')).name == 'device'
                and Path(env.get('BPFTIME_PTXPASS_LIBRARIES', '')).name == 'libpod_ptx_adapter.so'
                and 'BPFTIME_RUN_WITH_KERNEL' not in env, 'wrong actual device-BPF preparation path')
        loader_env = execution.get('loader_environment', {})
        loader_command = execution.get('loader_command', [])
        require(Path(loader_env.get('LD_PRELOAD', '')).name == 'libbpftime-syscall-server.so'
                and loader_env.get('BPFTIME_GLOBAL_SHM_NAME') == segment
                and 'BPFTIME_RUN_WITH_KERNEL' not in loader_env
                and len(loader_command) == 3
                and [Path(x).name for x in loader_command] == ['pod-loader', 'selector.bpf.o', 'exact-kernels.txt'],
                'wrong syscall-server loader or exact selector input')
        loader_log = loader_path.read_text()
        require(loader_log.count('POD_LOADER_READY kernels=6\n') == 1
                and loader_log.count('POD_LOADER_CLOSED\n') == 1,
                'missing or duplicate real loader readiness/detach')
    else:
        require(not bpf_keys and segment is None and not loader_path.exists(),
                'non-BPF arm unexpectedly used private policy injection')
        if arm == 'pod_cuda':
            require(preloads == ['libpod_launch_bridge.so'] and env.get('POD_LAUNCH_BRIDGE') == 'cuda',
                    'CUDA adapter is not the shared launch-path control')
        else:
            require(not preloads and env.get('POD_LAUNCH_BRIDGE', 'off') == 'off',
                    'original arm unexpectedly used the launch adapter')
    log = (directory / 'client.log').read_text()
    require('Traceback (most recent call last)' not in log and 'POD_BRIDGE_FATAL' not in log,
            'client contains a numerical/runtime failure')
    printed = re.findall(r'^POD_CELL arm=(\S+) model=(\S+) bs=(\d+) mean_cuda_ms=(\S+)$', log, re.M)
    require(len(printed) == 10, 'missing or duplicate successful real operator log rows')
    for row, cell in zip(printed, report['cells']):
        require(row[:3] == (arm, cell['model'], str(cell['decode_batch']))
                and abs(float(row[3]) - statistics.fmean(s['cuda_ms'] for s in cell['samples'])) <= 5.01e-7,
                'logged real operator row differs from raw samples')
    before, after = execution['safety_before'], execution['safety_after']
    require(type(before.get('timestamp_ns')) is int and type(after.get('timestamp_ns')) is int
            and 0 < before['timestamp_ns'] < after['timestamp_ns'], 'invalid cell safety time window')
    run.safety.validate_pre_server_safety(before)
    run.safety.validate_post_server_safety(before, after)
    require(before['gpu']['driver'] == after['gpu']['driver'] == '575.57.08', 'wrong prepared driver')
    telemetry = run.safety.validate_gpu_telemetry(directory / 'gpu-telemetry.csv', allow_fixed_power_cap=True)
    require(telemetry == execution.get('telemetry'), 'raw telemetry differs from saved audit')
    return segment, before['timestamp_ns'], after['timestamp_ns']


def audit_characterization(directory, cell):
    rows = {}
    for phase, stats in cell['fp32_characterization'].items():
        if not stats['exceeding_elements']:
            continue
        diagnostic = directory / stats['diagnostic_directory']
        metadata = read_json(diagnostic / 'diagnostic.json')
        require(metadata.get('complete') is True and metadata.get('numeric_protocol') == bench.NUMERIC_PROTOCOL
                and metadata.get('model') == cell['model'] and metadata.get('decode_batch') == cell['decode_batch']
                and metadata.get('phase') == phase and metadata.get('seed') == cell['seed'],
                'extra-precision diagnostic belongs to a different real shape')
        for field in ('output_shape', 'checked_elements', 'exceeding_elements', 'max_abs_error',
                      'mean_abs_error', 'rms_error', 'atol', 'rtol'):
            require(metadata.get(field) == stats[field], 'saved row/full-shape characterization disagree')
        # This is a CPU recheck of the saved one-row diagnosis, not a full
        # numerical correctness gate and not a reinterpretation of v1 as passed.
        rows[phase] = bench.recompute_saved_fp64(diagnostic)
    return rows


def audit_campaign(directory):
    manifest = read_json(directory / 'manifest.json')
    expected = run.orders('full')
    require(manifest.get('mode') == 'full' and manifest.get('excluded_from_formal') is False
            and manifest.get('numeric_protocol') == bench.NUMERIC_PROTOCOL,
            'preflight or non-formal data cannot enter the performance comparison')
    require(manifest.get('complete') is True and not manifest.get('error')
            and manifest.get('order') == expected and manifest.get('completed') == expected
            and manifest.get('seed') == 20260903, 'incomplete, failed, duplicated or changed campaign order')
    require(manifest.get('lease_paths') == ['/tmp/gpubpf-revision-gpu0.lock',
                                          '/tmp/gpubpf-revision-struct-ops.lock'],
            'campaign did not declare the shared exclusive leases')
    runtime = manifest.get('runtime')
    require(isinstance(runtime, dict) and bool(runtime), 'missing fixed runtime inventory')
    require(isinstance(manifest.get('preflight'), str) and bool(manifest['preflight']),
            'formal run does not identify its required preflight')
    run.validate_preflight(Path(manifest['preflight']), runtime)
    names = {f"block-{item['block']:02d}-{item['arm']}" for item in expected}
    actual = {path.parent.relative_to(directory).as_posix() for path in directory.rglob('operator.json')}
    require(actual == names, 'missing, duplicate, nested-attempt or unexpected operator cells')
    actual_dirs = {p.name for p in directory.iterdir() if p.is_dir() and p.name.startswith('block-')}
    require(actual_dirs == names, 'unexpected/failed block directory')
    measurements = {shape: {arm: {} for arm in bench.ARMS} for shape in SHAPES}
    segments, previous_end, characterizations = set(), None, []
    for item in expected:
        arm, block = item['arm'], item['block']
        cell_dir = directory / f'block-{block:02d}-{arm}'
        report = read_json(cell_dir / 'operator.json')
        # Recompute claims, exactly-once work and 111 bridge launches from the
        # actual saved contexts. The same frozen typed ABI interpreter is used.
        run.validate_report(report, arm, block, False)
        segment, start, end = audit_execution(cell_dir, arm, block, runtime, report)
        require(previous_end is None or start >= previous_end, 'adjacent formal cell windows overlap or are out of order')
        previous_end = end
        if segment:
            require(segment not in segments, 'private segment reused across independent cells')
            segments.add(segment)
        for cell in report['cells']:
            shape = (cell['model'], cell['decode_batch'])
            measurements[shape][arm][block] = statistics.fmean(sample['cuda_ms'] for sample in cell['samples'])
            characterizations.append(dict(block=block, arm=arm, model=shape[0], decode_batch=shape[1],
                full_phase_statistics=cell['fp32_characterization'], saved_row_fp64=audit_characterization(cell_dir, cell)))
    return manifest, measurements, characterizations


def analyze(directory):
    directory = Path(directory)
    manifest, measured, characterizations = audit_campaign(directory)
    indices = bootstrap_indices()  # One shared whole-block resample across all comparisons/shapes.
    results = []
    for model, batch in SHAPES:
        means = {arm: [measured[(model, batch)][arm][block] for block in range(1, 6)] for arm in bench.ARMS}
        ratios = {}
        for label, numerator, denominator in COMPARISONS:
            ratios[label] = dict(numerator=numerator, denominator=denominator,
                                 **paired_ratio(means[numerator], means[denominator], indices))
        results.append(dict(model=model, decode_batch=batch, cell_mean_cuda_ms=means,
            mean_of_five_cell_means_ms={arm: statistics.fmean(values) for arm, values in means.items()},
            comparisons=ratios))
    return dict(complete=True, formal_complete=True, numeric_protocol=bench.NUMERIC_PROTOCOL,
        arm_processes=25, operator_cells=250,
        raw_cuda_event_observations=25000, per_cell_observations=100, warmups_per_cell=10,
        cell_estimator='arithmetic mean of all 100 unfiltered complete-operator CUDA-event samples',
        ratio_estimator='geometric mean of five paired block cell-mean latency ratios; lower is better',
        uncertainty=dict(method='whole-block percentile bootstrap, linear-interpolated quantiles',
                         draws=DRAWS, seed=SEED, confidence=.95,
                         scope='pointwise intervals; no multiple-comparison adjustment or formal equivalence test'),
        scope='official operator shapes on the recorded sm_120 compatibility build, not the full POD system',
        audit_scope='raw samples, saved numerical-validation attestations, actual CTA contexts, launch bridge, logs, cleanup and telemetry; no GPU numerical rerun',
        source_preflight=manifest['preflight'], results=results,
        fp32_characterizations=characterizations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error('refusing to overwrite an existing offline analysis')
    result = analyze(args.campaign)
    if args.output is None:
        print(json.dumps(result, indent=2))
    else:
        with args.output.open('x') as file:
            json.dump(result, file, indent=2)
            file.write('\n')


if __name__ == '__main__':
    main()
