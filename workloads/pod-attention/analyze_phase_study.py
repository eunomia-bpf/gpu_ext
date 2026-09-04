#!/usr/bin/env python3
"""Offline audit and paired phase analysis for the POD three-arm study.

The script reads only retained records.  It does not import CUDA/torch, attach
programs, or rerun the workload.  Phase timings are one observation per fresh
process; uncertainty resamples the five complete randomized blocks.
"""
import argparse
import json
import math
from pathlib import Path
import random
import re
import statistics

import bench
import run_phase_study as phase


SEED = 20260907
DRAWS = 10000
COMPARABLE_PHASES = (
    'coordinator_pre_client_ns',
    'client_lifetime_ns',
    'pre_python_main_ns',
    'stdlib_imports_ns',
    'pre_runtime_imports_ns',
    'runtime_imports_ns',
    'post_import_setup_ns',
    'first_diagnostic_sync_ns',
    'warmups_ns',
    'steady_samples_ns',
    'post_steady_process_ns',
    'post_process_cleanup_ns',
    'whole_cell_ns',
)
COMPARISONS = (
    ('device_bpf_vs_cuda_adapter', 'pod_bpf', 'pod_cuda'),
    ('device_bpf_vs_original_inline', 'pod_bpf', 'pod_inline'),
    ('cuda_adapter_vs_original_inline', 'pod_cuda', 'pod_inline'),
)


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    with path.open() as stream:
        return json.load(stream)


def percentile(values, quantile):
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_indices():
    rng = random.Random(SEED)
    return [[rng.randrange(5) for _ in range(5)] for _ in range(DRAWS)]


def paired_ratio(numerator, denominator, indices):
    require(len(numerator) == len(denominator) == 5,
            'paired phase statistic requires five complete blocks')
    values = numerator + denominator
    require(all(type(value) in (int, float) and math.isfinite(value) and value > 0
                for value in values), 'phase duration is nonpositive or nonfinite')
    logs = [math.log(left / right) for left, right in zip(numerator, denominator)]
    draws = [math.exp(statistics.fmean(logs[index] for index in draw))
             for draw in indices]
    return {
        'geometric_mean_ratio': math.exp(statistics.fmean(logs)),
        'block_ratios': [left / right for left, right in zip(numerator, denominator)],
        'confidence_interval_95': [percentile(draws, .025), percentile(draws, .975)],
        'blocks': 5,
        'lower_is_better': True,
    }


def archived_preflight(directory, recorded):
    require(isinstance(recorded, str) and Path(recorded).is_absolute()
            and '..' not in Path(recorded).parts,
            'formal campaign lacks an absolute recorded preflight path')
    local = directory.parent / Path(recorded).name
    return local if local.exists() or local.is_symlink() else Path(recorded)


def audit_safety(directory, execution, previous_end):
    before, after = execution.get('safety_before'), execution.get('safety_after')
    require(isinstance(before, dict) and isinstance(after, dict),
            'missing before/after safety record: ' + str(directory))
    start, end = before.get('timestamp_ns'), after.get('timestamp_ns')
    require(type(start) is int and type(end) is int and 0 < start < end,
            'invalid safety interval: ' + str(directory))
    require(previous_end is None or start >= previous_end,
            'formal cell safety intervals overlap or are out of order')
    phase.base.safety.validate_pre_server_safety(before)
    phase.base.safety.validate_post_server_safety(before, after)
    require(before['gpu']['driver'] == after['gpu']['driver'] == '575.57.08',
            'phase cell did not use the prepared driver')
    telemetry = phase.base.safety.validate_gpu_telemetry(
        directory / 'gpu-telemetry.csv', allow_fixed_power_cap=True)
    require(telemetry == execution.get('telemetry'),
            'saved telemetry summary differs from raw samples')
    return end


def audit_campaign(directory):
    directory = Path(directory)
    manifest = read_json(directory / 'manifest.json')
    expected = phase.orders('full')
    require(manifest.get('complete') is True and not manifest.get('error')
            and manifest.get('protocol') == phase.PROTOCOL
            and manifest.get('numeric_protocol') == bench.NUMERIC_PROTOCOL
            and manifest.get('mode') == 'full'
            and manifest.get('excluded_from_formal') is False
            and manifest.get('order') == expected
            and manifest.get('completed') == expected
            and manifest.get('seed') == phase.SEED,
            'campaign is incomplete, failed, preflight, or uses another protocol')
    require(manifest.get('arms') == list(phase.ARMS)
            and manifest.get('fixed_shape') == list(phase.FIXED_SHAPE)
            and manifest.get('warmups') == 10
            and manifest.get('samples_per_cell') == 100
            and manifest.get('fresh_process_per_cell') is True,
            'formal phase matrix or sampling protocol changed')
    require(manifest.get('lease_paths') == [str(path) for path in phase.LEASE_PATHS],
            'formal campaign did not declare both shared leases')
    runtime, targets = manifest.get('runtime'), manifest.get('exact_targets')
    require(isinstance(runtime, dict) and runtime and isinstance(targets, list),
            'missing runtime or target inventory')
    phase.validate_preflight(archived_preflight(directory, manifest.get('preflight')),
                             runtime, targets)

    expected_names = {f"block-{item['block']:02d}-{item['arm']}" for item in expected}
    actual_names = {path.name for path in directory.iterdir()
                    if path.is_dir() and path.name.startswith('block-')}
    require(actual_names == expected_names, 'missing, duplicate, or unexpected phase cell')
    require(manifest.get('matched_blocks') == [
        phase.validate_matched_block(directory, block) for block in range(1, 6)
    ], 'saved matched-work metadata differs from the retained arm records')

    durations = {name: {arm: {} for arm in phase.ARMS}
                 for name in COMPARABLE_PHASES}
    operator_latency = {name: {arm: {} for arm in phase.ARMS}
                        for name in ('cuda_ms', 'host_wall_ms')}
    loader_ready = {}
    previous_end = None
    for item in expected:
        arm, block = item['arm'], item['block']
        cell = directory / f'block-{block:02d}-{arm}'
        summary = phase.validate_cell(cell, item, False, runtime, targets)
        require(read_json(cell / 'phase.json') == summary,
                'saved phase summary differs from recomputed raw records')
        execution = read_json(cell / 'execution.json')
        require(execution.get('returncode') == 0 and not execution.get('error')
                and not execution.get('cleanup_errors'),
                'phase process failed or cleanup was incomplete')
        previous_end = audit_safety(cell, execution, previous_end)

        report = read_json(cell / 'operator.json')
        samples = report['cells'][0]['samples']
        row = re.findall(r'^POD_CELL arm=(\S+) model=(\S+) bs=(\d+) mean_cuda_ms=(\S+)$',
                         (cell / 'client.log').read_text(), re.M)
        require(len(row) == 1 and row[0][:3] == (arm, phase.FIXED_SHAPE[0],
                                                 str(phase.FIXED_SHAPE[1]))
                and abs(float(row[0][3]) - report['cells'][0]['mean_cuda_ms']) <= 5.01e-7,
                'client success row is missing or differs from raw samples')
        for name in operator_latency:
            value = statistics.fmean(sample[name] for sample in samples)
            require(math.isfinite(value) and value > 0,
                    'operator latency is nonpositive or nonfinite')
            operator_latency[name][arm][block] = value
        for name in COMPARABLE_PHASES:
            value = summary['durations'].get(name)
            require(type(value) is int and value > 0, 'missing/nonpositive phase duration')
            durations[name][arm][block] = value
        if arm == 'pod_bpf':
            value = summary['durations'].get('loader_ready_ns')
            require(type(value) is int and value > 0, 'missing BPF loader-ready duration')
            loader_ready[block] = value
        else:
            require(summary['durations'].get('loader_ready_ns') is None,
                    'non-BPF arm reports a loader-ready duration')
    require(set(loader_ready) == set(range(1, 6)), 'missing BPF loader observations')
    return manifest, durations, loader_ready, operator_latency


def analyze(directory):
    manifest, measured, loader_ready, operator_measured = audit_campaign(directory)
    indices = bootstrap_indices()
    block_ms = {
        name: {arm: [measured[name][arm][block] / 1e6 for block in range(1, 6)]
               for arm in phase.ARMS}
        for name in COMPARABLE_PHASES
    }
    ratios = {
        label: {
            name: paired_ratio(block_ms[name][numerator], block_ms[name][denominator], indices)
            for name in COMPARABLE_PHASES
        }
        for label, numerator, denominator in COMPARISONS
    }
    operator_block_ms = {
        name: {arm: [operator_measured[name][arm][block] for block in range(1, 6)]
               for arm in phase.ARMS}
        for name in operator_measured
    }
    operator_ratios = {
        label: {
            name: paired_ratio(operator_block_ms[name][numerator],
                               operator_block_ms[name][denominator], indices)
            for name in operator_block_ms
        }
        for label, numerator, denominator in COMPARISONS
    }
    medians = {
        name: {arm: statistics.median(values) for arm, values in arms.items()}
        for name, arms in block_ms.items()
    }
    bpf_whole = block_ms['whole_cell_ns']['pod_bpf']
    bpf_pre_main = block_ms['pre_python_main_ns']['pod_bpf']
    bpf_steady = block_ms['steady_samples_ns']['pod_bpf']
    return {
        'complete': True,
        'formal_complete': True,
        'protocol': phase.PROTOCOL,
        'numeric_protocol': bench.NUMERIC_PROTOCOL,
        'fresh_process_cells': 15,
        'blocks': 5,
        'measured_operator_samples_per_cell': 100,
        'phase_observations_per_arm': 5,
        'phase_estimator': 'median of five fresh-process block durations',
        'ratio_estimator': 'geometric mean of five paired block ratios; lower is better',
        'uncertainty': {
            'method': 'whole-block percentile bootstrap with shared resamples',
            'draws': DRAWS,
            'seed': SEED,
            'confidence': .95,
            'scope': 'pointwise intervals; no equivalence test or multiple-comparison adjustment',
        },
        'median_phase_ms': medians,
        'block_phase_ms': block_ms,
        'paired_ratios': ratios,
        'operator_latency': {
            'cell_estimator': 'arithmetic mean of all 100 unfiltered synchronized samples',
            'block_means_ms': operator_block_ms,
            'median_of_five_cell_means_ms': {
                name: {arm: statistics.median(values) for arm, values in arms.items()}
                for name, arms in operator_block_ms.items()
            },
            'paired_ratios': operator_ratios,
        },
        'bpf_loader_ready_ms': {
            'blocks': [loader_ready[block] / 1e6 for block in range(1, 6)],
            'median': statistics.median(loader_ready.values()) / 1e6,
        },
        'bpf_share_of_whole_cell': {
            'pre_python_main_block_fractions': [left / right for left, right in zip(bpf_pre_main, bpf_whole)],
            'steady_samples_block_fractions': [left / right for left, right in zip(bpf_steady, bpf_whole)],
        },
        'source_preflight': manifest['preflight'],
        'claim_boundary': (
            'Fresh-process phases for one frozen POD adapter, operator shape, runtime, and RTX 5090. '
            'pre_python_main includes all work before Python main on this injection path; it is not '
            'a generic attach-latency estimate. steady_samples covers the complete 100-sample '
            'measurement loop, including correctness and decision audits outside each timed '
            'operator; it is not operator latency or an end-to-end serving workload.'
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error('refusing to overwrite an existing phase analysis')
    result = analyze(args.campaign)
    if args.output is None:
        print(json.dumps(result, indent=2))
    else:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
