#!/usr/bin/env python3
"""Offline POD current-runtime campaign summarizer; reads retained records only.

Scans one campaign root for child cell directories holding operator.json and
optionally execution.json.  Every attempt is listed with child exit and errors
when known, the existing CUDA-event means, raw phase timestamps, and derived
pre-Python and client wall durations when spawn/exit stamps exist.  Only
completed attempts count as measurements; failed and still-running attempts
remain listed and are never counted.  No CUDA import, GPU work, or writes.
"""
import argparse
import json
import math
import statistics
from pathlib import Path


def read_json(path):
    try:
        with path.open() as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def stamp(record, key):
    """Integer timestamp at top level or inside phase_timestamps, else None."""
    if isinstance(record, dict):
        value = record.get(key)
        if type(value) is int:
            return value
        nested = record.get('phase_timestamps')
        if isinstance(nested, dict) and type(nested.get(key)) is int:
            return nested[key]
    return None


def span(later, earlier):
    return later - earlier if later is not None and earlier is not None else None


def finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def cell_row(name, path):
    report = read_json(path / 'operator.json')
    execution = read_json(path / 'execution.json')
    arm = report.get('arm') if isinstance(report, dict) else None
    block = report.get('block') if isinstance(report, dict) else None
    if isinstance(execution, dict):
        arm = arm if arm is not None else execution.get('arm')
        block = block if block is not None else execution.get('block')
    errors = []
    if isinstance(execution, dict):
        if execution.get('error'):
            errors.append('execution: ' + str(execution['error']))
        errors.extend('cleanup: ' + str(item)
                      for item in execution.get('cleanup_errors') or [])
    if isinstance(report, dict) and report.get('error'):
        errors.append('operator: ' + str(report['error']))
    exit_code = execution.get('returncode') if isinstance(execution, dict) else None
    complete = isinstance(report, dict) and report.get('complete') is True
    if complete and exit_code == 0 and execution.get('status') == 'passed' and not errors:
        state = 'completed'
    elif errors or exit_code not in (0, None) or (exit_code == 0 and report is not None and not complete):
        state = 'failed'
    else:
        state = 'incomplete'
    means, operator_phases = [], None
    if isinstance(report, dict):
        for cell in report.get('cells') or []:
            if isinstance(cell, dict):
                means.append({'model': cell.get('model'),
                              'decode_batch': cell.get('decode_batch'),
                              'warmups': cell.get('warmups'),
                              'timed_samples': len(cell['samples'])
                              if isinstance(cell.get('samples'), list) else None,
                              'mean_cuda_ms': cell.get('mean_cuda_ms')})
        if isinstance(report.get('phase_timestamps'), dict):
            operator_phases = report['phase_timestamps']
    if operator_phases is None and isinstance(execution, dict):
        operator_phases = execution.get('operator_timestamps')
    execution_phases = (execution.get('phase_timestamps')
                        if isinstance(execution, dict) else None)
    spawn = stamp(execution, 'client_spawn_ns')
    exit_ns = stamp(execution, 'client_exit_ns')
    first = means[0]['mean_cuda_ms'] if means else None
    return {
        'cell': name, 'arm': arm, 'block': block, 'state': state,
        'child_exit': exit_code, 'errors': errors,
        'cuda_event_means_ms': means,
        'mean_cuda_ms': first if finite_positive(first) else None,
        'pre_python_main_ns': span(stamp(operator_phases, 'process_main_ns'), spawn),
        'client_wall_ns': span(exit_ns, spawn),
        'loader_ready_ns': span(stamp(execution_phases, 'loader_ready_ns'),
                                stamp(execution_phases, 'loader_spawn_ns')),
        'cleanup_ns': span(stamp(execution_phases, 'cleanup_complete_ns'), exit_ns),
        'raw_timestamps': {'execution': execution_phases, 'operator': operator_phases},
    }


def summarize(rows):
    raw = {}
    for row in rows:
        if row['state'] != 'completed' or row['mean_cuda_ms'] is None or row['arm'] is None:
            continue
        entry = raw.setdefault(row['arm'], {})
        entry.setdefault('mean_cuda_ms', []).append(
            {'block': row['block'], 'value_ms': row['mean_cuda_ms']})
        for key in ('pre_python_main_ns', 'client_wall_ns', 'loader_ready_ns'):
            if finite_positive(row[key]):
                entry.setdefault(key, []).append(
                    {'block': row['block'], 'value_ms': row[key] / 1e6})
    grouped = {
        arm: {name: {'values': values,
                     'median_ms': statistics.median(v['value_ms'] for v in values)}
              for name, values in entry.items()}
        for arm, entry in raw.items()}
    per_block = {}
    for row in rows:
        if row['state'] == 'completed' and row['mean_cuda_ms'] is not None:
            per_block.setdefault(row['block'], {})[row['arm']] = row['mean_cuda_ms']
    ratios = []
    for block in sorted(per_block):
        entry = per_block[block]
        if finite_positive(entry.get('pod_bpf')) and finite_positive(entry.get('pod_cuda')):
            ratios.append({'block': block,
                           'pod_bpf_mean_cuda_ms': entry['pod_bpf'],
                           'pod_cuda_mean_cuda_ms': entry['pod_cuda'],
                           'bpf_over_cuda': entry['pod_bpf'] / entry['pod_cuda']})
    return grouped, ratios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path,
                        help='campaign root containing block-NN-arm cell directories')
    args = parser.parse_args()
    if not args.campaign.is_dir():
        parser.error('campaign root is not a directory')
    rows = [cell_row(path.name, path) for path in sorted(args.campaign.iterdir())
            if path.is_dir()
            and ((path / 'operator.json').is_file() or (path / 'execution.json').is_file())]
    grouped, ratios = summarize(rows)
    result = {
        'campaign': str(args.campaign),
        'cell_count': len(rows),
        'completed_count': sum(row['state'] == 'completed' for row in rows),
        'cells': rows,
        'completed_observations_ms': grouped,
        'bpf_cuda_operator_ratios': ratios,
        'scope': ('Each entry is one retained fresh-process attempt under this '
                  'campaign root. Only completed attempts count as measurements; '
                  'failed and incomplete attempts remain listed and are never '
                  'counted. Durations are null when spawn/exit stamps are '
                  'missing; nothing is estimated.'),
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
