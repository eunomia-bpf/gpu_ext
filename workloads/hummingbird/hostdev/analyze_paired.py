#!/usr/bin/env python3
"""Analyze the paired host-device BPF runs from flat block-XX-arm.log files.

Pure log analysis: it launches nothing and adds no GPU/existence gate. It reads
a directory of flat logs named `block-XX-<arm>.log` plus the retained first BPF
cell (block-00 bpf_device) from a separate path.

For every cell it recomputes, from the real GP_LOAD_STUDY request records (via
run_study.measurement), the HP (vgg) scheduled-arrival response p99, the BE
(resnet152) completed-in-window goodput (/60 s), and the numerical check.
Missing, partial, or failed cells are recorded and skipped, never fatal.

Paired ratios decompose the BPF device mechanism into four arms (numerator over
denominator, per block; reported as geometric mean, median, range, and paired
block-bootstrap CI95, with block 0 excluded for sensitivity):
  host_bpf          = inline_bpfhost  / original_inline   (host BPF policy only)
  callable_adapter  = native_adapter  / inline_bpfhost    (callable + 48-byte ctx)
  incremental_bpf   = bpf_device      / native_adapter    (BPF vs native compute)
  total_bpf         = bpf_device      / original_inline   (whole mechanism)

The existing unsplit native no-policy baseline stays historical and is not
relabeled here.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
HB = HERE.parent
sys.path.insert(0, str(HB))
sys.path.insert(0, str(HB.parent / 'gpreempt'))

HP_TASK = 'vgg_rt'
BE_TASK = 'resnet152_be'

ARM_DEFS = {
    'original_inline': ('idle_c', 'original'),
    'inline_bpfhost': ('idle_bpf', 'original'),
    'native_adapter': ('idle_bpf', 'adapter'),
    'bpf_device': ('idle_bpf', 'bpf'),
}
FLAT_LOG = re.compile(r'^block-(\d+)-([a-z_]+)\.log$')

# numerator / denominator arm for each named effect.
PAIRS = {
    'host_bpf': ('inline_bpfhost', 'original_inline'),
    'callable_adapter': ('native_adapter', 'inline_bpfhost'),
    'incremental_bpf': ('bpf_device', 'native_adapter'),
    'total_bpf': ('bpf_device', 'original_inline'),
}
METRICS = ('hp_p99_ns', 'be_goodput_rps')


def load_json(path: Path):
    with open(path, 'rb') as handle:
        return json.loads(handle.read())


def measure_cell(log: Path, cfg: dict) -> dict:
    import run_study
    measured = run_study.measurement(log.read_text(errors='replace'), cfg, None)
    metrics = measured['metrics']
    return {
        'hp_p99_ns': metrics[HP_TASK]['response_p99_ns'],
        'hp_coverage': metrics[HP_TASK]['completion_coverage'],
        'be_goodput_rps': metrics[BE_TASK]['goodput_rps'],
        'be_completed_in_window': metrics[BE_TASK]['completed_in_window'],
        'be_max_abs_error': metrics[BE_TASK]['numerics']['max_absolute_error'],
    }


def block_ratios(num_arm, den_arm, metric, data, exclude_block0):
    ratios = []
    for block in sorted(set(data.get(metric, {}).get(num_arm, {}))):
        if exclude_block0 and block == 0:
            continue
        num = data.get(metric, {}).get(num_arm, {}).get(block)
        den = data.get(metric, {}).get(den_arm, {}).get(block)
        if num and den:
            ratios.append(num / den)
    return ratios


def summarize_estimate(values):
    import analyze_three_way
    if len(values) < 2:
        return {'n': len(values), 'note': 'need >=2 paired blocks'}
    est = analyze_three_way.estimate_ratios(values)
    est['median'] = statistics.median(values)
    est['range'] = [min(values), max(values)]
    return est


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir', default=str(HERE / 'raw-paired-20260908.uDB4jK'),
                   help='directory of flat block-XX-arm.log files')
    p.add_argument('--retained-bpf',
                   default=str(HERE / 'raw-first-real-20260908.08lUpN'
                                '/full-bpf-compatible-core.log'),
                   help='retained first BPF cell (block-00 bpf_device) log')
    p.add_argument('--study', default=str(HB / 'raw/idle-study-575-01'))
    p.add_argument('--out', default=str(HERE / 'analysis-paired.json'))
    args = p.parse_args()

    study = Path(args.study)
    d = Path(args.dir)
    cells = {}
    if d.is_dir():
        for path in sorted(d.iterdir()):
            match = FLAT_LOG.match(path.name)
            if match and match.group(2) in ARM_DEFS:
                cells[(int(match.group(1)), match.group(2))] = path
    retained = Path(args.retained_bpf)
    if retained.exists():
        cells[(0, 'bpf_device')] = retained

    analysis = {'cells': [], 'estimate': {}}
    data = {'hp_p99_ns': {}, 'be_goodput_rps': {}}
    for (block, arm) in sorted(cells):
        entry = {'block': block, 'arm': arm, 'log': str(cells[(block, arm)])}
        cfg_path = study / f'block-{block:02d}' / 'periodic' / 'idle_bpf' / 'config.json'
        try:
            m = measure_cell(cells[(block, arm)], load_json(cfg_path))
            entry.update(m)
            entry['status'] = 'measured'
            data['hp_p99_ns'].setdefault(arm, {})[block] = m['hp_p99_ns']
            data['be_goodput_rps'].setdefault(arm, {})[block] = m['be_goodput_rps']
        except Exception as exc:
            entry['status'] = 'failed_or_partial'
            entry['analysis_error'] = f'{type(exc).__name__}: {exc}'
        analysis['cells'].append(entry)

    for pair, (num, den) in PAIRS.items():
        for metric in METRICS:
            for label, excl in (('all_blocks', False), ('excl_block0', True)):
                values = block_ratios(num, den, metric, data, excl)
                analysis['estimate'].setdefault(pair, {})[f'{metric}_{label}'] = \
                    summarize_estimate(values)

    out = Path(args.out)
    out.write_text(json.dumps(analysis, indent=2) + '\n')

    print(f"{'arm':<18}{'blk':>4}  {'status':<18}{'HP_p99_ns':>11}"
          f"{'BE_goodput':>12}{'BE_maxerr':>9}")
    for entry in analysis['cells']:
        goodput = entry.get('be_goodput_rps')
        print(f"{entry['arm']:<18}{entry['block']:>4}  {entry['status']:<18}"
              f"{str(entry.get('hp_p99_ns', '-')):>11}"
              f"{(f'{goodput:.2f}' if goodput is not None else '-'):>12}"
              f"{str(entry.get('be_max_abs_error', '-')):>9}")
    print('\npaired ratios (numerator/denominator, per block):')
    for pair, (num, den) in PAIRS.items():
        for metric in METRICS:
            for label, _ in (('all_blocks', False), ('excl_block0', True)):
                est = analysis['estimate'][pair][f'{metric}_{label}']
                if est.get('geometric_ratio') is not None:
                    ci = est['paired_block_bootstrap_ci95']
                    print(f"  {pair:<17}{metric:<15}{label:<11} "
                          f"geo={est['geometric_ratio']:.4f} "
                          f"median={est['median']:.4f} "
                          f"range=[{est['range'][0]:.4f},{est['range'][1]:.4f}] "
                          f"ci95=[{ci[0]:.4f},{ci[1]:.4f}] n={est.get('n', len(est.get('block_ratios', [])))}")
                else:
                    print(f"  {pair:<17}{metric:<15}{label:<11} "
                          f"{est.get('note', 'insufficient paired blocks')}")
    print(f'\nanalysis -> {out}')


if __name__ == '__main__':
    main()
