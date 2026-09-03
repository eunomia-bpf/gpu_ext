#!/usr/bin/env python3
"""Plot a closed, complete HB pipeline audit; never rerun its analysis or GPU work."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'gpreempt'))
# Same shared paper style and palette as workloads/finemoe/plot_results.py.
from plot_scheduling_comparison import COLORS, STYLE

ARMS = ('idle_c_d1', 'idle_bpf_d1', 'idle_c_d2', 'idle_bpf_d2')
LABELS = ('C\nd1', 'BPF\nd1', 'C\nd2', 'BPF\nd2')
SCENARIOS = ('periodic', 'burstgpt')
METRICS = ('lc_p99_ms', 'be_goodput_rps')


def number(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError(f'{name}: missing/nonfinite/invalid metric; never omit a cell or substitute zero')
    return float(value)


def plot_points(audit):
    for key in ('complete', 'formal_complete', 'pipeline_exercised', 'causal_interpretation_ready'):
        if audit.get(key) is not True:
            raise ValueError(f'requires the complete, exercised formal audit: {key}')
    if (audit.get('mode') != 'full'
            or any(type(audit.get(k)) is not int or audit[k] != 40
                   for k in ('required_cells', 'accepted_cells'))
            or any(audit.get(k) != [] for k in ('pending', 'rejected', 'unexpected'))):
        raise ValueError('requires exactly 40 accepted formal cells, without rejected/pending/extra cells')
    stats = audit.get('statistics', {})
    if any(type(stats.get(k)) is not int or stats[k] != v for k, v in
           (('paired_blocks', 5), ('draws', 10000), ('seed', 20260903))):
        raise ValueError('the frozen five-block statistics protocol differs')
    cells = audit.get('cells', [])
    expected = {(block, scenario, arm) for block in range(5) for scenario in SCENARIOS for arm in ARMS}
    if (not isinstance(cells, list) or len(cells) != 40
            or any(type(c.get('block')) is not int for c in cells)
            or {(c['block'], c.get('scenario'), c.get('arm')) for c in cells} != expected):
        raise ValueError('missing, duplicate, or unexpected block/scenario/arm')
    if any(audit.get('scenarios', {}).get(s, {}).get('complete_blocks') != list(range(5)) for s in SCENARIOS):
        raise ValueError('both arrivals require all five complete paired blocks')
    points = []
    for cell in sorted(cells, key=lambda c: (SCENARIOS.index(c['scenario']), c['block'], ARMS.index(c['arm']))):
        arm, bound = cell['arm'].rsplit('_d', 1)
        bound = int(bound)
        executor = cell['engagement']['executor']
        if (cell.get('actual_arm') != arm
                or any(type(executor.get(k)) is not int or executor[k] != bound
                       for k in ('configured_lp_inflight_bound', 'max_lp_inflight'))
                or type(executor.get('decisions')) is not int or executor['decisions'] <= 0
                or type(executor.get('jit_decisions')) is not int
                or executor['jit_decisions'] != (executor['decisions'] if arm == 'idle_bpf' else 0)
                or any(type(cell.get(k)) is not int or cell[k] <= 0 for k in ('begin_ns', 'end_ns'))
                or cell['end_ns'] - cell['begin_ns'] != 60_000_000_000):
            raise ValueError('wrong actual policy/bound/JIT evidence or nonformal measurement window')
        lc, be = cell['metrics']['vgg_rt'], cell['metrics']['resnet152_be']
        coverage = number(lc.get('completion_coverage'), 'LC window coverage')
        if (coverage > 1 or type(lc.get('conditional_p99')) is not bool
                or (coverage == 1 and lc['conditional_p99'])):
            raise ValueError('invalid LC coverage/conditional-p99 metadata')
        points.append({'block': cell['block'], 'scenario': cell['scenario'], 'arm': cell['arm'],
                       'lc_p99_ms': number(lc.get('response_p99_ns'), 'LC p99', positive=True) / 1e6,
                       'be_goodput_rps': number(be.get('goodput_rps'), 'BE goodput'),
                       'lc_incomplete_coverage': coverage < 1, 'lc_conditional_p99': lc['conditional_p99']})
    return points


def draw(points, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # Exactly the final two-column width: no tight-bbox canvas resizing.
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(2, 2, figsize=(7, 4.65), sharey='col')
        for row, (scenario, arrival) in enumerate(zip(SCENARIOS, ('Periodic', 'BurstGPT-derived'))):
            for col, (metric, ylabel) in enumerate(zip(METRICS,
                    ('LC arrival-to-verified\np99 (ms, ↓)', 'BE verified-in-window\ngoodput (req/s, ↑)'))):
                panel = axes[row, col]
                for index, arm in enumerate(ARMS):
                    cells = sorted((p for p in points if p['scenario'] == scenario and p['arm'] == arm),
                                   key=lambda p: p['block'])
                    bpf = arm.startswith('idle_bpf')
                    color, marker, hatch = (COLORS[2], 'D', '///') if bpf else (COLORS[1], 's', '')
                    panel.bar(index, statistics.median(p[metric] for p in cells), width=.68,
                              color=color, alpha=.4, edgecolor='#333333', linewidth=1, hatch=hatch, zorder=2)
                    for point in cells:
                        x = index + (point['block'] - 2) * .11
                        panel.scatter(x, point[metric], s=20, marker=marker, facecolors='white',
                                      edgecolors=color, linewidths=1, clip_on=False, zorder=3)
                        if col == 0 and point['lc_incomplete_coverage']:
                            panel.plot(x, point[metric], marker='x', markersize=4,
                                       color='#222222', markeredgewidth=1, zorder=4)
                panel.set_xticks(range(4), LABELS)
                panel.set_xlim(-.55, 3.55)
                upper = max(p[metric] for p in points)
                panel.set_ylim(0, upper * 1.16 if upper > 0 else 1)
                panel.set_ylabel(ylabel)
                panel.set_xlabel(f'{arrival} arrivals\nPolicy / event bound')
                panel.yaxis.set_major_locator(MaxNLocator(nbins=5))
                panel.ticklabel_format(axis='y', style='plain', useOffset=False)
                panel.grid(axis='y', alpha=.25, linewidth=.6, zorder=0)
                panel.text(.025, .975, f'({chr(97 + row * 2 + col)})', transform=panel.transAxes,
                           va='top', fontsize=8)
        figure.text(.5, .02,
                    'Bars: descriptive medians; markers: all five blocks. ×: incomplete LC window coverage.',
                    ha='center', fontsize=7.5)
        figure.subplots_adjust(left=.09, right=.99, bottom=.22, top=.98, hspace=.62, wspace=.38)
        try:
            for suffix in ('.pdf', '.png'):
                figure.savefig(prefix.with_suffix(suffix), dpi=300)
        finally:
            plt.close(figure)


def render(analysis_path, prefix):
    points = plot_points(json.loads(analysis_path.read_text()))
    if any(prefix.with_suffix(ext).exists() for ext in ('.pdf', '.png')):
        raise FileExistsError('choose a fresh output prefix; existing figures are retained')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    draw(points, prefix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    render(args.analysis, args.output_prefix)
