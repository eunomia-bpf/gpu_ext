"""Read completed sweep cells and produce review plots without editing the paper."""
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

POLICIES = ['baseline', 'sched_only', 'prefetch_only', 'memory_only', 'combined']
LABELS = ['Default', 'Sched', 'Prefetch', 'Evict', 'Mem + Sched']
KERNELS = ['hotspot', 'gemm', 'kmeans_sparse']
RATIOS = [0.8, 1.0, 1.2, 1.5]


def allocated_bytes(kernel, requested, log):
    # Geometry follows the campaign's saved inputs, corroborated by stdout.
    if kernel == 'hotspot':
        side = max(1024, math.isqrt(requested // 12) // 16 * 16)
        assert f'grid={side}x{side}' in log
        return 12 * side * side
    if kernel == 'gemm':
        layer = 4096 * 11008 * 4
        layers = max(1, min(200, requested // layer))
        assert f'layers={layers},' in log
        return layers * layer + (4096 + 11008) * 4
    points = max(1000, requested // (3 * 4096 + 200 * 4))
    return points * (3 * 4096 + 200 * 4) + 6 * 200 * 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'preview')
    parser.add_argument('--speedup-only', action='store_true',
                        help='plot paired high-priority speedup over Default')
    args = parser.parse_args()
    capacity = json.loads((args.campaign / 'gpu-memory.json').read_text())['total_bytes']
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cells, groups, checked = [], defaultdict(list), []
    for path in sorted(args.campaign.glob('*/ratio-*/*/combined_comparison.csv')):
        kernel = path.parents[2].name
        requested_ratio = float(path.parents[1].name.removeprefix('ratio-'))
        rows = list(csv.DictReader(path.open()))
        identities = [(int(r['block']), r['arm']) for r in rows]
        assert len(set(identities)) == len(identities), path
        for r in rows:
            block, policy = int(r['block']), r['arm']
            assert policy in POLICIES and 0 <= block < 5
            d = path.parent / f'block{block:02d}_{policy}'
            meta = json.loads((d / 'meta.json').read_text())
            assert meta['states_after_spawn'] == {'high': 'T', 'low': 'T'}, d
            assert meta['common_release_origin'] and meta['independent_exit_observation']
            assert meta['size_factor'] == requested_ratio / 2 and meta['iterations'] == 1
            tools = []
            if policy in ('sched_only', 'combined'):
                tools.append('sched')
                text = (d / 'sched_tool.log').read_text()
                assert 'GPU Scheduler struct_ops attached.' in text, d
                hits = re.findall(r'policy_hit:\s+(\d+)', text)
                assert hits and int(hits[-1]) > 0, d
                assert 'setter_error:  0' in text, d
            if policy in ('prefetch_only', 'memory_only', 'combined'):
                tools.append('mem')
                expected = 'prefetch_pid_tree' if policy == 'prefetch_only' else 'prefetch_eviction_pid'
                assert Path(meta['mem_tool']).name == expected, d
                assert meta['mem_params'] == 'high=20/low=80'
                text = (d / 'mem_tool.log').read_text()
                assert 'Successfully loaded' in text, d
                for role in ('high', 'low'):
                    matches = re.findall(rf'{role.capitalize()} priority PID {r[role + "_pid"]}:\s+Current active chunks: \d+\s+Total activated: (\d+)', text)
                    assert matches and int(matches[-1]) > 0, d
            for tool in tools:
                assert meta[f'rc_{tool}_tool'] == 0, d
                assert float(r[f't_attach_{tool}']) < float(r['t_release']), d
            allocated, sizes = [], []
            for role in ('high', 'low'):
                assert r[f'{role}_rc'] == '0', d
                elapsed = float(r[f'{role}_latency_s'])
                assert math.isfinite(elapsed) and elapsed > 0
                assert abs(float(r[f't_exit_{role}']) - float(r['t_release']) - elapsed) < 0.000003
                raw = list(csv.DictReader((d / f'uvmbench_uvmbench_{role}_results.csv').open()))
                assert len(raw) == 1
                raw = raw[0]
                assert raw['kernel'] == kernel and raw['mode'] == 'uvm' and raw['iterations'] == '1'
                assert abs(float(raw['size_factor']) - requested_ratio / 2) < 0.00001
                sizes.append(int(raw['working_set_bytes']))
                log = (d / f'tenant_uvmbench_{role}.log').read_text()
                assert 'Stride Bytes: 4096' in log
                allocated.append(allocated_bytes(kernel, sizes[-1], log))
            assert sizes[0] == sizes[1]
            cell = dict(kernel=kernel, requested_ratio=requested_ratio,
                        actual_allocation_ratio=sum(allocated) / capacity,
                        tenant_requested_bytes=sizes[0], block=block, policy=policy,
                        high_s=float(r['high_latency_s']), low_s=float(r['low_latency_s']),
                        both_finished_s=max(float(r['high_latency_s']), float(r['low_latency_s'])),
                        high_kernel_ms=float(r['high_median_ms']), low_kernel_ms=float(r['low_median_ms']),
                        release_skew_us=1e6 * (float(r['t_cont_low']) - float(r['t_cont_high'])),
                        source=str(d.relative_to(args.campaign)))
            cells.append(cell)
            groups[(kernel, requested_ratio)].append(cell)
        checked.append({'source': str(path.relative_to(args.campaign)), 'completed_cells': len(rows)})
    complete = {key: rows for key, rows in groups.items() if len(rows) == 25}
    summaries = []
    for (kernel, ratio), rows in complete.items():
        assert len({r['actual_allocation_ratio'] for r in rows}) == 1
        assert len({r['tenant_requested_bytes'] for r in rows}) == 1
        baseline = {r['block']: r for r in rows if r['policy'] == 'baseline'}
        for policy in POLICIES:
            selected = [r for r in rows if r['policy'] == policy]
            assert sorted(r['block'] for r in selected) == list(range(5))
            summary = dict(kernel=kernel, requested_ratio=ratio, policy=policy,
                           actual_allocation_ratio=selected[0]['actual_allocation_ratio'], n=5)
            for metric in ('high_s', 'low_s', 'both_finished_s', 'high_kernel_ms', 'low_kernel_ms'):
                values = [r[metric] for r in selected]
                summary.update({metric + '_median': median(values), metric + '_min': min(values), metric + '_max': max(values)})
            speedups = [baseline[r['block']]['high_s'] / r['high_s'] for r in selected]
            summary.update(high_speedup_median=median(speedups),
                           high_speedup_min=min(speedups), high_speedup_max=max(speedups))
            summaries.append(summary)
    for name, rows in [('cells.csv', cells), ('summary.csv', summaries)]:
        if rows:
            with (output / name).open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    status = dict(completed_cells=len(cells), expected_cells=len(KERNELS) * len(RATIOS) * len(POLICIES) * 5, complete_points=len(complete),
                  expected_points=len(KERNELS) * len(RATIOS), inputs=checked,
                  missing_points=[dict(kernel=k, requested_ratio=r) for k in KERNELS for r in RATIOS if (k, r) not in complete],
                  max_release_skew_us=max((r['release_skew_us'] for r in cells), default=0))
    (output / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    plt.style.use('seaborn-v0_8-whitegrid')
    # Match the original all_kernels_stacked.pdf, including its bottom legend,
    # repeated axis labels, prominent titles, and thin gray frame/grid.
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8, 'pdf.fonttype': 42,
                         'axes.labelsize': 8, 'axes.titlesize': 9,
                         'axes.linewidth': .4, 'axes.edgecolor': '.75',
                         'grid.linewidth': .4, 'grid.color': '.75',
                         'xtick.labelsize': 8, 'ytick.labelsize': 8})
    plots = ([('high_speedup', 'Speedup (×)')] if args.speedup_only else
             [('high_s', 'High-priority time (s)'), ('low_s', 'Low-priority time (s)'),
              ('both_finished_s', 'Both completed (s)')])
    for metric, ylabel in plots:
        fig, axes = plt.subplots(1, 3, figsize=(3.33, 1.30))
        for ax, kernel, title in zip(axes, KERNELS, ['HotSpot', 'GEMM', 'K-Means']):
            for policy, label, color, marker, style in zip(POLICIES, LABELS,
                    ['#777777', '#3498db', '#2ecc71', '#9b59b6', '#e74c3c'],
                    ['o', 's', '^', 'v', 'D'], ['-', '-', '--', '--', '-']):
                data = sorted([r for r in summaries if r['kernel'] == kernel and r['policy'] == policy], key=lambda r:r['requested_ratio'])
                x = [r['actual_allocation_ratio'] for r in data]
                y = [r[metric + '_median'] for r in data]
                err = [[r[metric + '_median'] - r[metric + '_min'] for r in data],
                       [r[metric + '_max'] - r[metric + '_median'] for r in data]]
                if data:
                    if metric == 'high_speedup' and policy == 'baseline':
                        style = '--'
                    ax.errorbar(x, y, yerr=err, color=color, marker=marker, linestyle=style,
                                linewidth=1.3 if policy == 'combined' else 1.0, markersize=4,
                                markerfacecolor=color, markeredgewidth=.4,
                                elinewidth=.5, capsize=1, alpha=.85, label=label)
            if metric == 'high_speedup':
                ax.axhline(1, color='#777777', linestyle='--', linewidth=1, alpha=.85)
            ax.set_xlim(.7, 1.6); ax.set_xticks([.8, 1.0, 1.2, 1.5], ['0.8', '1', '1.2', '1.5']); ax.set_ylim(bottom=0)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.10)
            ax.yaxis.set_major_locator(MaxNLocator(3)); ax.set_axisbelow(True)
            ax.grid(axis='both', alpha=.8)
            ax.set_title(title, fontsize=9, pad=2)
            ax.set_ylabel(ylabel, labelpad=1)
        handles, labels = axes[0].get_legend_handles_labels()
        # Legend samples show line/marker styles without the error-bar glyphs.
        fig.legend([handle.lines[0] for handle in handles], labels,
                   loc='lower center', ncol=5, fontsize=8, frameon=False,
                   columnspacing=.35, handlelength=1.0, handletextpad=.2,
                   bbox_to_anchor=(.5, -.015), borderaxespad=0)
        fig.supxlabel('Oversubscription ratio', fontsize=8, y=.135)
        fig.subplots_adjust(left=.12, right=.98, bottom=.36, top=.86, wspace=.68)
        fig.savefig(output / f'{metric}.pdf')
        fig.savefig(output / f'{metric}.png', dpi=220)
        plt.close(fig)
    print(json.dumps({k:v for k,v in status.items() if k not in ('inputs', 'missing_points')}, indent=2))


if __name__ == '__main__':
    main()
