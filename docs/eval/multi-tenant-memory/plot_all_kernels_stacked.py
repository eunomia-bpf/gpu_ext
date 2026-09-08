#!/usr/bin/env python3
"""
Plot policy comparison with stacked bars showing concurrent vs solo execution.

Visualization:
- Bottom segment: Time when both processes are running together (contention)
- Top segment: Time when only one process is running alone

Usage:
    python plot_all_kernels_stacked.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 28,
    'axes.labelsize': 32,
    'axes.titlesize': 34,
    'legend.fontsize': 28,
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,
    'figure.dpi': 150,
})

# Selected configurations to plot (policy, high_param, low_param, label, is_sched)
SELECTED_CONFIGS = [
    ('no_policy', None, None, 'No Policy', False),
    ('sched_timeslice', 1000000, 200, 'Scheduler', True),
    ('prefetch_pid_tree', 0, 20, 'Prefetch(0,20)', False),
    ('prefetch_pid_tree', 20, 80, 'Prefetch(20,80)', False),
    ('prefetch_eviction_pid', 20, 80, 'Evict(20,80)', False),
]

# ---------------------------------------------------------------------------
# New four-arm combined comparison (ORIGINAL Fig.13 follow-up). These arms come
# from the combined runner's fresh 5-block campaign and use a DIFFERENT timing
# convention (completion measured from the common release origin) than the
# original single-round configs above. They are therefore shown as a separate,
# labeled group and are never pooled or paired with the old values.
# ---------------------------------------------------------------------------
COMBINED_KERNEL_DIRS = {
    'Hotspot': 'hotspot',
    'GEMM': 'gemm',
    'K-Means': 'kmeans',
}
NEW_ARMS = [
    ('baseline', 'Baseline'),
    ('memory_only', 'Memory'),
    ('sched_only', 'Sched'),
    ('combined', 'Combined'),
]


def find_combined_root(base_dir, explicit=None):
    """Locate the per-kernel combined results root directory."""
    if explicit:
        return Path(explicit)
    candidates = sorted(p for p in base_dir.glob('results_combined_*') if p.is_dir())
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda p: p.stat().st_mtime)


def find_combined_csv(combined_root, kernel_dir):
    """Find the newest combined_comparison.csv for one kernel under the root."""
    kernel_root = combined_root / kernel_dir
    if not kernel_root.is_dir():
        return None
    candidates = list(kernel_root.glob('combined_*/combined_comparison.csv'))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def aggregate_arm(df, arm):
    """Summarize one arm's repeated blocks: median completion + total range.

    Returns None if the arm has no rows. 'both' is the overlap (time both
    tenants run), 'solo' the remaining execution of the slower tenant, both
    derived from the per-tenant medians; 'total_min'/'total_max' span the
    per-block total completion (max of the two tenants).
    """
    sub = df[df['arm'] == arm]
    if sub.empty:
        return None
    high = sub['high_latency_s'].astype(float).values
    low = sub['low_latency_s'].astype(float).values
    med_high = float(np.median(high))
    med_low = float(np.median(low))
    total = np.maximum(high, low)
    return {
        'n': int(len(sub)),
        'median_high': med_high,
        'median_low': med_low,
        'both': min(med_high, med_low),
        'solo': abs(med_high - med_low),
        'total': max(med_high, med_low),
        'total_min': float(total.min()),
        'total_max': float(total.max()),
    }


def load_data(csv_path):
    """Load and preprocess the CSV data."""
    df = pd.read_csv(csv_path)
    return df


def get_selected_rows(df, sched_df=None):
    """Filter dataframe to only selected configurations."""
    rows = []
    for policy, hp, lp, label, is_sched in SELECTED_CONFIGS:
        source_df = sched_df if is_sched and sched_df is not None else df
        if source_df is None:
            continue
        if policy == 'no_policy':
            row = source_df[source_df['policy'] == 'no_policy']
        else:
            row = source_df[(source_df['policy'] == policy) &
                     (source_df['high_param'] == hp) &
                     (source_df['low_param'] == lp)]
        if len(row) > 0:
            r = row.iloc[0].to_dict()
            r['label'] = label
            rows.append(r)
    return rows


def print_improvements(data, sched_data):
    """Print improvement ratios compared to no_policy."""
    print("\n" + "=" * 80)
    print("IMPROVEMENT RATIOS (vs No Policy)")
    print("=" * 80)

    for kernel_name, df in data.items():
        print(f"\n### {kernel_name} ###")
        sched_df = sched_data.get(kernel_name)
        rows = get_selected_rows(df, sched_df)

        baseline = None
        for r in rows:
            if r['label'] == 'No Policy':
                baseline = r
                break

        if baseline is None:
            print("  No baseline found")
            continue

        baseline_total = max(baseline['high_latency_s'], baseline['low_latency_s'])

        print(f"  {'Config':<20} {'Both(s)':<10} {'LowOnly(s)':<10} {'Total(s)':<10} {'Impr':<12}")
        print(f"  {'-'*62}")

        for r in rows:
            high = r['high_latency_s']
            low = r['low_latency_s']
            both_time = min(high, low)
            solo_time = abs(high - low)
            total = max(high, low)
            total_impr = (baseline_total - total) / baseline_total * 100

            print(f"  {r['label']:<20} {both_time:<10.1f} {solo_time:<10.1f} {total:<10.1f} {total_impr:>+10.1f}%")


def plot_kernel_subplot(ax, df, kernel_name, sched_df=None):
    """Plot a single kernel's data with stacked bars."""
    rows = get_selected_rows(df, sched_df)

    if not rows:
        ax.set_title(f"{kernel_name} (No Data)")
        return ax

    labels = [r['label'] for r in rows]

    # Calculate stacked bar segments
    both_running = []  # min(high, low) - time when both are running
    solo_running = []  # |high - low| - time when only one is running

    for r in rows:
        high = r['high_latency_s']
        low = r['low_latency_s']
        both_running.append(min(high, low))
        solo_running.append(abs(high - low))

    x = np.arange(len(labels))
    width = 0.6

    # Plot stacked bars
    bars1 = ax.bar(x, both_running, width,
                   label='Both Running', color='#e74c3c', alpha=0.85)
    bars2 = ax.bar(x, solo_running, width, bottom=both_running,
                   label='Only Low Running', color='#3498db', alpha=0.85)

    # Add baseline lines
    single_1x = df[df['policy'] == 'single_1x']['high_latency_s'].values

    if len(single_1x) > 0:
        ax.axhline(y=single_1x[0], color='#2ecc71', linestyle='--', linewidth=2.5,
                   label='Single 1x')
        # Theoretical optimum: 2 × Single 1x (sequential execution of two workloads)
        theoretical_opt = single_1x[0] * 2
        ax.axhline(y=theoretical_opt, color='#9b59b6', linestyle='--', linewidth=2.5,
                   label='2×Single 1x')

    ax.set_ylabel('Completion Time (s)')
    ax.set_title(kernel_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')

    # Set y-axis limit with some padding
    max_val = max([b + s for b, s in zip(both_running, solo_running)])
    ax.set_ylim(0, max_val * 1.15)

    return ax


def _plot_combined_panel(ax, kernel_name, old_df, old_sched_df, new_df):
    """Draw one workload panel: original configs (left) + combined arms (right)."""
    ax.set_title(kernel_name)
    if old_df is None and new_df is None:
        ax.set_title(f"{kernel_name} (No Data)")
        return ax

    # Left group: the original five configurations (single-round, old timing).
    old_rows = get_selected_rows(old_df, old_sched_df) if old_df is not None else []
    old_labels = [r['label'] for r in old_rows]
    old_both = [min(r['high_latency_s'], r['low_latency_s']) for r in old_rows]
    old_solo = [abs(r['high_latency_s'] - r['low_latency_s']) for r in old_rows]

    # Right group: the new four arms (median over blocks + range of total).
    # Derived completion bounds; keep x arrays and values consistently sized so
    # a missing/partial new_df never triggers a bar shape error.
    new_pairs = []
    if new_df is not None:
        for arm, lbl in NEW_ARMS:
            s = aggregate_arm(new_df, arm)
            if s is not None:
                new_pairs.append((lbl, s))
    new_labels = [lbl for lbl, _ in new_pairs]
    new_stats = [s for _, s in new_pairs]
    new_both = [s['both'] for s in new_stats]
    new_solo = [s['solo'] for s in new_stats]

    n_old = len(old_labels)
    gap = 1.0
    old_x = list(range(n_old))
    new_x = [n_old + gap + i for i in range(len(new_labels))]
    width = 0.7

    if old_x:
        ax.bar(old_x, old_both, width, label='Both Tenants Unfinished', color='#e74c3c', alpha=0.85)
        # Neutral residual label: the slower tenant depends on the arm
        # (e.g. GEMM baseline high > low), so never name one tenant.
        ax.bar(old_x, old_solo, width, bottom=old_both, label='One Tenant Unfinished',
               color='#3498db', alpha=0.85)
    if new_x:
        ax.bar(new_x, new_both, width, color='#e74c3c', alpha=0.85)
        ax.bar(new_x, new_solo, width, bottom=new_both, color='#3498db', alpha=0.85)
        for x, s in zip(new_x, new_stats):
            if s:
                cap = 0.15
                ax.plot([x, x], [s['total_min'], s['total_max']], color='#222222',
                        linewidth=1.2, zorder=5)
                ax.plot([x - cap, x + cap], [s['total_min'], s['total_min']],
                        color='#222222', linewidth=1.2)
                ax.plot([x - cap, x + cap], [s['total_max'], s['total_max']],
                        color='#222222', linewidth=1.2)

    # The original Single 1x reference, restricted to the old (left) group.
    single_y = None
    if old_df is not None:
        single = old_df[old_df['policy'] == 'single_1x']['high_latency_s']
        if len(single) > 0:
            single_y = float(single.iloc[0])
            x0, x1 = -0.4, (n_old - 1 + 0.4) if n_old else 0.4
            ax.plot([x0, x1], [single_y, single_y], color='#2ecc71',
                    linestyle='--', linewidth=1.5, label='Single 1x')
            ax.plot([x0, x1], [2 * single_y, 2 * single_y], color='#9b59b6',
                    linestyle='--', linewidth=1.5, label='2x Single 1x')

    all_x = old_x + new_x
    all_labels = old_labels + new_labels
    if all_x:
        ax.set_xticks(all_x)
        ax.set_xticklabels(all_labels, rotation=40, ha='right')

    # Y range covering the bars, the range whiskers and the reference lines.
    max_val = 0.0
    for b, s in list(zip(old_both, old_solo)) + list(zip(new_both, new_solo)):
        max_val = max(max_val, b + s)
    for s in new_stats:
        if s:
            max_val = max(max_val, s['total_max'])
    if single_y is not None:
        max_val = max(max_val, 2 * single_y)
    if max_val <= 0:
        max_val = 1.0
    ax.set_ylim(0, max_val * 1.18)

    # Group labels + a divider between the two timing conventions.
    group_y = max_val * 1.05
    if n_old:
        ax.text((old_x[0] + old_x[-1]) / 2.0, group_y, 'Original',
                ha='center', va='bottom', fontsize=7.5, color='#555555')
    if new_x:
        ax.text((new_x[0] + new_x[-1]) / 2.0, group_y, 'Combined runner',
                ha='center', va='bottom', fontsize=7.5, color='#555555')
    if n_old and new_x:
        ax.axvline(n_old + gap / 2.0, color='#bbbbbb', linestyle=':', linewidth=1.0)

    ax.set_ylabel('Completion Time (s)')
    return ax


def run_combined_figure(args):
    """Produce the new 3-panel figure: original configs + combined four arms."""
    base_dir = Path(__file__).parent

    # Compact fonts / width for the ~7in-wide combined figure (>=7pt).
    plt.rcParams.update({
        'font.size': 8,
        'axes.labelsize': 8.5,
        'axes.titlesize': 9.5,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'figure.dpi': 150,
    })

    combined_root = find_combined_root(base_dir, args.combined_dir)
    if combined_root is None:
        print("Error: could not locate the combined results directory; pass --combined-dir")
        return
    print(f"Combined root: {combined_root}")

    kernels = [
        ('Hotspot', base_dir / 'results_hotspot'),
        ('GEMM', base_dir / 'results_gemm'),
        ('K-Means', base_dir / 'results_kmeans'),
    ]
    old_data, old_sched = {}, {}
    for kernel_name, result_dir in kernels:
        policy_csvs = list(result_dir.glob('policy_comparison_*.csv'))
        if policy_csvs:
            old_data[kernel_name] = load_data(
                max(policy_csvs, key=lambda p: p.stat().st_mtime))
        sched_csvs = list(result_dir.glob('sched_comparison_*.csv'))
        if sched_csvs:
            old_sched[kernel_name] = load_data(
                max(sched_csvs, key=lambda p: p.stat().st_mtime))

    new_data = {}
    for kernel_name, _ in kernels:
        csv_path = find_combined_csv(combined_root, COMBINED_KERNEL_DIRS[kernel_name])
        if csv_path is None:
            print(f"Warning: no combined CSV found for {kernel_name}")
            continue
        new_data[kernel_name] = load_data(csv_path)
        print(f"Loaded combined {kernel_name}: {csv_path}")

    if not new_data:
        print("Error: no combined data loaded")
        return

    width = args.width
    fig, axes = plt.subplots(1, 3, figsize=(width, width * 0.5))
    for idx, (kernel_name, _) in enumerate(kernels):
        _plot_combined_panel(axes[idx], kernel_name,
                             old_data.get(kernel_name), old_sched.get(kernel_name),
                             new_data.get(kernel_name))

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=4,
                   bbox_to_anchor=(0.5, 1.03), frameon=False)

    plt.tight_layout(rect=(0, 0, 1, 0.93))

    out = Path(args.output) if args.output else base_dir / 'all_kernels_stacked_combined'
    if not out.is_absolute():
        out = base_dir / out
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{out}.pdf', bbox_inches='tight')
    plt.savefig(f'{out}.png', bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}.pdf / {out}.png")


def main():
    base_dir = Path(__file__).parent

    # Define kernel data paths
    kernels = [
        ('Hotspot', base_dir / 'results_hotspot'),
        ('GEMM', base_dir / 'results_gemm'),
        ('K-Means', base_dir / 'results_kmeans'),
    ]

    # Load all data
    data = {}
    for kernel_name, result_dir in kernels:
        csv_files = list(result_dir.glob('policy_comparison_*.csv'))
        if not csv_files:
            print(f"Warning: No CSV found in {result_dir}")
            continue
        csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
        data[kernel_name] = load_data(csv_path)
        print(f"Loaded {kernel_name}: {csv_path}")

    # Load scheduler data
    sched_data = {}
    for kernel_name, result_dir in kernels:
        csv_files = list(result_dir.glob('sched_comparison_*.csv'))
        if csv_files:
            csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
            sched_data[kernel_name] = load_data(csv_path)
            print(f"Loaded Scheduler - {kernel_name}: {csv_path}")

    if not data:
        print("Error: No data loaded")
        return

    # Print improvement ratios
    print_improvements(data, sched_data)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    # Plot each kernel
    for idx, (kernel_name, df) in enumerate(data.items()):
        sched_df = sched_data.get(kernel_name)
        plot_kernel_subplot(axes[idx], df, kernel_name, sched_df)

    # Add shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Save
    output_path = base_dir / 'all_kernels_stacked'
    plt.savefig(f'{output_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{output_path}.png', bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}.pdf/png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Plot policy comparison figures')
    ap.add_argument('--combined', action='store_true',
                    help='Produce the new four-arm combined figure (separate output)')
    ap.add_argument('--combined-dir', default=None,
                    help='Root dir with per-kernel combined CSVs '
                         '(default: auto-detect results_combined_*)')
    ap.add_argument('--output', default=None,
                    help='Output base name for the combined figure '
                         '(default: all_kernels_stacked_combined)')
    ap.add_argument('--width', type=float, default=7.0,
                    help='Combined figure width in inches (default: 7.0)')
    cli_args = ap.parse_args()
    if cli_args.combined:
        run_combined_figure(cli_args)
    else:
        main()
