#!/usr/bin/env python3
"""Reproducible expanded Fig.13: four panels, 7in wide, 2x2 at ~3.5in tall.

Panels (a)-(c) reuse the audited historical single-round policy comparison
(n=1 per config) exactly as in
docs/paper/tex-revision/img/results-raw/multi-tenant/plot_all_kernels_stacked.py:
the same six source CSVs and selected configs, bottom segment =
min(high, low) of the recorded durations, top segment = absolute difference
of those recorded durations, total height = completion-time proxy. The
scheduler launcher records the low duration after waiting for high, so these
segments cannot recover true overlap or independent tenant finish times;
scheduler engagement was not verified. No variance, CI, significance or
scheduler-ineffectiveness claim. Twice the single-process duration is a
sequential reference, not a lower bound.

Panel (d) plots the fresh independently timed HotSpot run from
workloads/fig13-fast/results/fig13_fast_20260907_005958/fig13_fast.csv:
per-tenant wall duration (s), each tenant timed independently from its own
SIGCONT to its own exit, median bar with full min-max error bars across the
interleaved blocks (not a CI). All recorded rows are used as recorded;
no filtering.

Usage:
    python3 workloads/fig13-fast/plot_expanded.py
        [--output-dir workloads/fig13-fast/figures] [--repo-root REPO_ROOT]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'hatch.linewidth': 0.6,
    'pdf.fonttype': 42,
    'figure.dpi': 300,
})

BLUE = '#1f77b4'
ORANGE = '#ff7f0e'
HATCH = '//'
BAR_EDGE = {'edgecolor': 'black', 'linewidth': 0.5}

# Selected configurations (policy, high_param, low_param, label, is_sched),
# identical to the historical script.
SELECTED_CONFIGS = [
    ('no_policy', None, None, 'No Policy', False),
    ('sched_timeslice', 1000000, 200, 'Scheduler*', True),
    ('prefetch_pid_tree', 0, 20, 'Prefetch(0,20)', False),
    ('prefetch_pid_tree', 20, 80, 'Prefetch(20,80)', False),
    ('prefetch_eviction_pid', 20, 80, 'Evict(20,80)', False),
]

# Complete source files selected by docs/eval/rq3-revision-audit.md. Never choose
# a run by modification time or silently substitute a different/incomplete CSV.
HISTORICAL_SOURCES = [
    ('HotSpot', 'results_hotspot/policy_comparison_20251208_101609.csv',
     'results_hotspot/sched_comparison_20251208_113441.csv'),
    ('GEMM', 'results_gemm/policy_comparison_20251208_102321.csv',
     'results_gemm/sched_comparison_20251208_113846.csv'),
    ('K-Means', 'results_kmeans/policy_comparison_20251208_103714.csv',
     'results_kmeans/sched_comparison_20251208_114516.csv'),
]

FRESH_SOURCE = ('workloads/fig13-fast/results/'
                'fig13_fast_20260907_005958/fig13_fast.csv')
FRESH_ARMS = (('baseline', 'Baseline'), ('memory_only', 'Memory only'),
              ('sched_only', 'Sched. only'), ('combined', 'Combined'))


def load_data(csv_path):
    """Load one fixed historical CSV, rejecting changed repetition semantics."""
    df = pd.read_csv(csv_path)
    if df.empty or not df['round'].eq(1).all():
        raise ValueError(f'Expected historical round=1 rows: {csv_path}')
    return df


def get_selected_rows(df, sched_df=None):
    """Filter dataframe to only selected configurations."""
    rows = []
    for policy, hp, lp, label, is_sched in SELECTED_CONFIGS:
        source_df = sched_df if is_sched else df
        if source_df is None:
            raise ValueError(f'Missing source for {label}')
        if policy == 'no_policy':
            row = source_df[source_df['policy'] == 'no_policy']
        else:
            row = source_df[(source_df['policy'] == policy) &
                      (source_df['high_param'] == hp) &
                      (source_df['low_param'] == lp)]
        if len(row) != 1:
            raise ValueError(f'Expected exactly one historical row for {label}')
        r = row.iloc[0].to_dict()
        if not all(np.isfinite(r[key]) and r[key] > 0
                   for key in ('high_latency_s', 'low_latency_s')):
            raise ValueError(f'Invalid recorded duration for {label}')
        r['label'] = label
        rows.append(r)
    return rows


def load_fresh(csv_path):
    """Load the fresh per-block four-arm CSV, all rows used as recorded."""
    df = pd.read_csv(csv_path)
    for col in ('arm', 'high_latency_s', 'low_latency_s'):
        if col not in df.columns:
            raise ValueError(f'Missing column {col} in {csv_path}')
    for arm, _ in FRESH_ARMS:
        if df['arm'].eq(arm).sum() < 1:
            raise ValueError(f'Missing fresh arm {arm!r} in {csv_path}')
    if not (np.isfinite(df['high_latency_s']).all()
            and np.isfinite(df['low_latency_s']).all()):
        raise ValueError(f'Non-finite recorded duration in {csv_path}')
    return df


def plot_historical_panel(ax, df, panel_title, sched_df=None, show_legend=False):
    """One historical kernel: stacked min + absolute difference of recorded durations."""
    rows = get_selected_rows(df, sched_df)

    if not rows:
        ax.set_title(f'{panel_title} (No Data)')
        return ax

    labels = [r['label'] for r in rows]

    min_durations = []
    duration_differences = []
    for r in rows:
        high = r['high_latency_s']
        low = r['low_latency_s']
        min_durations.append(min(high, low))
        duration_differences.append(abs(high - low))

    x = np.arange(len(labels))
    width = 0.6

    ax.bar(x, min_durations, width, color=BLUE,
           label='Min. recorded' if show_legend else None, **BAR_EDGE)
    ax.bar(x, duration_differences, width, bottom=min_durations, color=ORANGE,
           hatch=HATCH, label='Abs. diff.' if show_legend else None, **BAR_EDGE)

    single_1x = df[df['policy'] == 'single_1x']['high_latency_s'].values
    if len(single_1x) > 0:
        s1 = float(single_1x[0])
        ax.axhline(s1, color='black', linestyle=(0, (4, 2)), linewidth=0.8,
                   label='1x single' if show_legend else None)
        ax.axhline(2 * s1, color='black', linestyle=(0, (1.5, 1.5)), linewidth=0.8,
                   label='2x seq. ref.' if show_legend else None)

    ax.set_title(panel_title)
    ax.set_ylabel('Completion-time proxy (s)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')

    max_val = max(b + s for b, s in zip(min_durations, duration_differences))
    ax.set_ylim(0, max_val * 1.3)
    if show_legend:
        ax.legend(loc='upper right', ncol=2, framealpha=0.9,
                  handlelength=1.5, handletextpad=0.4, borderpad=0.3)
    return ax


def plot_fresh_panel(ax, df, panel_title):
    """Fresh HotSpot: per-tenant median wall duration, full min-max error bars."""
    x = np.arange(len(FRESH_ARMS))
    w = 0.36
    med_hi, med_lo = [], []
    err_hi, err_lo = [], []
    for arm, _ in FRESH_ARMS:
        sub = df[df['arm'] == arm]
        hi = sub['high_latency_s'].to_numpy(dtype=float)
        lo = sub['low_latency_s'].to_numpy(dtype=float)
        mh, ml = float(np.median(hi)), float(np.median(lo))
        med_hi.append(mh)
        med_lo.append(ml)
        err_hi.append([mh - float(hi.min()), float(hi.max()) - mh])
        err_lo.append([ml - float(lo.min()), float(lo.max()) - ml])

    error_kw = {'elinewidth': 0.7, 'capsize': 1.5, 'capthick': 0.7,
                'color': 'black'}
    ax.bar(x - w / 2, med_hi, w, color=BLUE, yerr=np.asarray(err_hi).T,
           label='High tenant', **BAR_EDGE, error_kw=error_kw)
    ax.bar(x + w / 2, med_lo, w, color=ORANGE, hatch=HATCH,
           yerr=np.asarray(err_lo).T, label='Low tenant', **BAR_EDGE,
           error_kw=error_kw)

    ax.set_title(panel_title)
    ax.set_ylabel('Process duration (s)')
    ax.set_xticks(x)
    ax.set_xticklabels([disp for _, disp in FRESH_ARMS])

    top = max(max(np.asarray(med_hi) + np.asarray(err_hi)[:, 1]),
              max(np.asarray(med_lo) + np.asarray(err_lo)[:, 1]))
    ax.set_ylim(0, top * 1.15)
    ax.legend(loc='upper right', framealpha=0.9,
              handlelength=1.5, handletextpad=0.4, borderpad=0.3)
    return ax


def parse_args():
    script_dir = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description='Plot the expanded 4-panel Fig.13.')
    ap.add_argument('--output-dir', type=Path, default=script_dir / 'figures')
    ap.add_argument('--repo-root', type=Path, default=script_dir.parents[1])
    return ap.parse_args()


def main():
    args = parse_args()
    repo_root = args.repo_root
    data_dir = repo_root / 'docs' / 'eval' / 'multi-tenant-memory'

    data = {}
    sched_data = {}
    for kernel_name, memory_file, scheduler_file in HISTORICAL_SOURCES:
        data[kernel_name] = load_data(data_dir / memory_file)
        sched_data[kernel_name] = load_data(data_dir / scheduler_file)

    fresh_df = load_fresh(repo_root / FRESH_SOURCE)

    fig, axes = plt.subplots(2, 2, figsize=(7, 3.5))
    plot_historical_panel(axes[0][0], data['HotSpot'], '(a) Historical HotSpot',
                          sched_data['HotSpot'], show_legend=True)
    plot_historical_panel(axes[0][1], data['GEMM'], '(b) Historical GEMM',
                          sched_data['GEMM'])
    plot_historical_panel(axes[1][0], data['K-Means'], '(c) Historical K-Means',
                          sched_data['K-Means'])
    plot_fresh_panel(axes[1][1], fresh_df, '(d) Fresh HotSpot')

    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / 'fig13_expanded'
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"Historical n=1 sources: {len(HISTORICAL_SOURCES)} kernels, "
          f"memory+scheduler CSVs under {data_dir}")
    print(f"Fresh source: {repo_root / FRESH_SOURCE} "
          f"({len(fresh_df)} rows, {fresh_df['block'].nunique()} blocks)")
    print(f"Saved: {out.with_suffix('.pdf')}, {out.with_suffix('.png')}")


if __name__ == '__main__':
    main()
