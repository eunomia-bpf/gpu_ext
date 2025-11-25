#!/usr/bin/env python3
"""
Visualize prefetch patterns from prefetch_trace data.

This script analyzes the prefetch_trace CSV data to understand:
1. Virtual address space access patterns over time
2. Prefetch computation frequency and distribution
3. Faulted region sizes and patterns
4. Working set evolution (pages_accessed over time)
5. VA block hotspots and access density
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

def load_and_prepare_data(filename):
    """Load CSV and prepare data structures."""
    print(f"[1/7] Loading data from {filename}...")

    df = pd.read_csv(filename, on_bad_lines='skip', low_memory=False)
    print(f"  → Read {len(df):,} rows")

    # Convert time_ms
    df['time_ms'] = pd.to_numeric(df['time_ms'], errors='coerce')
    df = df[df['time_ms'].notna()].copy()

    # Parse hex addresses
    def parse_hex(x):
        try:
            if pd.isna(x) or x == '' or x == '0x0':
                return 0
            return int(x, 16) if isinstance(x, str) else int(x)
        except:
            return 0

    print(f"  → Parsing addresses...")
    df['va_start_int'] = df['va_start'].apply(parse_hex)
    df['va_end_int'] = df['va_end'].apply(parse_hex)

    # Calculate derived fields
    df['faulted_size'] = df['faulted_outer'] - df['faulted_first']
    df['density'] = df['pages_accessed'] / 512 * 100  # percentage

    print(f"  → Loaded {len(df):,} valid events")
    print(f"  → Time range: {df['time_ms'].min():.0f} - {df['time_ms'].max():.0f} ms")
    print(f"  → Unique VA blocks: {df['va_start_int'].nunique():,}")

    return df


def plot_va_access_timeline(df, output_file):
    """Plot VA address space access pattern over time."""
    print(f"\n[2/7] Generating VA access timeline...")

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Sample for visualization if too large
    sample_size = min(100000, len(df))
    sample = df.sample(sample_size).sort_values('time_ms')

    # Plot 1: VA address vs time scatter
    ax1 = axes[0]
    scatter = ax1.scatter(sample['time_ms'], sample['va_start_int'],
                         c=sample['pages_accessed'], cmap='YlOrRd',
                         alpha=0.3, s=1, rasterized=True)

    ax1.set_ylabel('Virtual Address', fontsize=12)
    ax1.set_title('Virtual Address Space Access Pattern Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'0x{int(x):x}' if x > 0 else '0'))

    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Pages Accessed', fontsize=10)

    # Plot 2: Prefetch rate over time
    ax2 = axes[1]
    window_ms = 100
    bins = np.arange(0, df['time_ms'].max() + window_ms, window_ms)
    hist, edges = np.histogram(df['time_ms'], bins=bins)
    rate = hist / (window_ms / 1000)  # events per second

    ax2.fill_between(edges[:-1], rate, alpha=0.5, color='steelblue')
    ax2.plot(edges[:-1], rate, linewidth=1, color='steelblue')
    ax2.set_xlabel('Time (ms)', fontsize=12)
    ax2.set_ylabel('Prefetch Rate (events/sec)', fontsize=12)
    ax2.set_title(f'Prefetch Computation Rate Over Time (window: {window_ms}ms)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_va_heatmap(df, output_file):
    """Plot VA address access density heatmap."""
    print(f"\n[3/7] Generating VA access heatmap...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: 2D histogram - time vs VA address
    ax1 = axes[0]
    H, xedges, yedges = np.histogram2d(
        df['time_ms'],
        df['va_start_int'],
        bins=[100, 100]
    )

    im = ax1.imshow(H.T, aspect='auto', origin='lower', cmap='hot',
                    extent=[0, df['time_ms'].max(),
                           df['va_start_int'].min(),
                           df['va_start_int'].max()],
                    interpolation='bilinear')

    ax1.set_xlabel('Time (ms)', fontsize=11)
    ax1.set_ylabel('Virtual Address', fontsize=11)
    ax1.set_title('VA Access Density Heatmap', fontsize=12, fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'0x{int(x):x}' if x > 0 else '0'))

    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Access Count', fontsize=9)

    # Plot 2: VA block access frequency distribution
    ax2 = axes[1]
    va_access_freq = df.groupby('va_start_int').size()

    ax2.hist(va_access_freq, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.set_xlabel('Prefetch Events per VA Block', fontsize=11)
    ax2.set_ylabel('Number of VA Blocks', fontsize=11)
    ax2.set_title('VA Block Access Frequency Distribution', fontsize=12, fontweight='bold')
    ax2.axvline(va_access_freq.median(), color='red', linestyle='--', linewidth=2,
                label=f'Median: {va_access_freq.median():.0f}')
    ax2.axvline(va_access_freq.mean(), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {va_access_freq.mean():.1f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_faulted_region_analysis(df, output_file):
    """Analyze faulted region patterns."""
    print(f"\n[4/7] Generating faulted region analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Faulted size distribution
    ax1 = axes[0, 0]
    ax1.hist(df['faulted_size'], bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax1.set_xlabel('Faulted Region Size (pages)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Faulted Region Size Distribution', fontsize=12, fontweight='bold')
    ax1.axvline(df['faulted_size'].median(), color='red', linestyle='--', linewidth=2,
                label=f'Median: {df["faulted_size"].median():.0f}')
    ax1.axvline(df['faulted_size'].mean(), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {df["faulted_size"].mean():.1f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Faulted_first (start position) distribution
    ax2 = axes[0, 1]
    ax2.hist(df['faulted_first'], bins=64, edgecolor='black', alpha=0.7, color='teal')
    ax2.set_xlabel('Faulted Region Start (page index)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Fault Start Position in VA Block', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: page_index distribution (where fault triggered)
    ax3 = axes[1, 0]
    ax3.hist(df['page_index'], bins=64, edgecolor='black', alpha=0.7, color='purple')
    ax3.set_xlabel('Page Index (triggering page)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Triggering Page Index Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Faulted size over time
    ax4 = axes[1, 1]
    window_ms = 100
    df['time_bucket'] = (df['time_ms'] // window_ms) * window_ms
    time_avg = df.groupby('time_bucket')['faulted_size'].mean()

    ax4.plot(time_avg.index, time_avg.values, linewidth=1.5, color='coral', alpha=0.8)
    ax4.fill_between(time_avg.index, time_avg.values, alpha=0.3, color='coral')
    ax4.set_xlabel('Time (ms)', fontsize=11)
    ax4.set_ylabel('Average Faulted Size (pages)', fontsize=11)
    ax4.set_title('Average Faulted Region Size Over Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_working_set_evolution(df, output_file):
    """Plot working set evolution (pages_accessed over time)."""
    print(f"\n[5/7] Generating working set evolution...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: pages_accessed distribution
    ax1 = axes[0, 0]
    ax1.hist(df['pages_accessed'], bins=50, edgecolor='black', alpha=0.7, color='green')
    ax1.set_xlabel('Pages Accessed (per VA block)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Pages Accessed Distribution', fontsize=12, fontweight='bold')
    ax1.axvline(df['pages_accessed'].median(), color='red', linestyle='--', linewidth=2,
                label=f'Median: {df["pages_accessed"].median():.0f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Access density distribution
    ax2 = axes[0, 1]
    ax2.hist(df['density'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax2.set_xlabel('Access Density (%)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('VA Block Access Density Distribution', fontsize=12, fontweight='bold')
    ax2.axvline(df['density'].median(), color='red', linestyle='--', linewidth=2,
                label=f'Median: {df["density"].median():.1f}%')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: pages_accessed over time
    ax3 = axes[1, 0]
    window_ms = 100
    df['time_bucket'] = (df['time_ms'] // window_ms) * window_ms
    time_stats = df.groupby('time_bucket')['pages_accessed'].agg(['mean', 'std', 'median'])

    ax3.plot(time_stats.index, time_stats['mean'], linewidth=1.5, color='green', label='Mean')
    ax3.plot(time_stats.index, time_stats['median'], linewidth=1.5, color='blue', alpha=0.7, label='Median')
    ax3.fill_between(time_stats.index,
                     time_stats['mean'] - time_stats['std'],
                     time_stats['mean'] + time_stats['std'],
                     alpha=0.2, color='green', label='±1 Std')
    ax3.set_xlabel('Time (ms)', fontsize=11)
    ax3.set_ylabel('Pages Accessed', fontsize=11)
    ax3.set_title('Working Set Size Over Time', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Unique VA blocks over time (cumulative)
    ax4 = axes[1, 1]
    df_sorted = df.sort_values('time_ms')
    cumulative_va = df_sorted.groupby('time_bucket')['va_start_int'].apply(lambda x: x.nunique()).cumsum()

    ax4.plot(cumulative_va.index, cumulative_va.values, linewidth=2, color='purple')
    ax4.set_xlabel('Time (ms)', fontsize=11)
    ax4.set_ylabel('Cumulative Unique VA Blocks', fontsize=11)
    ax4.set_title('VA Block Discovery Over Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_access_pattern_analysis(df, output_file):
    """Analyze access patterns within VA blocks."""
    print(f"\n[6/7] Generating access pattern analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Sequential vs Random access detection
    ax1 = axes[0, 0]

    # Calculate page_index differences for same VA block
    df_sorted = df.sort_values(['va_start_int', 'time_ms'])
    df_sorted['prev_va'] = df_sorted['va_start_int'].shift(1)
    df_sorted['prev_page'] = df_sorted['page_index'].shift(1)

    same_va = df_sorted[df_sorted['va_start_int'] == df_sorted['prev_va']].copy()
    if len(same_va) > 0:
        same_va['page_diff'] = (same_va['page_index'] - same_va['prev_page']).abs()

        ax1.hist(same_va['page_diff'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.set_xlabel('Page Index Difference (within same VA block)', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Access Stride Distribution', fontsize=12, fontweight='bold')
        ax1.axvline(16, color='red', linestyle='--', linewidth=2, label='Sequential threshold (16)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Calculate sequential ratio
        seq_ratio = (same_va['page_diff'] <= 16).sum() / len(same_va) * 100
        ax1.text(0.95, 0.95, f'Sequential: {seq_ratio:.1f}%\nRandom: {100-seq_ratio:.1f}%',
                transform=ax1.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Top VA blocks access timeline
    ax2 = axes[0, 1]

    va_counts = df.groupby('va_start_int').size()
    top_vas = va_counts.nlargest(5).index

    for i, va in enumerate(top_vas):
        va_events = df[df['va_start_int'] == va].sort_values('time_ms')
        ax2.scatter(va_events['time_ms'], [i] * len(va_events),
                   c=va_events['page_index'], cmap='viridis', s=3, alpha=0.5)

    ax2.set_xlabel('Time (ms)', fontsize=11)
    ax2.set_ylabel('VA Block Rank', fontsize=11)
    ax2.set_title('Top 5 VA Blocks - Access Timeline', fontsize=12, fontweight='bold')
    ax2.set_yticks(range(5))
    ax2.set_yticklabels([f'VA #{i+1}' for i in range(5)])
    ax2.grid(True, alpha=0.3)

    # Plot 3: Global page_index heatmap over time (all VA blocks)
    ax3 = axes[1, 0]

    # Create 2D histogram: time vs page_index (across all VA blocks)
    time_bins = 100
    page_bins = 64  # 512/8 = 64 bins for page_index

    H, xedges, yedges = np.histogram2d(
        df['time_ms'],
        df['page_index'],
        bins=[time_bins, page_bins]
    )

    im = ax3.imshow(H.T, aspect='auto', origin='lower', cmap='YlOrRd',
                   extent=[df['time_ms'].min(), df['time_ms'].max(), 0, 512],
                   interpolation='bilinear')

    ax3.set_xlabel('Time (ms)', fontsize=11)
    ax3.set_ylabel('Page Index (within VA block)', fontsize=11)
    ax3.set_title('Page Index Access Heatmap Over Time (All VA Blocks)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Access Count')

    # Plot 4: Inter-event time distribution
    ax4 = axes[1, 1]

    df_sorted = df.sort_values('time_ms')
    inter_times = df_sorted['time_ms'].diff().dropna()
    inter_times = inter_times[inter_times > 0]

    if len(inter_times) > 0:
        ax4.hist(inter_times, bins=100, edgecolor='black', alpha=0.7, color='purple')
        ax4.set_xlabel('Inter-event Time (ms)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('Inter-Prefetch Time Distribution', fontsize=12, fontweight='bold')
        ax4.set_yscale('log')
        ax4.axvline(inter_times.median(), color='red', linestyle='--', linewidth=2,
                    label=f'Median: {inter_times.median():.3f}ms')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def plot_summary_dashboard(df, output_file):
    """Create a summary dashboard with key metrics."""
    print(f"\n[7/7] Generating summary dashboard...")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Calculate key metrics
    total_events = len(df)
    duration_sec = df['time_ms'].max() / 1000
    event_rate = total_events / duration_sec if duration_sec > 0 else 0
    unique_vas = df['va_start_int'].nunique()
    avg_density = df['density'].mean()
    avg_faulted_size = df['faulted_size'].mean()

    # Plot 1: Key metrics text box
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')

    metrics_text = f"""
    PREFETCH TRACE SUMMARY
    ══════════════════════════════════

    Duration:         {duration_sec:.2f} seconds
    Total Events:     {total_events:,}
    Event Rate:       {event_rate:,.0f} /sec

    Unique VA Blocks: {unique_vas:,}
    Events per VA:    {total_events/unique_vas:.1f}

    Avg Density:      {avg_density:.1f}%
    Avg Fault Size:   {avg_faulted_size:.1f} pages

    VA Address Span:  {(df['va_end_int'].max() - df['va_start_int'].min()) / (1024**3):.2f} GB
    """
    ax1.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Plot 2: Event rate over time
    ax2 = fig.add_subplot(gs[0, 1:])
    window_ms = 100
    bins = np.arange(0, df['time_ms'].max() + window_ms, window_ms)
    hist, edges = np.histogram(df['time_ms'], bins=bins)
    rate = hist / (window_ms / 1000)

    ax2.fill_between(edges[:-1], rate, alpha=0.5, color='steelblue')
    ax2.plot(edges[:-1], rate, linewidth=1, color='steelblue')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Events/sec')
    ax2.set_title('Prefetch Rate Over Time')
    ax2.grid(True, alpha=0.3)

    # Plot 3: VA address scatter
    ax3 = fig.add_subplot(gs[1, :2])
    sample_size = min(50000, len(df))
    sample = df.sample(sample_size).sort_values('time_ms')

    scatter = ax3.scatter(sample['time_ms'], sample['va_start_int'],
                         c=sample['density'], cmap='YlOrRd',
                         alpha=0.3, s=1, rasterized=True)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('VA Address')
    ax3.set_title('VA Access Pattern (color = density %)')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'0x{int(x):x}' if x > 0 else '0'))
    plt.colorbar(scatter, ax=ax3, label='Density %')

    # Plot 4: Faulted size distribution
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(df['faulted_size'], bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax4.set_xlabel('Faulted Size (pages)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Fault Batch Size')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Density distribution
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist(df['density'], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax5.set_xlabel('Density (%)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Access Density Distribution')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Page index distribution
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(df['page_index'], bins=32, edgecolor='black', alpha=0.7, color='purple')
    ax6.set_xlabel('Page Index')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Fault Position in VA Block')
    ax6.grid(True, alpha=0.3)

    # Plot 7: VA block event count distribution
    ax7 = fig.add_subplot(gs[2, 2])
    va_counts = df.groupby('va_start_int').size()
    ax7.hist(va_counts, bins=30, edgecolor='black', alpha=0.7, color='teal')
    ax7.set_xlabel('Events per VA Block')
    ax7.set_ylabel('Number of VA Blocks')
    ax7.set_title('VA Block Activity Distribution')
    ax7.grid(True, alpha=0.3)

    plt.suptitle('Prefetch Trace Analysis Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize prefetch patterns from trace data')
    parser.add_argument('input', nargs='?', default='/tmp/prefetch_combined.csv',
                        help='Input CSV file (default: /tmp/prefetch_combined.csv)')
    parser.add_argument('-o', '--output-dir', default='/tmp',
                        help='Output directory for plots (default: /tmp)')
    args = parser.parse_args()

    try:
        # Load data
        df = load_and_prepare_data(args.input)

        # Generate visualizations
        output_dir = args.output_dir

        plot_summary_dashboard(df, os.path.join(output_dir, 'prefetch_summary.png'))
        plot_va_access_timeline(df, os.path.join(output_dir, 'prefetch_va_timeline.png'))
        plot_va_heatmap(df, os.path.join(output_dir, 'prefetch_va_heatmap.png'))
        plot_faulted_region_analysis(df, os.path.join(output_dir, 'prefetch_faulted_region.png'))
        plot_working_set_evolution(df, os.path.join(output_dir, 'prefetch_working_set.png'))
        plot_access_pattern_analysis(df, os.path.join(output_dir, 'prefetch_access_pattern.png'))

        print("\n" + "="*80)
        print("Visualization complete!")
        print(f"All plots saved to: {output_dir}")
        print("  - prefetch_summary.png       : Overall dashboard")
        print("  - prefetch_va_timeline.png   : VA access over time")
        print("  - prefetch_va_heatmap.png    : VA access density heatmap")
        print("  - prefetch_faulted_region.png: Faulted region analysis")
        print("  - prefetch_working_set.png   : Working set evolution")
        print("  - prefetch_access_pattern.png: Access pattern analysis")
        print("="*80)

    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
