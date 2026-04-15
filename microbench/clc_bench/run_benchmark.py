#!/usr/bin/env python3
"""
CLC Policy Benchmark Driver

Runs the CLC policy benchmark and provides analysis and visualization.
"""

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import sys
import argparse
from pathlib import Path

def run_benchmark(binary="./clc_policy_benchmark", size=1048576, threads=256):
    """Run the benchmark binary and return CSV output."""
    print(f"Running benchmark: size={size}, threads={threads}")
    result = subprocess.run(
        [binary, str(size), str(threads)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error running benchmark!", file=sys.stderr)
        print(f"STDERR: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    return result.stdout

def parse_csv_output(output):
    """Parse CSV output into DataFrames (single table only)."""
    lines = output.strip().split('\n')

    # Parse main results (only one table now)
    results_data = []
    header = None

    for line in lines:
        if line.startswith("Workload,Prologue"):
            header = line.strip().split(',')
        elif header and line.strip() and not line.startswith('#'):
            values = line.strip().split(',')
            if len(values) == len(header):
                results_data.append(dict(zip(header, values)))

    df_results = pd.DataFrame(results_data)

    # Calculate speedup in the dataframe
    df_speedup = df_results.copy()
    if 'CLCBaseline_ms' in df_speedup.columns:
        for col in df_speedup.columns:
            if col.endswith('_ms') and col != 'CLCBaseline_ms':
                policy_name = col.replace('_ms', '')
                baseline = pd.to_numeric(df_speedup['CLCBaseline_ms'], errors='coerce')
                policy_time = pd.to_numeric(df_speedup[col], errors='coerce')
                df_speedup[f'{policy_name}_speedup'] = ((baseline - policy_time) / baseline * 100)

    return df_results, df_speedup

def save_csv(df_results, df_speedup, output_file):
    """Save results to CSV file (single table only)."""
    with open(output_file, 'w') as f:
        df_results.to_csv(f, index=False)
    print(f"Results saved to {output_file}")

def print_summary(df_results, df_speedup):
    """Print summary of benchmark results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    # Convert numeric columns
    for col in df_results.columns:
        if col not in ['Workload']:
            df_results[col] = pd.to_numeric(df_results[col], errors='ignore')

    # Extract policy names from columns ending with _ms (excluding baselines)
    policies = []
    for col in df_results.columns:
        if col.endswith('_ms') and col not in ['FixedWork_ms', 'FixedBlocks_ms', 'CLCBaseline_ms']:
            policies.append(col.replace('_ms', ''))

    print(f"\nWorkloads tested: {len(df_results)}")
    print("\nBest policy per workload (lowest execution time):")
    print("-" * 80)

    for _, row in df_results.iterrows():
        workload = row['Workload']
        best_policy = None
        best_time = float('inf')

        for policy in policies:
            time_col = f"{policy}_ms"
            if time_col in row:
                time = float(row[time_col])
                if time < best_time:
                    best_time = time
                    best_policy = policy

        baseline = float(row['CLCBaseline_ms'])
        speedup = ((baseline - best_time) / baseline * 100) if baseline > 0 else 0
        print(f"{workload:40s} -> {best_policy:15s} ({best_time:.3f}ms, {speedup:+.1f}%)")

    if df_speedup is not None:
        print("\n\nAverage speedup vs CLC Baseline:")
        print("-" * 80)

        for col in df_speedup.columns:
            if col != 'Workload':
                df_speedup[col] = pd.to_numeric(df_speedup[col], errors='ignore')

        for policy in policies:
            speedup_col = f"{policy}_speedup"
            if speedup_col in df_speedup.columns:
                avg = df_speedup[speedup_col].mean()
                print(f"{policy:20s}: {avg:+7.2f}%")

    print("="*80)

def generate_plots(df_results, df_speedup, output_prefix='benchmark'):
    """Generate a single combined visualization plot with all policies."""
    import numpy as np

    # Convert numeric columns
    for col in df_results.columns:
        if col not in ['Workload']:
            try:
                df_results[col] = pd.to_numeric(df_results[col])
            except:
                pass

    workloads = df_results['Workload'].tolist()

    # Create a grid of subplots (3 rows x 3 columns)
    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.suptitle('CLC Policy Benchmark - Execution Time Comparison (All Policies)', fontsize=18, fontweight='bold')

    # Flatten axes for easier iteration
    axes = axes.flatten()

    # Extract all policies from columns ending with _ms
    all_policies = []
    for col in df_results.columns:
        if col.endswith('_ms'):
            all_policies.append(col.replace('_ms', ''))

    # Colors for the bars (use tab20 colormap)
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_policies)))

    # Plot each workload in a separate subplot
    for idx, workload in enumerate(workloads):
        if idx >= 9:  # Only plot first 9 workloads
            break

        ax = axes[idx]
        row = df_results[df_results['Workload'] == workload].iloc[0]

        # Get times for all policies
        times = []
        policy_labels = []
        for p in all_policies:
            col = f'{p}_ms'
            if col in row:
                times.append(float(row[col]))
                policy_labels.append(p)

        # Create bar chart
        x = np.arange(len(policy_labels))
        width = 0.7

        bars = ax.bar(x, times, width, color=colors[:len(policy_labels)], alpha=0.85, edgecolor='black', linewidth=0.5)

        # Formatting
        ax.set_ylabel('Execution Time (ms)', fontsize=10)
        ax.set_title(workload[:35], fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        # Generate short labels dynamically (first 6 chars or abbreviations)
        short_labels = []
        for label in policy_labels:
            if len(label) <= 6:
                short_labels.append(label)
            else:
                # Create abbreviation from capital letters or first 6 chars
                abbrev = ''.join([c for c in label if c.isupper()])
                short_labels.append(abbrev if abbrev else label[:6])
        ax.set_xticklabels(short_labels, fontsize=8, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

        # Add value labels on top of bars for the best (lowest) time
        min_time = min(times)
        min_idx = times.index(min_time)
        ax.text(min_idx, times[min_idx], f'{times[min_idx]:.3f}',
                ha='center', va='bottom', fontsize=7, fontweight='bold', color='red')

    # Remove any unused subplots
    for idx in range(len(workloads), 9):
        fig.delaxes(axes[idx])

    # Add a legend at the bottom with actual policy names
    legend_handles = [plt.Rectangle((0,0),1,1, fc=colors[i], alpha=0.85, edgecolor='black', linewidth=0.5)
                      for i in range(len(all_policies))]
    fig.legend(legend_handles, all_policies, loc='lower center', ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    output_file = f"{output_prefix}_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run CLC Policy Benchmark")
    parser.add_argument('--size', type=int, default=1048576, help='Problem size (default: 1048576)')
    parser.add_argument('--threads', type=int, default=256, help='Threads per block (default: 256)')
    parser.add_argument('--output', type=str, default='benchmark_results.csv', help='Output CSV file')
    parser.add_argument('--binary', type=str, default='./clc_policy_benchmark', help='Benchmark binary path')
    parser.add_argument('--no-summary', action='store_true', help='Skip summary output')
    parser.add_argument('--no-plot', action='store_true', help='Skip generating visualization plots')
    parser.add_argument('--plot-prefix', type=str, default='benchmark', help='Prefix for plot filenames')

    args = parser.parse_args()

    if not Path(args.binary).exists():
        print(f"ERROR: Binary not found: {args.binary}", file=sys.stderr)
        print("Please run 'make clc_policy_benchmark' first", file=sys.stderr)
        sys.exit(1)

    # Run benchmark
    output = run_benchmark(args.binary, args.size, args.threads)

    # Parse results
    df_results, df_speedup = parse_csv_output(output)

    # Save to file
    save_csv(df_results, df_speedup, args.output)

    # Print summary
    if not args.no_summary:
        print_summary(df_results, df_speedup)

    # Generate plots by default
    if not args.no_plot:
        print("\nGenerating plots...")
        generate_plots(df_results, df_speedup, args.plot_prefix)

    print("\nDone!")

if __name__ == "__main__":
    main()
