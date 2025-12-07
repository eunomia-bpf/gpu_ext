#!/usr/bin/env python3
"""
RQ1: Performance Comparison Across Memory Modes
Compares device-only, UVM, and UVM+prefetch modes for Tier-0 synthetic kernels
"""

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Configuration
KERNELS = ['seq_stream', 'rand_stream', 'pointer_chase']
MODES = ['device', 'uvm', 'uvm_prefetch']
SIZE_FACTOR = 0.25  # Use 0.25x GPU memory for fair comparison (Tier-0 only)
STRIDE_BYTES = 4    # Element-level for fair throughput comparison
ITERATIONS = 5      # Reduced for faster execution
EXECUTABLE = './uvmbench'

def run_benchmark(kernel, mode, size_factor, output_file):
    """Run a single benchmark configuration"""
    cmd = [
        EXECUTABLE,
        f'--kernel={kernel}',
        f'--mode={mode}',
        f'--size_factor={size_factor}',
        f'--stride_bytes={STRIDE_BYTES}',
        f'--iterations={ITERATIONS}',
        f'--output={output_file}'
    ]

    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False

def collect_results():
    """Run all benchmark configurations and collect results"""
    results = []

    for kernel in KERNELS:
        for mode in MODES:
            output_file = f'results_rq1_{kernel}_{mode}.csv'

            if run_benchmark(kernel, mode, SIZE_FACTOR, output_file):
                # Read the result
                try:
                    df = pd.read_csv(output_file)
                    results.append(df)
                    # Clean up individual CSV
                    os.remove(output_file)
                except Exception as e:
                    print(f"Error reading {output_file}: {e}")

    if results:
        combined = pd.concat(results, ignore_index=True)
        combined.to_csv('rq1_results.csv', index=False)
        return combined
    else:
        print("No results collected!")
        return None

def plot_results(df):
    """Generate visualization for RQ1"""
    if df is None or df.empty:
        print("No data to plot!")
        return

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 5)

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # Plot 1: Absolute runtime comparison
    pivot_data = df.pivot(index='kernel', columns='mode', values='median_ms')
    pivot_data.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_xlabel('Kernel Type', fontsize=12)
    ax1.set_ylabel('Median Runtime (ms)', fontsize=12)
    ax1.set_title('RQ1: Absolute Runtime by Mode', fontsize=14, fontweight='bold')
    ax1.legend(title='Memory Mode', loc='upper left')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Slowdown relative to device-only
    slowdown_data = []
    for kernel in KERNELS:
        kernel_df = df[df['kernel'] == kernel]
        device_time = kernel_df[kernel_df['mode'] == 'device']['median_ms'].values

        if len(device_time) > 0:
            baseline = device_time[0]
            for _, row in kernel_df.iterrows():
                slowdown = row['median_ms'] / baseline
                slowdown_data.append({
                    'kernel': kernel,
                    'mode': row['mode'],
                    'slowdown': slowdown
                })

    slowdown_df = pd.DataFrame(slowdown_data)
    slowdown_pivot = slowdown_df.pivot(index='kernel', columns='mode', values='slowdown')

    # Remove device column (always 1.0x)
    if 'device' in slowdown_pivot.columns:
        slowdown_pivot = slowdown_pivot.drop(columns=['device'])

    slowdown_pivot.plot(kind='bar', ax=ax2, width=0.8, color=['#ff7f0e', '#2ca02c'])
    ax2.set_xlabel('Kernel Type', fontsize=12)
    ax2.set_ylabel('Slowdown vs Device-Only', fontsize=12)
    ax2.set_title('RQ1: UVM Overhead (Tier-0 Synthetic)', fontsize=14, fontweight='bold')
    ax2.legend(title='Memory Mode', loc='upper left')
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Device baseline')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Bandwidth comparison
    bw_pivot = df.pivot(index='kernel', columns='mode', values='bw_GBps')
    bw_pivot.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_xlabel('Kernel Type', fontsize=12)
    ax3.set_ylabel('Effective Bandwidth (GB/s)', fontsize=12)
    ax3.set_title('RQ1: Memory Bandwidth by Mode', fontsize=14, fontweight='bold')
    ax3.legend(title='Memory Mode', loc='upper left')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('rq1_mode_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('rq1_mode_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved: rq1_mode_comparison.{pdf,png}")

    # Print summary statistics
    print("\n" + "="*60)
    print("RQ1 SUMMARY: UVM Overhead on Tier-0 Synthetic Kernels")
    print("="*60)
    print(f"\nConfiguration: size_factor={SIZE_FACTOR}, stride_bytes={STRIDE_BYTES}, iterations={ITERATIONS}")
    print("\nSlowdown vs Device-Only Baseline:")
    print(slowdown_pivot.to_string())
    print("\nKey Findings:")
    for kernel in KERNELS:
        kernel_slowdown = slowdown_df[slowdown_df['kernel'] == kernel]
        uvm_slowdown = kernel_slowdown[kernel_slowdown['mode'] == 'uvm']['slowdown'].values
        if len(uvm_slowdown) > 0:
            print(f"  - {kernel}: {uvm_slowdown[0]:.2f}x slowdown with UVM")

def main():
    if not os.path.exists(EXECUTABLE):
        print(f"Error: {EXECUTABLE} not found!")
        print("Please run 'make' in the micro directory first.")
        sys.exit(1)

    print("="*60)
    print("RQ1: Performance Comparison Across Memory Modes")
    print("="*60)
    print(f"Kernels: {', '.join(KERNELS)}")
    print(f"Modes: {', '.join(MODES)}")
    print(f"Size Factor: {SIZE_FACTOR}x GPU memory")
    print(f"Iterations: {ITERATIONS}")
    print("="*60)
    print()

    # Collect results
    df = collect_results()

    if df is not None:
        # Generate plots
        plot_results(df)
        print("\nRQ1 evaluation complete!")
    else:
        print("\nRQ1 evaluation failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
