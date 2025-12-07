#!/usr/bin/env python3
"""
RQ2: Access Pattern Impact on UVM Performance
Analyzes how different memory access patterns (sequential, random, pointer-chase)
affect UVM performance compared to device-only mode
"""

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Configuration
KERNELS = ['seq_stream', 'rand_stream', 'pointer_chase']
MODE = 'uvm'  # Focus on UVM behavior
SIZE_FACTOR = 0.5  # Use 0.5x GPU memory
STRIDE_BYTES = 4    # Element-level for throughput comparison
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
    """Run benchmarks for all access patterns"""
    results_device = []
    results_uvm = []

    # Collect both device and UVM results for comparison
    for kernel in KERNELS:
        # Device baseline
        output_file = f'results_rq2_{kernel}_device.csv'
        if run_benchmark(kernel, 'device', SIZE_FACTOR, output_file):
            try:
                df = pd.read_csv(output_file)
                results_device.append(df)
                os.remove(output_file)
            except Exception as e:
                print(f"Error reading {output_file}: {e}")

        # UVM
        output_file = f'results_rq2_{kernel}_uvm.csv'
        if run_benchmark(kernel, 'uvm', SIZE_FACTOR, output_file):
            try:
                df = pd.read_csv(output_file)
                results_uvm.append(df)
                os.remove(output_file)
            except Exception as e:
                print(f"Error reading {output_file}: {e}")

    if results_device and results_uvm:
        df_device = pd.concat(results_device, ignore_index=True)
        df_uvm = pd.concat(results_uvm, ignore_index=True)
        combined = pd.concat([df_device, df_uvm], ignore_index=True)
        combined.to_csv('rq2_results.csv', index=False)
        return combined
    else:
        print("No results collected!")
        return None

def plot_results(df):
    """Generate visualization for RQ2"""
    if df is None or df.empty:
        print("No data to plot!")
        return

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 5)

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # Calculate metrics
    access_patterns = {
        'seq_stream': 'Sequential',
        'rand_stream': 'Random',
        'pointer_chase': 'Pointer Chase'
    }

    # Compute slowdown and throughput for each pattern
    pattern_analysis = []
    for kernel in KERNELS:
        kernel_df = df[df['kernel'] == kernel]
        device_row = kernel_df[kernel_df['mode'] == 'device']
        uvm_row = kernel_df[kernel_df['mode'] == 'uvm']

        if not device_row.empty and not uvm_row.empty:
            device_time = device_row['median_ms'].values[0]
            uvm_time = uvm_row['median_ms'].values[0]
            device_bw = device_row['bw_GBps'].values[0]
            uvm_bw = uvm_row['bw_GBps'].values[0]

            slowdown = uvm_time / device_time

            pattern_analysis.append({
                'kernel': kernel,
                'pattern': access_patterns[kernel],
                'slowdown': slowdown,
                'device_throughput_gbs': device_bw,
                'uvm_throughput_gbs': uvm_bw,
                'device_time_ms': device_time,
                'uvm_time_ms': uvm_time
            })

    analysis_df = pd.DataFrame(pattern_analysis)

    # Plot 1: Slowdown by access pattern
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    ax1.bar(range(len(analysis_df)), analysis_df['slowdown'], color=colors, width=0.6)
    ax1.set_xticks(range(len(analysis_df)))
    ax1.set_xticklabels(analysis_df['pattern'], rotation=45, ha='right')
    ax1.set_ylabel('Slowdown (UVM vs Device)', fontsize=12)
    ax1.set_title('RQ2: UVM Overhead by Access Pattern', fontsize=14, fontweight='bold')
    ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(analysis_df['slowdown']):
        ax1.text(i, v + 0.1, f'{v:.2f}x', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Throughput comparison
    x = range(len(analysis_df))
    width = 0.35
    ax2.bar([i - width/2 for i in x], analysis_df['device_throughput_gbs'],
            width, label='Device', color='#3498db')
    ax2.bar([i + width/2 for i in x], analysis_df['uvm_throughput_gbs'],
            width, label='UVM', color='#e67e22')
    ax2.set_xticks(x)
    ax2.set_xticklabels(analysis_df['pattern'], rotation=45, ha='right')
    ax2.set_ylabel('Effective Throughput (GB/s)', fontsize=12)
    ax2.set_title('RQ2: Memory Throughput', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Runtime comparison
    ax3.bar([i - width/2 for i in x], analysis_df['device_time_ms'],
            width, label='Device', color='#3498db')
    ax3.bar([i + width/2 for i in x], analysis_df['uvm_time_ms'],
            width, label='UVM', color='#e67e22')
    ax3.set_xticks(x)
    ax3.set_xticklabels(analysis_df['pattern'], rotation=45, ha='right')
    ax3.set_ylabel('Median Runtime (ms)', fontsize=12)
    ax3.set_title('RQ2: Absolute Runtime', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('rq2_access_pattern.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('rq2_access_pattern.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved: rq2_access_pattern.{pdf,png}")

    # Print summary
    print("\n" + "="*70)
    print("RQ2 SUMMARY: Access Pattern Impact on UVM Performance")
    print("="*70)
    print(f"\nConfiguration: size_factor={SIZE_FACTOR}, stride_bytes={STRIDE_BYTES}, iterations={ITERATIONS}")
    print("\nPerformance by Access Pattern:")
    print(analysis_df.to_string(index=False))

    print("\nKey Findings:")
    print(f"  - Sequential access: {analysis_df[analysis_df['kernel']=='seq_stream']['slowdown'].values[0]:.2f}x slowdown")
    print(f"  - Random access: {analysis_df[analysis_df['kernel']=='rand_stream']['slowdown'].values[0]:.2f}x slowdown")
    print(f"  - Pointer chase: {analysis_df[analysis_df['kernel']=='pointer_chase']['slowdown'].values[0]:.2f}x slowdown")

    # Identify worst case
    worst_case = analysis_df.loc[analysis_df['slowdown'].idxmax()]
    print(f"\n  → Worst case: {worst_case['pattern']} with {worst_case['slowdown']:.2f}x slowdown")
    print(f"  → This demonstrates TLB/page-fault overhead in UVM for irregular access")

def main():
    if not os.path.exists(EXECUTABLE):
        print(f"Error: {EXECUTABLE} not found!")
        print("Please run 'make' in the micro directory first.")
        sys.exit(1)

    print("="*70)
    print("RQ2: Access Pattern Impact on UVM Performance")
    print("="*70)
    print(f"Access Patterns: Sequential, Random, Pointer Chase")
    print(f"Size Factor: {SIZE_FACTOR}x GPU memory")
    print(f"Iterations: {ITERATIONS}")
    print("="*70)
    print()

    # Collect results
    df = collect_results()

    if df is not None:
        # Generate plots
        plot_results(df)
        print("\nRQ2 evaluation complete!")
    else:
        print("\nRQ2 evaluation failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
