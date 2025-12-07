#!/usr/bin/env python3
"""
RQ4: Test cudaMemAdvise and cudaMemPrefetch effectiveness
Compare different UVM memory management strategies across all kernels
"""

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Configuration
OUTPUT_CSV = "rq4_memadvise_results.csv"
OUTPUT_PLOT = "rq4_memadvise.png"
OUTPUT_PDF = "rq4_memadvise.pdf"
ITERATIONS = 5
SIZE_FACTOR = 0.25

# Test modes
MODES = [
    ("uvm", "Baseline UVM"),
    ("uvm_prefetch", "Prefetch"),
    ("uvm_advise_read", "ReadMostly"),
    ("uvm_advise_pref_gpu", "PrefGPU"),
    ("uvm_advise_pref_cpu", "PrefCPU"),
    ("uvm_advise_access", "AccessedBy"),
]

KERNELS = [
    ("seq_stream", "Sequential"),
    ("rand_stream", "Random"),
    ("pointer_chase", "PtrChase"),
]

def run_benchmark(kernel, mode, output_file):
    """Run a single benchmark configuration"""
    cmd = [
        "./uvmbench",
        f"--kernel={kernel}",
        f"--mode={mode}",
        f"--size_factor={SIZE_FACTOR}",
        f"--iterations={ITERATIONS}",
        f"--output={output_file}",
    ]

    print(f"  Running: {kernel} with {mode}...", flush=True)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ERROR: {e}")
        print(f"    stdout: {e.stdout}")
        print(f"    stderr: {e.stderr}")
        return False

def main():
    print("RQ4: Testing cudaMemAdvise Effectiveness")
    print("=" * 50)
    print()

    # Collect results
    all_results = []

    for kernel, kernel_label in KERNELS:
        print(f"Testing kernel: {kernel} ({kernel_label})")

        for mode, mode_label in MODES:
            temp_file = "temp_result.csv"

            if run_benchmark(kernel, mode, temp_file):
                # Read result
                try:
                    df = pd.read_csv(temp_file)
                    df['mode_label'] = mode_label
                    df['kernel_label'] = kernel_label
                    all_results.append(df)
                except Exception as e:
                    print(f"    ERROR reading results: {e}")

        print()

    # Combine all results
    if not all_results:
        print("ERROR: No results collected!")
        sys.exit(1)

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to: {OUTPUT_CSV}")
    print()

    # Create visualization
    print("Creating visualizations...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (kernel, kernel_label) in enumerate(KERNELS):
        ax = axes[idx]

        # Filter data for this kernel
        df_kernel = df_all[df_all['kernel'] == kernel].copy()

        # Sort by median time for better visualization
        df_kernel = df_kernel.sort_values('median_ms')

        # Create bar plot
        bars = ax.barh(df_kernel['mode_label'], df_kernel['median_ms'])

        # Color bars: baseline in gray, others in color
        colors = ['gray' if mode == 'Baseline UVM' else 'steelblue'
                  for mode in df_kernel['mode_label']]
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel('Execution Time (ms)', fontsize=11)
        ax.set_title(f'{kernel_label}', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels on bars
        for i, (idx_row, row) in enumerate(df_kernel.iterrows()):
            ax.text(row['median_ms'], i, f" {row['median_ms']:.3f}ms",
                   va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_PDF, bbox_inches='tight')
    print(f"Plots saved to: {OUTPUT_PLOT} and {OUTPUT_PDF}")
    print()

    # Print summary
    print("Summary Statistics:")
    print("=" * 50)
    for kernel, kernel_label in KERNELS:
        print(f"\n{kernel_label}:")
        df_k = df_all[df_all['kernel'] == kernel][['mode_label', 'median_ms', 'bw_GBps']]
        df_k = df_k.sort_values('median_ms')
        print(df_k.to_string(index=False))

    print("\n" + "=" * 50)
    print("Mode Descriptions:")
    print("  Baseline UVM   - On-demand page faulting (no hints)")
    print("  Prefetch       - cudaMemPrefetchAsync (explicit migration)")
    print("  ReadMostly     - cudaMemAdviseSetReadMostly (allow read duplication)")
    print("  PrefGPU        - cudaMemAdviseSetPreferredLocation(GPU)")
    print("  PrefCPU        - cudaMemAdviseSetPreferredLocation(CPU)")
    print("  AccessedBy     - cudaMemAdviseSetAccessedBy(GPU)")

if __name__ == "__main__":
    main()
