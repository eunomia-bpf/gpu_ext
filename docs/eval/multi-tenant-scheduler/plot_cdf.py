#!/usr/bin/env python3
"""
Generate CDF plot for LC launch latency: Native vs Policy
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Configure matplotlib for better quality
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
})

DATA_DIR = Path(__file__).parent / "simple_test_results"
NUM_RUNS = 10

def load_lc_data(mode):
    """Load all LC latency data for a mode (native or policy)."""
    latencies = []
    for run in range(NUM_RUNS):
        for proc in range(2):  # 2 LC processes
            csv_file = DATA_DIR / f"{mode}_run{run}_lc_{proc}.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file)
                # Convert ms to µs
                latencies.extend(df['launch_latency_ms'].values * 1000)
    return np.array(latencies)

def plot_cdf(data, label, color, linestyle='-'):
    """Plot CDF for given data."""
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, label=label, color=color, linestyle=linestyle, linewidth=2)

def main():
    # Load data
    native_lc = load_lc_data("native")
    policy_lc = load_lc_data("policy")

    print(f"Native LC samples: {len(native_lc)}")
    print(f"Policy LC samples: {len(policy_lc)}")
    print(f"Native LC P99: {np.percentile(native_lc, 99):.1f} µs")
    print(f"Policy LC P99: {np.percentile(policy_lc, 99):.1f} µs")

    # Create figure
    fig, ax = plt.subplots()

    # Plot CDFs
    plot_cdf(native_lc, "Native", color='#E74C3C', linestyle='--')
    plot_cdf(policy_lc, "BPF Policy", color='#2ECC71', linestyle='-')

    # Add P99 lines
    native_p99 = np.percentile(native_lc, 99)
    policy_p99 = np.percentile(policy_lc, 99)

    ax.axhline(y=0.99, color='gray', linestyle=':', alpha=0.7)
    ax.text(ax.get_xlim()[1] * 0.95, 0.985, 'P99', ha='right', va='top', color='gray')

    # Set log scale for x-axis to show tail better
    ax.set_xscale('log')

    # Labels
    ax.set_xlabel('Launch Latency (µs)')
    ax.set_ylabel('CDF')
    ax.set_title('LC Workload Launch Latency Distribution')

    # Legend
    ax.legend(loc='lower right')

    # Grid
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.02)

    # Add annotation for P99 values
    ax.annotate(f'Native P99: {native_p99:.0f}µs',
                xy=(native_p99, 0.99),
                xytext=(native_p99 * 2, 0.85),
                arrowprops=dict(arrowstyle='->', color='#E74C3C'),
                color='#E74C3C')
    ax.annotate(f'Policy P99: {policy_p99:.0f}µs',
                xy=(policy_p99, 0.99),
                xytext=(policy_p99 * 0.3, 0.75),
                arrowprops=dict(arrowstyle='->', color='#2ECC71'),
                color='#2ECC71')

    # Save
    plt.tight_layout()
    output_path = Path(__file__).parent / "fig_cdf.pdf"
    plt.savefig(output_path, bbox_inches='tight')
    print(f"\nSaved to {output_path}")

    # Also save PNG for quick viewing
    output_png = Path(__file__).parent / "fig_cdf.png"
    plt.savefig(output_png, bbox_inches='tight', dpi=150)
    print(f"Saved to {output_png}")

if __name__ == "__main__":
    main()
