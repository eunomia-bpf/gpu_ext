#!/usr/bin/env python3
"""
Analyze and visualize GPU performance bottleneck benchmark results.
Covers: Memory Coalescing, Thread Divergence, Atomic Contention, Roofline
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    try:
        df = pd.read_csv('results.csv')
    except FileNotFoundError:
        print("Error: results.csv not found. Run ./bench first.")
        sys.exit(1)

    # Separate test types
    mem_df = df[df['test_type'] == 'memory']
    div_df = df[df['test_type'] == 'divergence']
    atomic_df = df[df['test_type'] == 'atomic']
    roofline_df = df[df['test_type'] == 'roofline']

    # Create figure with 2x2 grid + 1 extra row for roofline
    fig = plt.figure(figsize=(14, 14))

    # ===== Plot 1: Memory Coalescing - Slowdown =====
    ax1 = fig.add_subplot(3, 2, 1)
    if len(mem_df) > 0:
        baseline_time = mem_df.iloc[0]['time_ms']
        slowdowns = mem_df['time_ms'] / baseline_time

        # Create labels - handle "random" specially
        labels = []
        colors = []
        for p in mem_df['parameter']:
            if str(p) == 'random':
                labels.append('random')
                colors.append('darkred')  # Different color for random
            else:
                labels.append(f"stride={p}")
                colors.append('steelblue')

        bars = ax1.bar(range(len(mem_df)), slowdowns, color=colors)
        ax1.set_xticks(range(len(mem_df)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylabel('Slowdown (x)')
        ax1.set_xlabel('Access Pattern')
        ax1.set_title('1. Memory Coalescing: Slowdown vs Access Pattern')
        ax1.grid(axis='y', alpha=0.3)

        for i, (s, bw) in enumerate(zip(slowdowns, mem_df['bandwidth_gbps'])):
            ax1.annotate(f"{s:.1f}x\n({bw:.0f} GB/s)",
                         (i, s), ha='center', va='bottom', fontsize=7)

    # ===== Plot 2: Thread Divergence - Slowdown =====
    ax2 = fig.add_subplot(3, 2, 2)
    if len(div_df) > 0:
        baseline_time = div_df.iloc[0]['time_ms']
        slowdowns = div_df['time_ms'] / baseline_time

        ax2.bar(range(len(div_df)), slowdowns, color='coral')
        ax2.set_xticks(range(len(div_df)))
        ax2.set_xticklabels([f"div={p}" for p in div_df['parameter']])
        ax2.set_ylabel('Slowdown (x)')
        ax2.set_xlabel('Divergence Factor')
        ax2.set_title('2. Thread Divergence: Slowdown vs Factor')
        ax2.grid(axis='y', alpha=0.3)

        for i, s in enumerate(slowdowns):
            ax2.annotate(f"{s:.1f}x", (i, s), ha='center', va='bottom', fontsize=8)

    # ===== Plot 3: Atomic Contention - Slowdown =====
    ax3 = fig.add_subplot(3, 2, 3)
    if len(atomic_df) > 0:
        baseline_time = atomic_df.iloc[0]['time_ms']
        slowdowns = atomic_df['time_ms'] / baseline_time

        ax3.bar(range(len(atomic_df)), slowdowns, color='forestgreen')
        ax3.set_xticks(range(len(atomic_df)))
        ax3.set_xticklabels([str(p) for p in atomic_df['parameter']])
        ax3.set_ylabel('Slowdown (x)')
        ax3.set_xlabel('Contention Factor (threads per counter)')
        ax3.set_title('3. Atomic Contention: Slowdown vs Contention')
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_yscale('log')

        for i, (s, tp) in enumerate(zip(slowdowns, atomic_df['bandwidth_gbps'])):
            ax3.annotate(f"{s:.0f}x\n({tp:.0f}M/s)",
                         (i, s), ha='center', va='bottom', fontsize=8)

    # ===== Plot 4: Roofline - Slowdown =====
    ax4 = fig.add_subplot(3, 2, 4)
    if len(roofline_df) > 0:
        baseline_time = roofline_df.iloc[0]['time_ms']
        slowdowns = roofline_df['time_ms'] / baseline_time
        ai_values = roofline_df['parameter'].astype(float) * 2 / 8  # AI = flops * 2 / 8

        ax4.bar(range(len(roofline_df)), slowdowns, color='purple')
        ax4.set_xticks(range(len(roofline_df)))
        ax4.set_xticklabels([f"{ai:.1f}" for ai in ai_values], rotation=45)
        ax4.set_ylabel('Slowdown (x)')
        ax4.set_xlabel('Arithmetic Intensity (FLOP/Byte)')
        ax4.set_title('4. Roofline: Slowdown vs AI (more compute = slower)')
        ax4.grid(axis='y', alpha=0.3)

        for i, (s, g) in enumerate(zip(slowdowns, roofline_df['bandwidth_gbps'])):
            if i % 2 == 0:  # Show every other label to avoid crowding
                ax4.annotate(f"{s:.1f}x", (i, s), ha='center', va='bottom', fontsize=7)

    # ===== Plot 5: Complete Roofline Model (log-log) =====
    ax5 = fig.add_subplot(3, 1, 3)
    if len(roofline_df) > 0:
        ai_values = roofline_df['parameter'].astype(float) * 2 / 8
        gflops = roofline_df['bandwidth_gbps']  # This stores GFLOPS
        bandwidth = roofline_df['efficiency']   # This stores bandwidth for roofline

        # Plot actual performance
        ax5.loglog(ai_values, gflops, 'o-', color='purple', linewidth=2, markersize=8, label='Measured GFLOPS')

        # Draw roofline model lines
        peak_bandwidth = bandwidth.max()  # Approximate peak bandwidth from data
        peak_gflops = gflops.max()        # Approximate peak compute from data

        # Memory-bound line: GFLOPS = AI * bandwidth
        ai_range = np.logspace(-1, 3, 100)
        mem_bound_line = ai_range * peak_bandwidth
        ax5.loglog(ai_range, mem_bound_line, '--', color='steelblue', linewidth=1.5, alpha=0.7, label=f'Memory Bound ({peak_bandwidth:.0f} GB/s)')

        # Compute-bound line: GFLOPS = peak
        ax5.axhline(y=peak_gflops, color='coral', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Compute Bound ({peak_gflops:.0f} GFLOPS)')

        # Ridge point
        ridge_ai = peak_gflops / peak_bandwidth
        ax5.axvline(x=ridge_ai, color='gray', linestyle=':', alpha=0.5)
        ax5.annotate(f'Ridge Point\nAI={ridge_ai:.1f}', (ridge_ai, peak_gflops/2),
                     ha='center', fontsize=9, color='gray')

        ax5.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=11)
        ax5.set_ylabel('Performance (GFLOPS)', fontsize=11)
        ax5.set_title('5. Roofline Model: Performance vs Arithmetic Intensity', fontsize=12)
        ax5.legend(loc='lower right')
        ax5.grid(True, alpha=0.3, which='both')
        ax5.set_xlim(0.1, 500)

        # Add annotations for key points
        for i in [0, len(ai_values)//2, len(ai_values)-1]:
            ax5.annotate(f'AI={ai_values.iloc[i]:.1f}\n{gflops.iloc[i]:.0f} GFLOPS',
                         (ai_values.iloc[i], gflops.iloc[i]),
                         textcoords='offset points', xytext=(10, 10), fontsize=8)

    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    print("Chart saved to: results.png")

    # Print summary with slowdowns
    print("\n" + "="*70)
    print("GPU Performance Bottleneck Benchmark Summary (with Slowdowns)")
    print("="*70)

    if len(mem_df) > 0:
        baseline_time = mem_df.iloc[0]['time_ms']
        max_slowdown = mem_df.iloc[-1]['time_ms'] / baseline_time
        print(f"\n1. Memory Coalescing:")
        print(f"   stride=1 → stride={mem_df.iloc[-1]['parameter']}: "
              f"{max_slowdown:.1f}x slowdown")
        print(f"   Bandwidth: {mem_df.iloc[0]['bandwidth_gbps']:.0f} → {mem_df.iloc[-1]['bandwidth_gbps']:.0f} GB/s")

    if len(div_df) > 0:
        baseline_time = div_df.iloc[0]['time_ms']
        max_slowdown = div_df.iloc[-1]['time_ms'] / baseline_time
        print(f"\n2. Thread Divergence:")
        print(f"   div=1 → div={div_df.iloc[-1]['parameter']}: {max_slowdown:.1f}x slowdown")

    if len(atomic_df) > 0:
        baseline_time = atomic_df.iloc[0]['time_ms']
        max_slowdown = atomic_df.iloc[-1]['time_ms'] / baseline_time
        print(f"\n3. Atomic Contention:")
        print(f"   contention=1 → ALL: {max_slowdown:.0f}x slowdown")
        print(f"   Throughput: {atomic_df.iloc[0]['bandwidth_gbps']:.0f} → {atomic_df.iloc[-1]['bandwidth_gbps']:.0f} M ops/s")

    if len(roofline_df) > 0:
        baseline_time = roofline_df.iloc[0]['time_ms']
        max_slowdown = roofline_df.iloc[-1]['time_ms'] / baseline_time
        ai_values = roofline_df['parameter'].astype(float) * 2 / 8
        print(f"\n4. Roofline (Arithmetic Intensity):")
        print(f"   AI={ai_values.iloc[0]:.2f} → AI={ai_values.iloc[-1]:.1f}: {max_slowdown:.1f}x slowdown (more compute)")
        print(f"   GFLOPS: {roofline_df.iloc[0]['bandwidth_gbps']:.0f} → {roofline_df.iloc[-1]['bandwidth_gbps']:.0f}")

if __name__ == '__main__':
    main()
