#!/usr/bin/env python3
"""
Configuration Explorer for CLC Policy Benchmark

Systematically explores different configurations:
- Number of elements (n)
- Threads per block
- Imbalance scale
- Workload scale

Runs all workloads and policies, saves comprehensive results to CSV.
"""

import subprocess
import csv
import sys
import time
from datetime import datetime
from typing import List, Dict, Tuple
import os

# Configuration ranges (from small to large)
CONFIGURATIONS = {
    'n': [
        1048576,    # 1M elements (small, fast)
        2097152,    # 2M elements (medium, default)
        4194304,    # 4M elements (large)
    ],
    'threads': [
        128,        # Small block size
        256,        # Medium (default)
        512,        # Large block size
    ],
    'imbalance_scale': [
        0.5,        # Low imbalance
        1.0,        # Medium (default)
        2.0,        # High imbalance
        5.0,        # Very high imbalance
    ],
    'workload_scale': [
        0.5,        # Light workload
        1.0,        # Medium (default)
        2.0,        # Heavy workload
    ],
}

# Policies we're comparing
POLICIES = [
    'FixedWork',
    'FixedBlocks',
    'CLCBaseline',
    'Greedy',
    'MaxSteals',
    'NeverSteal',
    'Selective',
    'WorkloadAware',
    'LatencyBudget',
    'TokenBucket',
    'Voting',
    'ClusterAware',
]

# Workloads in the benchmark
WORKLOADS = [
    'Clustered Heavy Workload (Tail Latency)',
    'MoE: Mixture of Experts Routing',
    'NLP: Variable Sequence Lengths (BERT/GPT)',
    'GNN: Variable Graph Node Degrees',
    'AI Inference: Dynamic Batching',
    'Transformer: Sparse Attention',
    'CV: Video Frame Processing',
    'GEMM: Balanced (16x16x16)',
    'GEMM: Imbalanced (Variable MxNxK)',
    'GEMM: Variable Matrix Sizes',
]


def run_benchmark(n: int, threads: int, imb_scale: float, work_scale: float) -> List[Dict]:
    """Run the benchmark with given configuration and parse results."""

    cmd = ['./clc_policy_benchmark', str(n), str(threads), str(imb_scale), str(work_scale)]

    print(f"  Running: n={n}, threads={threads}, imb={imb_scale:.1f}, work={work_scale:.1f}", flush=True)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"    ERROR: Benchmark failed with return code {result.returncode}", file=sys.stderr)
            return []

        # Parse CSV output
        lines = result.stdout.strip().split('\n')

        # Find the header line
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith('Workload,Prologue,'):
                header_idx = i
                break

        if header_idx is None:
            print(f"    ERROR: Could not find CSV header", file=sys.stderr)
            return []

        # Parse CSV
        csv_lines = lines[header_idx:]
        reader = csv.DictReader(csv_lines)

        results = []
        for row in reader:
            # Add configuration parameters to each row
            row['config_n'] = n
            row['config_threads'] = threads
            row['config_imbalance_scale'] = imb_scale
            row['config_workload_scale'] = work_scale
            row['config_blocks'] = (n + threads - 1) // threads
            results.append(row)

        print(f"    SUCCESS: Parsed {len(results)} workload results", flush=True)
        return results

    except subprocess.TimeoutExpired:
        print(f"    ERROR: Benchmark timed out after 300 seconds", file=sys.stderr)
        return []
    except Exception as e:
        print(f"    ERROR: {str(e)}", file=sys.stderr)
        return []


def calculate_speedups(row: Dict) -> Dict:
    """Calculate speedups relative to different baselines."""

    speedups = {}

    # Extract times
    try:
        greedy_time = float(row['Greedy_ms'])
        clc_baseline_time = float(row['CLCBaseline_ms'])
        fixed_work_time = float(row['FixedWork_ms'])

        for policy in POLICIES:
            policy_col = f'{policy}_ms'
            if policy_col in row and row[policy_col]:
                policy_time = float(row[policy_col])

                # Speedup vs Greedy (%)
                speedups[f'{policy}_vs_Greedy_pct'] = \
                    ((greedy_time - policy_time) / greedy_time * 100) if greedy_time > 0 else 0

                # Speedup vs CLCBaseline (%)
                speedups[f'{policy}_vs_CLCBaseline_pct'] = \
                    ((clc_baseline_time - policy_time) / clc_baseline_time * 100) if clc_baseline_time > 0 else 0

                # Speedup vs FixedWork (%)
                speedups[f'{policy}_vs_FixedWork_pct'] = \
                    ((fixed_work_time - policy_time) / fixed_work_time * 100) if fixed_work_time > 0 else 0

    except (ValueError, KeyError) as e:
        print(f"    WARNING: Could not calculate speedups: {e}", file=sys.stderr)

    return speedups


def generate_configurations() -> List[Tuple[int, int, float, float]]:
    """Generate all configuration combinations, ordered from small to large."""

    configs = []

    for n in CONFIGURATIONS['n']:
        for threads in CONFIGURATIONS['threads']:
            for imb in CONFIGURATIONS['imbalance_scale']:
                for work in CONFIGURATIONS['workload_scale']:
                    configs.append((n, threads, imb, work))

    return configs


def get_completed_configs(output_file: str) -> set:
    """Load already completed configurations from existing CSV."""

    if not os.path.exists(output_file):
        return set()

    completed = set()

    try:
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                config = (
                    int(float(row['config_n'])),
                    int(float(row['config_threads'])),
                    float(row['config_imbalance_scale']),
                    float(row['config_workload_scale'])
                )
                completed.add(config)

        print(f"  Found {len(completed)} completed configurations", flush=True)
    except Exception as e:
        print(f"  Warning: Could not read existing file: {e}", file=sys.stderr)
        return set()

    return completed


def append_results_to_csv(output_file: str, results: List[Dict], fieldnames: List[str] = None):
    """Append results to CSV file, creating it if needed."""

    file_exists = os.path.exists(output_file)

    if not results:
        return

    # Determine fieldnames
    if fieldnames is None:
        if file_exists:
            # Read existing fieldnames
            with open(output_file, 'r') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
        else:
            # Create fieldnames from first result
            fieldnames = list(results[0].keys())
            # Reorder: put config columns first
            config_cols = [col for col in fieldnames if col.startswith('config_')]
            other_cols = [col for col in fieldnames if not col.startswith('config_')]
            fieldnames = config_cols + other_cols

    # Write to file
    mode = 'a' if file_exists else 'w'
    with open(output_file, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerows(results)


def main():
    """Main execution function."""

    # Check if benchmark binary exists
    if not os.path.exists('./clc_policy_benchmark'):
        print("ERROR: clc_policy_benchmark binary not found. Run 'make' first.", file=sys.stderr)
        sys.exit(1)

    # Use fixed output filename for resumability
    output_file = 'exploration_results.csv'

    print("=" * 70)
    print("CLC Policy Benchmark - Configuration Explorer")
    print("=" * 70)
    print()
    print(f"Output file: {output_file}")
    print()

    # Check for existing results
    completed_configs = get_completed_configs(output_file)

    print("Configuration space:")
    print(f"  Elements (n):        {CONFIGURATIONS['n']}")
    print(f"  Threads per block:   {CONFIGURATIONS['threads']}")
    print(f"  Imbalance scales:    {CONFIGURATIONS['imbalance_scale']}")
    print(f"  Workload scales:     {CONFIGURATIONS['workload_scale']}")

    configs = generate_configurations()
    total_configs = len(configs)

    # Filter out already completed configs
    remaining_configs = [(i+1, cfg) for i, cfg in enumerate(configs) if cfg not in completed_configs]
    completed_count = total_configs - len(remaining_configs)

    print()
    print(f"Total configurations: {total_configs}")
    print(f"Already completed: {completed_count}")
    print(f"Remaining: {len(remaining_configs)}")
    print(f"Expected workloads per config: {len(WORKLOADS)}")
    print()
    print("=" * 70)
    print()

    if not remaining_configs:
        print("All configurations already completed!")
        print()

        # Load all results for summary
        all_results = []
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            all_results = list(reader)

        print_summary(all_results, output_file)
        return

    start_time = time.time()

    for config_idx, (n, threads, imb, work) in remaining_configs:
        print(f"[{config_idx}/{total_configs}] Configuration {config_idx}:", flush=True)

        config_start = time.time()
        results = run_benchmark(n, threads, imb, work)
        config_time = time.time() - config_start

        if results:
            # Calculate speedups for each result
            for row in results:
                speedups = calculate_speedups(row)
                row.update(speedups)

            # Persist results immediately after each test
            append_results_to_csv(output_file, results)

            print(f"    Time: {config_time:.1f}s", flush=True)
            print(f"    ✅ Results saved to {output_file}", flush=True)
        else:
            print(f"    FAILED (skipping)", flush=True)

        print(flush=True)

    total_time = time.time() - start_time

    print("=" * 70)
    print("Exploration Complete!")
    print("=" * 70)
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print()

    # Load all results for summary
    all_results = []
    with open(output_file, 'r') as f:
        reader = csv.DictReader(f)
        all_results = list(reader)

    print(f"Total results collected: {len(all_results)}")
    print()

    # Print summary statistics
    print_summary(all_results, output_file)


def print_summary(results: List[Dict], output_file: str):
    """Print summary statistics."""

    print("=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print()

    # Find best speedups for ClusteredHeavy workload
    clustered_heavy = [r for r in results if 'Clustered Heavy' in r['Workload']]

    if clustered_heavy:
        print("Best speedups for ClusteredHeavy workload:")
        print()

        # Find best LatencyBudget vs Greedy speedup
        best_latency_vs_greedy = max(
            clustered_heavy,
            key=lambda r: float(r.get('LatencyBudget_vs_Greedy_pct', 0))
        )

        speedup = float(best_latency_vs_greedy.get('LatencyBudget_vs_Greedy_pct', 0))
        config = (
            best_latency_vs_greedy['config_n'],
            best_latency_vs_greedy['config_threads'],
            best_latency_vs_greedy['config_imbalance_scale'],
            best_latency_vs_greedy['config_workload_scale'],
        )

        print(f"  LatencyBudget vs Greedy: {speedup:.2f}%")
        print(f"    at config: n={config[0]}, threads={config[1]}, imb={config[2]}, work={config[3]}")
        print()

        # Find worst policy
        policies_with_speedup = [p for p in POLICIES if f'{p}_vs_Greedy_pct' in clustered_heavy[0]]

        worst_policy = None
        worst_speedup = float('inf')

        for r in clustered_heavy:
            for policy in policies_with_speedup:
                speedup_col = f'{policy}_vs_Greedy_pct'
                if speedup_col in r:
                    speedup = float(r.get(speedup_col, 0))
                    if speedup < worst_speedup:
                        worst_speedup = speedup
                        worst_policy = policy
                        worst_config = r

        if worst_policy:
            config = (
                worst_config['config_n'],
                worst_config['config_threads'],
                worst_config['config_imbalance_scale'],
                worst_config['config_workload_scale'],
            )
            print(f"  {worst_policy} vs Greedy: {worst_speedup:.2f}% (worst)")
            print(f"    at config: n={config[0]}, threads={config[1]}, imb={config[2]}, work={config[3]}")

    print()
    print("=" * 70)
    print()
    print("Next steps:")
    print(f"  1. Analyze results: python3 analyze_results.py {output_file}")
    print(f"  2. Open in spreadsheet: libreoffice {output_file}")
    print(f"  3. Load in pandas:")
    print(f"     import pandas as pd")
    print(f"     df = pd.read_csv('{output_file}')")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...", file=sys.stderr)
        sys.exit(1)
