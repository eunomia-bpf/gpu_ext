#!/usr/bin/env python3
"""
Quick Analysis: Check if CLC beats static scheduling and if policies beat baseline CLC
"""

import pandas as pd
import sys

def analyze_clc_vs_static(df):
    """Question 1: Can CLC be much better than FixedWork or FixedBlocks?"""

    print("=" * 70)
    print("QUESTION 1: Can CLC (dynamic scheduling) beat static scheduling?")
    print("=" * 70)
    print()

    # Compare CLCBaseline vs FixedWork
    print("CLCBaseline vs FixedWork:")
    print("-" * 70)

    for workload in df['Workload'].unique():
        workload_df = df[df['Workload'] == workload]

        # Calculate average speedup across all configs
        if 'CLCBaseline_vs_FixedWork_pct' in workload_df.columns:
            avg_speedup = workload_df['CLCBaseline_vs_FixedWork_pct'].astype(float).mean()
            max_speedup = workload_df['CLCBaseline_vs_FixedWork_pct'].astype(float).max()
            min_speedup = workload_df['CLCBaseline_vs_FixedWork_pct'].astype(float).min()

            status = "✅ CLC WINS" if avg_speedup > 5 else "⚠️  MARGINAL" if avg_speedup > 0 else "❌ STATIC WINS"

            print(f"{workload[:50]:50s}")
            print(f"  Avg: {avg_speedup:6.1f}%  Max: {max_speedup:6.1f}%  Min: {min_speedup:6.1f}%  {status}")

            # Show best configuration
            if max_speedup > 10:
                best_row = workload_df.loc[workload_df['CLCBaseline_vs_FixedWork_pct'].astype(float).idxmax()]
                print(f"  Best config: n={int(float(best_row['config_n']))}, threads={int(float(best_row['config_threads']))}, imb={float(best_row['config_imbalance_scale']):.1f}, work={float(best_row['config_workload_scale']):.1f}")

    print()
    print("=" * 70)
    print()


def analyze_policies_vs_clc(df):
    """Question 2: Can a policy be much better than baseline CLC (Greedy)?"""

    print("=" * 70)
    print("QUESTION 2: Can specialized policies beat baseline CLC (Greedy)?")
    print("=" * 70)
    print()

    # List of policies to check (excluding baselines)
    policies = ['LatencyBudget', 'TokenBucket', 'Voting', 'ClusterAware',
                'WorkloadAware', 'Selective', 'MaxSteals', 'NeverSteal']

    for workload in df['Workload'].unique():
        workload_df = df[df['Workload'] == workload]

        print(f"{workload}")
        print("-" * 70)

        best_policy = None
        best_speedup = -999

        for policy in policies:
            speedup_col = f'{policy}_vs_Greedy_pct'
            if speedup_col in workload_df.columns:
                avg_speedup = workload_df[speedup_col].astype(float).mean()
                max_speedup = workload_df[speedup_col].astype(float).max()

                if avg_speedup > best_speedup:
                    best_speedup = avg_speedup
                    best_policy = policy
                    best_max = max_speedup

        if best_policy:
            status = "🏆 BIG WIN" if best_speedup > 10 else "✅ WIN" if best_speedup > 5 else "⚠️  SMALL" if best_speedup > 1 else "❌ NO WIN"
            print(f"  Best: {best_policy:20s}  Avg: {best_speedup:6.1f}%  Max: {best_max:6.1f}%  {status}")

            # Show best configuration for this policy
            if best_speedup > 5:
                speedup_col = f'{best_policy}_vs_Greedy_pct'
                best_row = workload_df.loc[workload_df[speedup_col].astype(float).idxmax()]
                print(f"    Best config: n={int(float(best_row['config_n']))}, threads={int(float(best_row['config_threads']))}, imb={float(best_row['config_imbalance_scale']):.1f}, work={float(best_row['config_workload_scale']):.1f}")
                print(f"    Greedy: {float(best_row['Greedy_ms']):.3f}ms, {best_policy}: {float(best_row[f'{best_policy}_ms']):.3f}ms")

        print()

    print("=" * 70)
    print()


def find_smoking_guns(df):
    """Find the most dramatic evidence for both questions."""

    print("=" * 70)
    print("🔍 TOP CASES FOR EACH QUESTION")
    print("=" * 70)
    print()

    # Question 1: CLC vs Static - show top 5
    print("1️⃣  TOP 5: CLC beats static scheduling")
    print("-" * 70)

    if 'CLCBaseline_vs_FixedWork_pct' in df.columns:
        top_clc = df.nlargest(5, 'CLCBaseline_vs_FixedWork_pct')

        for i, (idx, row) in enumerate(top_clc.iterrows(), 1):
            speedup = float(row['CLCBaseline_vs_FixedWork_pct'])
            print(f"\n#{i}  Speedup: {speedup:.1f}%")
            print(f"    Workload: {row['Workload']}")
            print(f"    FixedWork: {float(row['FixedWork_ms']):.3f}ms → CLCBaseline: {float(row['CLCBaseline_ms']):.3f}ms")
            print(f"    Config: n={int(float(row['config_n']))}, threads={int(float(row['config_threads']))}, imb={float(row['config_imbalance_scale']):.1f}, work={float(row['config_workload_scale']):.1f}")

    print()
    print("2️⃣  TOP 5: Specialized policy beats baseline CLC (Greedy)")
    print("-" * 70)

    # Find best policy speedup for each row, then get top 5
    policies = ['LatencyBudget', 'TokenBucket', 'Voting', 'ClusterAware',
                'WorkloadAware', 'Selective', 'MaxSteals', 'NeverSteal']

    best_policy_speedups = []

    for idx, row in df.iterrows():
        best_speedup = -999
        best_policy = None

        for policy in policies:
            speedup_col = f'{policy}_vs_Greedy_pct'
            if speedup_col in row:
                speedup = float(row[speedup_col])
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_policy = policy

        if best_policy:
            best_policy_speedups.append({
                'idx': idx,
                'speedup': best_speedup,
                'policy': best_policy,
                'row': row
            })

    # Sort and get top 5
    top_policies = sorted(best_policy_speedups, key=lambda x: x['speedup'], reverse=True)[:5]

    for i, item in enumerate(top_policies, 1):
        row = item['row']
        policy = item['policy']
        speedup = item['speedup']

        print(f"\n#{i}  Policy: {policy}  Speedup: {speedup:.1f}%")
        print(f"    Workload: {row['Workload']}")
        print(f"    Greedy: {float(row['Greedy_ms']):.3f}ms ({int(float(row['Greedy_steals']))} steals)")
        print(f"    {policy}: {float(row[f'{policy}_ms']):.3f}ms ({int(float(row[f'{policy}_steals']))} steals)")
        print(f"    Config: n={int(float(row['config_n']))}, threads={int(float(row['config_threads']))}, imb={float(row['config_imbalance_scale']):.1f}, work={float(row['config_workload_scale']):.1f}")

    print()
    print("=" * 70)


def show_combined_metrics_table(df):
    """Show a table with both metrics for all workload cases."""

    print()
    print("=" * 70)
    print("📊 COMBINED METRICS TABLE: All Workload Cases")
    print("=" * 70)
    print()
    print("For each configuration, showing:")
    print("  Metric 1: CLC vs FixedWork speedup (%)")
    print("  Metric 2: Best policy vs Greedy speedup (%)")
    print("=" * 70)
    print()

    policies = ['LatencyBudget', 'TokenBucket', 'Voting', 'ClusterAware',
                'WorkloadAware', 'Selective', 'MaxSteals', 'NeverSteal']

    # Collect all data
    table_data = []

    for idx, row in df.iterrows():
        clc_vs_fixed = float(row.get('CLCBaseline_vs_FixedWork_pct', 0))

        # Find best policy
        best_speedup = -999
        best_policy = None

        for policy in policies:
            speedup_col = f'{policy}_vs_Greedy_pct'
            if speedup_col in row:
                speedup = float(row[speedup_col])
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_policy = policy

        table_data.append({
            'workload': row['Workload'],
            'config_n': int(float(row['config_n'])),
            'config_threads': int(float(row['config_threads'])),
            'config_imb': float(row['config_imbalance_scale']),
            'config_work': float(row['config_workload_scale']),
            'metric1_clc_vs_fixed': clc_vs_fixed,
            'metric2_policy_vs_greedy': best_speedup,
            'best_policy': best_policy if best_policy else 'N/A',
            'both_positive': clc_vs_fixed > 0 and best_speedup > 0,
            'both_good': clc_vs_fixed > 5 and best_speedup > 5,
            'both_great': clc_vs_fixed > 10 and best_speedup > 10,
        })

    # Sort by combined score
    table_data.sort(key=lambda x: x['metric1_clc_vs_fixed'] + x['metric2_policy_vs_greedy'], reverse=True)

    # Print header
    print(f"{'#':<4} {'Workload':<35} {'Config':<22} {'M1:CLC':<9} {'M2:Policy':<12} {'Best Policy':<15} {'Status':<10}")
    print("-" * 140)

    # Print rows
    for i, item in enumerate(table_data, 1):
        config_str = f"n={item['config_n']//1000}K,t={item['config_threads']},i={item['config_imb']:.1f},w={item['config_work']:.1f}"

        # Determine status
        if item['both_great']:
            status = "🏆 BOTH>10"
        elif item['both_good']:
            status = "✅ BOTH>5"
        elif item['both_positive']:
            status = "⚠️  BOTH>0"
        elif item['metric1_clc_vs_fixed'] > 5:
            status = "M1 only"
        elif item['metric2_policy_vs_greedy'] > 5:
            status = "M2 only"
        else:
            status = "-"

        workload_short = item['workload'][:35]

        print(f"{i:<4} {workload_short:<35} {config_str:<22} {item['metric1_clc_vs_fixed']:>7.1f}% {item['metric2_policy_vs_greedy']:>10.1f}% {item['best_policy']:<15} {status:<10}")

    print()
    print("=" * 70)
    print()

    # Summary statistics
    both_positive_count = sum(1 for item in table_data if item['both_positive'])
    both_good_count = sum(1 for item in table_data if item['both_good'])
    both_great_count = sum(1 for item in table_data if item['both_great'])

    print("Summary:")
    print(f"  Total configurations: {len(table_data)}")
    print(f"  Both metrics > 0%:    {both_positive_count} ({both_positive_count*100//len(table_data)}%)")
    print(f"  Both metrics > 5%:    {both_good_count} ({both_good_count*100//len(table_data) if len(table_data) > 0 else 0}%)")
    print(f"  Both metrics > 10%:   {both_great_count} ({both_great_count*100//len(table_data) if len(table_data) > 0 else 0}%)")
    print()
    print("=" * 70)


def find_both_satisfied(df):
    """Find cases where BOTH conditions are satisfied simultaneously."""

    print()
    print("=" * 70)
    print("🎯 HOLY GRAIL: Both conditions satisfied simultaneously!")
    print("=" * 70)
    print("   Condition 1: CLC beats FixedWork by >10%")
    print("   Condition 2: A policy beats Greedy by >10%")
    print("=" * 70)
    print()

    policies = ['LatencyBudget', 'TokenBucket', 'Voting', 'ClusterAware',
                'WorkloadAware', 'Selective', 'MaxSteals', 'NeverSteal']

    both_satisfied = []

    for idx, row in df.iterrows():
        # Check condition 1: CLC beats FixedWork
        clc_vs_fixed = float(row.get('CLCBaseline_vs_FixedWork_pct', -999))

        if clc_vs_fixed > 10:
            # Check condition 2: Best policy beats Greedy
            best_speedup = -999
            best_policy = None

            for policy in policies:
                speedup_col = f'{policy}_vs_Greedy_pct'
                if speedup_col in row:
                    speedup = float(row[speedup_col])
                    if speedup > best_speedup:
                        best_speedup = speedup
                        best_policy = policy

            if best_speedup > 10:
                both_satisfied.append({
                    'clc_speedup': clc_vs_fixed,
                    'policy_speedup': best_speedup,
                    'policy': best_policy,
                    'total_score': clc_vs_fixed + best_speedup,
                    'row': row
                })

    if not both_satisfied:
        print("❌ No configurations found where BOTH conditions are satisfied with >10% threshold.")
        print()
        print("Trying lower threshold (>5% for both)...")
        print()

        # Try with lower threshold
        for idx, row in df.iterrows():
            clc_vs_fixed = float(row.get('CLCBaseline_vs_FixedWork_pct', -999))

            if clc_vs_fixed > 5:
                best_speedup = -999
                best_policy = None

                for policy in policies:
                    speedup_col = f'{policy}_vs_Greedy_pct'
                    if speedup_col in row:
                        speedup = float(row[speedup_col])
                        if speedup > best_speedup:
                            best_speedup = speedup
                            best_policy = policy

                if best_speedup > 5:
                    both_satisfied.append({
                        'clc_speedup': clc_vs_fixed,
                        'policy_speedup': best_speedup,
                        'policy': best_policy,
                        'total_score': clc_vs_fixed + best_speedup,
                        'row': row
                    })

    if both_satisfied:
        # Sort by total score
        both_satisfied.sort(key=lambda x: x['total_score'], reverse=True)

        print(f"✅ Found {len(both_satisfied)} configuration(s) satisfying both conditions!")
        print()

        for i, item in enumerate(both_satisfied[:5], 1):  # Show top 5
            row = item['row']

            print(f"#{i}  Total Score: {item['total_score']:.1f}%")
            print(f"    Workload: {row['Workload']}")
            print(f"    Config: n={int(float(row['config_n']))}, threads={int(float(row['config_threads']))}, imb={float(row['config_imbalance_scale']):.1f}, work={float(row['config_workload_scale']):.1f}")
            print()
            print(f"    ✅ Condition 1: CLC beats FixedWork by {item['clc_speedup']:.1f}%")
            print(f"       FixedWork: {float(row['FixedWork_ms']):.3f}ms → CLCBaseline: {float(row['CLCBaseline_ms']):.3f}ms")
            print()
            print(f"    ✅ Condition 2: {item['policy']} beats Greedy by {item['policy_speedup']:.1f}%")
            print(f"       Greedy: {float(row['Greedy_ms']):.3f}ms → {item['policy']}: {float(row[f'{item['policy']}_ms']):.3f}ms")
            print()
            print(f"    📊 Combined benefit:")
            print(f"       FixedWork: {float(row['FixedWork_ms']):.3f}ms")
            print(f"       → {item['policy']}: {float(row[f'{item['policy']}_ms']):.3f}ms")
            print(f"       Total speedup: {((float(row['FixedWork_ms']) - float(row[f'{item['policy']}_ms'])) / float(row['FixedWork_ms']) * 100):.1f}%")
            print()
            print("-" * 70)
            print()
    else:
        print("❌ No configurations found even with lower threshold (>5%).")
        print()
        print("This suggests: Dynamic workloads benefit from CLC, but don't need")
        print("               throttling policies (Greedy is already good).")
        print("               Static workloads need throttling, but CLC itself loses")
        print("               to FixedWork baseline.")

    print("=" * 70)


def main():
    csv_file = '/home/yunwei37/workspace/playground/co-processor-demo/gpu_examples/clc_bench/exploration_results.csv'

    print()
    print("Loading partial results...")

    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} results from {df['Workload'].nunique()} workloads")

        # Count unique configs
        unique_configs = df.groupby(['config_n', 'config_threads', 'config_imbalance_scale', 'config_workload_scale']).size()
        print(f"Configurations tested: {len(unique_configs)} / 108")
        print()

    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    # Run analyses
    analyze_clc_vs_static(df)
    analyze_policies_vs_clc(df)
    find_smoking_guns(df)
    show_combined_metrics_table(df)
    find_both_satisfied(df)

    print()
    print("Note: Analysis based on partial results.")
    print()


if __name__ == '__main__':
    main()
