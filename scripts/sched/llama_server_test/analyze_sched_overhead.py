#!/usr/bin/env python3
"""
Analyze CPU scheduler overhead for llama-server tests.
Parses vllm bench output and cuda_sched_trace data.
Supports multiple runs per scenario for averaging.
"""

import pandas as pd
import numpy as np
import re
import sys
import os
import glob
from pathlib import Path

SCENARIOS = [
    ("Baseline", "baseline"),
    ("CPU Stress", "cpu_stress"),
    ("Network Stress", "network_stress"),
    ("Disk Stress", "disk_stress"),
    ("Heavy Load", "heavy_load"),
    ("CPU Pinned", "cpu_pinned"),
]


def parse_vllm_bench_output(filepath):
    """Parse vllm bench serve output for metrics."""
    metrics = {
        'successful_requests': 0,
        'benchmark_duration': 0.0,
        'total_input_tokens': 0,
        'total_generated_tokens': 0,
        'request_throughput': 0.0,
        'output_token_throughput': 0.0,
        'mean_ttft': 0.0,
        'median_ttft': 0.0,
        'p99_ttft': 0.0,
        'mean_tpot': 0.0,
        'median_tpot': 0.0,
        'p99_tpot': 0.0,
        'mean_itl': 0.0,
        'median_itl': 0.0,
        'p99_itl': 0.0,
    }

    try:
        with open(filepath) as f:
            content = f.read()

        patterns = {
            'successful_requests': r'Successful requests:\s+(\d+)',
            'benchmark_duration': r'Benchmark duration \(s\):\s+([\d.]+)',
            'total_input_tokens': r'Total input tokens:\s+(\d+)',
            'total_generated_tokens': r'Total generated tokens:\s+(\d+)',
            'request_throughput': r'Request throughput \(req/s\):\s+([\d.]+)',
            'output_token_throughput': r'Output token throughput \(tok/s\):\s+([\d.]+)',
            'mean_ttft': r'Mean TTFT \(ms\):\s+([\d.]+)',
            'median_ttft': r'Median TTFT \(ms\):\s+([\d.]+)',
            'p99_ttft': r'P99 TTFT \(ms\):\s+([\d.]+)',
            'mean_tpot': r'Mean TPOT \(ms\):\s+([\d.]+)',
            'median_tpot': r'Median TPOT \(ms\):\s+([\d.]+)',
            'p99_tpot': r'P99 TPOT \(ms\):\s+([\d.]+)',
            'mean_itl': r'Mean ITL \(ms\):\s+([\d.]+)',
            'median_itl': r'Median ITL \(ms\):\s+([\d.]+)',
            'p99_itl': r'P99 ITL \(ms\):\s+([\d.]+)',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                value = match.group(1)
                metrics[key] = int(value) if key in ['successful_requests', 'total_input_tokens', 'total_generated_tokens'] else float(value)

    except Exception as e:
        print(f"Warning: Failed to parse {filepath}: {e}", file=sys.stderr)

    return metrics


def parse_trace_log(filepath):
    """Parse cuda_sched_trace log for event counts."""
    stats = {
        'launches': 0,
        'sched_switches': 0,
        'hard_irqs': 0,
        'soft_irqs': 0,
    }

    try:
        with open(filepath) as f:
            content = f.read()

        patterns = {
            'launches': r'cuLaunchKernel\s+(\d+)',
            'sched_switches': r'Sched Switches Tracked\s+(\d+)',
            'hard_irqs': r'Hard IRQs Tracked\s+(\d+)',
            'soft_irqs': r'Soft IRQs Tracked\s+(\d+)',
        }

        for key, pattern in patterns.items():
            for line in content.split('\n'):
                if key == 'launches' and 'cuLaunchKernel' in line and 'Tracked' not in line:
                    match = re.search(pattern, line)
                    if match:
                        stats[key] = int(match.group(1))
                elif key != 'launches':
                    match = re.search(pattern, line)
                    if match:
                        stats[key] = int(match.group(1))

    except Exception as e:
        print(f"Warning: Failed to parse {filepath}: {e}", file=sys.stderr)

    return stats


def parse_trace_csv(filepath):
    """Parse cuda_sched_trace CSV for detailed analysis."""
    result = {
        'runtime_s': 0.0,
        'irq_time_ms': 0.0,
        'softirq_details': {},
    }

    try:
        df = pd.read_csv(filepath, low_memory=False)

        # Runtime from timestamps
        if 'timestamp_ns' in df.columns:
            result['runtime_s'] = df['timestamp_ns'].max() / 1e9

        # IRQ time from exit events
        if 'event_type' in df.columns and 'duration_ns' in df.columns:
            irq_exits = df[df['event_type'].str.contains('irqExit', na=False)]
            result['irq_time_ms'] = irq_exits['duration_ns'].sum() / 1e6

            # Soft IRQ breakdown
            soft_irq_exits = df[df['event_type'] == 'softirqExit']
            for _, row in soft_irq_exits.iterrows():
                irq_name = str(row.get('irq_name', 'unknown'))
                duration_ns = row.get('duration_ns', 0)

                if irq_name not in result['softirq_details']:
                    result['softirq_details'][irq_name] = {'count': 0, 'total_time_us': 0}

                result['softirq_details'][irq_name]['count'] += 1
                result['softirq_details'][irq_name]['total_time_us'] += duration_ns / 1000

    except Exception as e:
        print(f"Warning: Failed to parse {filepath}: {e}", file=sys.stderr)

    return result


def collect_multi_run_results(output_dir, prefix, mode):
    """Collect results from multiple runs and compute average/std."""
    # Try new format (run1, run2, run3)
    run_files = sorted(glob.glob(str(output_dir / f"{prefix}_{mode}_run*.log")))

    # Fallback to old format (single file)
    if not run_files:
        old_file = output_dir / f"{prefix}_{mode}_bench.log"
        if old_file.exists():
            run_files = [str(old_file)]

    if not run_files:
        return None

    all_metrics = []
    for f in run_files:
        metrics = parse_vllm_bench_output(f)
        if metrics['successful_requests'] > 0:
            all_metrics.append(metrics)

    if not all_metrics:
        return None

    # Compute average and std for key metrics
    result = {
        'scenario': '',
        'prefix': prefix,
        'num_runs': len(all_metrics),
    }

    numeric_keys = ['output_token_throughput', 'mean_tpot', 'p99_tpot',
                    'mean_ttft', 'p99_ttft', 'successful_requests']

    for key in numeric_keys:
        values = [m[key] for m in all_metrics]
        result[key] = np.mean(values)
        result[f'{key}_std'] = np.std(values)

    # Copy other metrics from first run
    for key in all_metrics[0]:
        if key not in result:
            result[key] = all_metrics[0][key]

    return result


def main():
    if len(sys.argv) < 2:
        output_dir = "/tmp/llama_sched_test"
    else:
        output_dir = sys.argv[1]

    output_dir = Path(output_dir)

    print("=" * 100)
    print("CPU Scheduler Overhead Analysis - llama-server")
    print("=" * 100)
    print()

    # Collect results
    notrace_results = []
    trace_results = []

    for name, prefix in SCENARIOS:
        # No-trace results (actual performance)
        result = collect_multi_run_results(output_dir, prefix, "notrace")
        if result:
            result['scenario'] = name
            notrace_results.append(result)

        # Trace results (scheduler metrics)
        result = collect_multi_run_results(output_dir, prefix, "trace")
        trace_log = output_dir / f"{prefix}_trace.log"
        trace_csv = output_dir / f"{prefix}_trace.csv"

        if result:
            result['scenario'] = name
            log_stats = parse_trace_log(trace_log) if trace_log.exists() else {}
            csv_stats = parse_trace_csv(trace_csv) if trace_csv.exists() else {}

            # Calculate normalized metrics
            launches = log_stats.get('launches', 0)
            sched_per_1k = (log_stats.get('sched_switches', 0) / launches * 1000) if launches > 0 else 0
            soft_irq_per_1k = (log_stats.get('soft_irqs', 0) / launches * 1000) if launches > 0 else 0
            hard_irq_per_1k = (log_stats.get('hard_irqs', 0) / launches * 1000) if launches > 0 else 0

            result.update(log_stats)
            result.update(csv_stats)
            result['sched_per_1k'] = sched_per_1k
            result['soft_irq_per_1k'] = soft_irq_per_1k
            result['hard_irq_per_1k'] = hard_irq_per_1k

            trace_results.append(result)

    # ==========================================================================
    # Table 1: Performance Results (Without Tracing)
    # ==========================================================================
    num_runs = notrace_results[0].get('num_runs', 1) if notrace_results else 1
    print(f"Table 1: Performance Results (Without Tracing) - {num_runs} runs averaged")
    print("-" * 120)
    print(f"{'Scenario':<20} {'tok/s':>14} {'TPOT Mean':>14} {'TPOT P99':>12} {'TTFT Mean':>14} {'TTFT P99':>12} {'Slowdown':>10}")
    print("-" * 120)

    baseline_toks = notrace_results[0]['output_token_throughput'] if notrace_results else 0

    for r in notrace_results:
        slowdown = ((baseline_toks - r['output_token_throughput']) / baseline_toks * 100) if baseline_toks > 0 else 0
        slowdown_str = f"{slowdown:.1f}%" if r['scenario'] != "Baseline" else "-"

        # Show ± std if available
        tok_std = r.get('output_token_throughput_std', 0)
        tpot_std = r.get('mean_tpot_std', 0)
        ttft_std = r.get('mean_ttft_std', 0)

        tok_str = f"{r['output_token_throughput']:.1f}±{tok_std:.1f}" if tok_std > 0 else f"{r['output_token_throughput']:.2f}"
        tpot_str = f"{r['mean_tpot']:.2f}±{tpot_std:.2f}" if tpot_std > 0 else f"{r['mean_tpot']:.2f}"
        ttft_str = f"{r['mean_ttft']:.1f}±{ttft_std:.1f}" if ttft_std > 0 else f"{r['mean_ttft']:.2f}"

        print(f"{r['scenario']:<20} {tok_str:>14} "
              f"{tpot_str:>12}ms {r['p99_tpot']:>10.2f}ms "
              f"{ttft_str:>12}ms {r['p99_ttft']:>10.2f}ms "
              f"{slowdown_str:>10}")

    print()

    # ==========================================================================
    # Table 2: Scheduler Metrics (With Tracing)
    # ==========================================================================
    print("Table 2: Scheduler Metrics (With Tracing)")
    print("-" * 100)
    print(f"{'Scenario':<20} {'Launches':>10} {'Sched/1K':>10} {'SoftIRQ/1K':>12} {'HardIRQ/1K':>12} {'IRQ Time':>12}")
    print("-" * 100)

    for r in trace_results:
        print(f"{r['scenario']:<20} {r.get('launches', 0):>10,} "
              f"{r.get('sched_per_1k', 0):>10.1f} "
              f"{r.get('soft_irq_per_1k', 0):>12.1f} "
              f"{r.get('hard_irq_per_1k', 0):>12.1f} "
              f"{r.get('irq_time_ms', 0):>10.2f}ms")

    print()

    # ==========================================================================
    # Table 3: Tracer Overhead
    # ==========================================================================
    print("Table 3: Tracer Overhead")
    print("-" * 100)
    print(f"{'Scenario':<20} {'No Trace tok/s':>15} {'With Trace tok/s':>18} {'Overhead':>12}")
    print("-" * 100)

    for nt in notrace_results:
        tr = next((t for t in trace_results if t['prefix'] == nt['prefix']), None)
        if tr:
            nt_toks = nt['output_token_throughput']
            tr_toks = tr['output_token_throughput']
            overhead = ((nt_toks - tr_toks) / nt_toks * 100) if nt_toks > 0 else 0
            print(f"{nt['scenario']:<20} {nt_toks:>15.2f} {tr_toks:>18.2f} {overhead:>10.1f}%")

    print()

    # ==========================================================================
    # Key Findings
    # ==========================================================================
    print("=" * 100)
    print("Key Findings")
    print("=" * 100)
    print()

    if len(notrace_results) >= 6 and len(trace_results) >= 6:
        baseline_nt = notrace_results[0]
        cpu_stress_nt = notrace_results[1]
        heavy_load_nt = notrace_results[4]
        cpu_pinned_nt = notrace_results[5]

        baseline_tr = trace_results[0]
        cpu_stress_tr = trace_results[1]
        heavy_load_tr = trace_results[4]
        cpu_pinned_tr = trace_results[5]

        # RQ1: Clean environment impact
        print("RQ1: CPU Scheduler Impact in Clean Environment")
        print(f"  - Scheduler Impact: {baseline_tr.get('sched_per_1k', 0):.1f} context switches per 1K launches")
        print(f"  - IRQ Time: {baseline_tr.get('irq_time_ms', 0):.2f}ms total")
        print()

        # RQ3: Noisy neighbor impact
        print("RQ3: Noisy Neighbor Impact")
        for r in notrace_results[1:5]:
            slowdown = ((baseline_nt['output_token_throughput'] - r['output_token_throughput']) / baseline_nt['output_token_throughput'] * 100) if baseline_nt['output_token_throughput'] > 0 else 0
            print(f"  - {r['scenario']}: {slowdown:.1f}% slowdown")
        print()

        # Context switch comparison
        print("Context Switch Comparison (Sched/1K):")
        print(f"  - Baseline:    {baseline_tr.get('sched_per_1k', 0):.1f}")
        print(f"  - CPU Stress:  {cpu_stress_tr.get('sched_per_1k', 0):.1f} ({cpu_stress_tr.get('sched_per_1k', 0) / max(baseline_tr.get('sched_per_1k', 1), 1):.1f}x increase)")
        print(f"  - Heavy Load:  {heavy_load_tr.get('sched_per_1k', 0):.1f} ({heavy_load_tr.get('sched_per_1k', 0) / max(baseline_tr.get('sched_per_1k', 1), 1):.1f}x increase)")
        print()

        # RQ4: CPU pinning effectiveness
        print("RQ4: CPU Pinning Effectiveness")
        if cpu_stress_tr.get('sched_per_1k', 0) > 0:
            reduction = (1 - cpu_pinned_tr.get('sched_per_1k', 0) / cpu_stress_tr.get('sched_per_1k', 0)) * 100
            print(f"  - Context switch reduction: {reduction:.1f}%")
        perf_recovery = ((cpu_pinned_nt['output_token_throughput'] - cpu_stress_nt['output_token_throughput']) / cpu_stress_nt['output_token_throughput'] * 100) if cpu_stress_nt['output_token_throughput'] > 0 else 0
        print(f"  - Performance recovery: {perf_recovery:+.1f}%")
        print()

        # Heavy Load IRQ details
        if heavy_load_tr.get('softirq_details'):
            print("Heavy Load Soft IRQ Breakdown:")
            for irq_name, stats in sorted(heavy_load_tr['softirq_details'].items(),
                                          key=lambda x: x[1]['total_time_us'], reverse=True)[:5]:
                count = stats['count']
                total_us = stats['total_time_us']
                avg_us = total_us / count if count > 0 else 0
                print(f"  - {irq_name:<12}: {count:>5} events, {total_us:>10.1f}us total, {avg_us:>6.1f}us avg")
            print()

    print("=" * 100)
    print(f"Full results saved to: {output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
