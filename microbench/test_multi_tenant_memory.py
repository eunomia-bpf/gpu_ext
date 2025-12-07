#!/usr/bin/env python3
"""
Test PID-based eviction policies.
Compares performance of high-priority vs low-priority processes.

Usage:
  python3 test_pid_lfu.py --policy eviction_pid_quota -P 80 -L 20
  python3 test_pid_lfu.py --policy eviction_freq_pid_decay -P 1 -L 10
"""

import subprocess
import time
import os
import sys
import re

BASE_PATH = "/home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/src"
UVMBENCH = "/home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/microbench/memory/uvmbench"
CLEANUP_TOOL = f"{BASE_PATH}/cleanup_struct_ops_tool"

def get_available_policies():
    """Scan BASE_PATH for available eviction policy executables."""
    policies = []
    # Look for eviction_* executables (not .c, .o, .h files)
    for f in os.listdir(BASE_PATH):
        path = os.path.join(BASE_PATH, f)
        if (f.startswith('eviction_') and
            os.path.isfile(path) and
            os.access(path, os.X_OK) and
            not f.endswith(('.c', '.o', '.h', '.bpf.c'))):
            policies.append(f)
    return sorted(policies)

def cleanup():
    """Kill existing processes and detach struct_ops."""
    # Kill eviction policy processes (but not this script)
    # Use full path match to avoid killing python scripts
    for policy in get_available_policies():
        subprocess.run(["sudo", "pkill", "-f", f"{BASE_PATH}/{policy}"], capture_output=True)
    subprocess.run(["pkill", "-9", "uvmbench"], capture_output=True)
    # Run cleanup tool to detach any remaining struct_ops
    subprocess.run(["sudo", CLEANUP_TOOL], capture_output=True)
    time.sleep(1)

def parse_uvmbench_output(output):
    """Extract median time and bandwidth from uvmbench output."""
    median_match = re.search(r'Median time:\s+([\d.]+)\s+ms', output)
    bw_match = re.search(r'Bandwidth:\s+([\d.]+)\s+GB/s', output)
    median = float(median_match.group(1)) if median_match else None
    bw = float(bw_match.group(1)) if bw_match else None
    return median, bw

def run_baseline(kernel='rand_stream', size_factor=0.6, iterations=3, stride_bytes=4096, round_num=None):
    """Run baseline test without policy."""
    print("\n" + "="*60)
    if round_num is not None:
        print(f"BASELINE TEST (No Policy) - Round {round_num}")
    else:
        print("BASELINE TEST (No Policy)")
    print("="*60)

    cleanup()

    print("Starting two uvmbench processes...")
    start_time = time.time()

    uvm1 = subprocess.Popen(
        [UVMBENCH, f"--size_factor={size_factor}", "--mode=uvm", f"--iterations={iterations}",
         f"--kernel={kernel}", f"--stride_bytes={stride_bytes}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    uvm2 = subprocess.Popen(
        [UVMBENCH, f"--size_factor={size_factor}", "--mode=uvm", f"--iterations={iterations}",
         f"--kernel={kernel}", f"--stride_bytes={stride_bytes}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    print(f"  Process A (PID {uvm1.pid})")
    print(f"  Process B (PID {uvm2.pid})")

    out1, _ = uvm1.communicate()
    out2, _ = uvm2.communicate()
    end_time = time.time()

    out1_str = out1.decode()
    out2_str = out2.decode()

    median1, bw1 = parse_uvmbench_output(out1_str)
    median2, bw2 = parse_uvmbench_output(out2_str)

    print(f"\nProcess A: Median={median1:.2f}ms, BW={bw1:.2f}GB/s")
    print(f"Process B: Median={median2:.2f}ms, BW={bw2:.2f}GB/s")
    print(f"Total time: {end_time - start_time:.2f}s")

    return {
        'a_median': median1, 'a_bw': bw1,
        'b_median': median2, 'b_bw': bw2,
        'total_time': end_time - start_time
    }

def run_with_policy(policy_binary, high_param, low_param, kernel='rand_stream',
                    size_factor=0.6, iterations=3, stride_bytes=4096, round_num=None):
    """Run test with specified eviction policy."""
    policy_path = os.path.join(BASE_PATH, policy_binary)
    if not os.path.exists(policy_path):
        raise ValueError(f"Policy binary not found: {policy_path}")

    print("\n" + "="*60)
    if round_num is not None:
        print(f"POLICY TEST ({policy_binary}) - Round {round_num}")
    else:
        print(f"POLICY TEST ({policy_binary})")
    print("="*60)

    cleanup()

    # Start uvmbench processes first to get PIDs
    print("Starting uvmbench processes...")
    uvm_high = subprocess.Popen(
        [UVMBENCH, f"--size_factor={size_factor}", "--mode=uvm", f"--iterations={iterations}",
         f"--kernel={kernel}", f"--stride_bytes={stride_bytes}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    uvm_low = subprocess.Popen(
        [UVMBENCH, f"--size_factor={size_factor}", "--mode=uvm", f"--iterations={iterations}",
         f"--kernel={kernel}", f"--stride_bytes={stride_bytes}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    high_pid = uvm_high.pid
    low_pid = uvm_low.pid
    print(f"  High priority: PID {high_pid}")
    print(f"  Low priority:  PID {low_pid}")

    # Start policy with correct PIDs
    print(f"Starting {policy_binary}...")
    print(f"  High priority param (-P): {high_param}")
    print(f"  Low priority param (-L):  {low_param}")

    # Use a temp file to capture policy output reliably
    import tempfile
    policy_output_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt')
    policy_output_path = policy_output_file.name
    policy_output_file.close()

    # Run policy with output redirected to file via shell
    policy_cmd = f"sudo {policy_path} -p {high_pid} -P {high_param} -l {low_pid} -L {low_param} > {policy_output_path} 2>&1"
    policy_proc = subprocess.Popen(policy_cmd, shell=True)

    start_time = time.time()

    # Wait for uvmbench to complete
    out_high, _ = uvm_high.communicate()
    out_low, _ = uvm_low.communicate()
    end_time = time.time()

    # Give policy a moment to print final stats
    time.sleep(2)

    # Stop policy gracefully with SIGINT to trigger final stats printout
    subprocess.run(["sudo", "pkill", "-INT", "-f", policy_path], capture_output=True)

    # Wait for policy to exit and write final output
    try:
        policy_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        policy_proc.kill()

    time.sleep(1)

    # Read policy output from file
    try:
        with open(policy_output_path, 'r') as f:
            policy_output = f.read()
    finally:
        os.unlink(policy_output_path)

    out_high_str = out_high.decode()
    out_low_str = out_low.decode()

    median_high, bw_high = parse_uvmbench_output(out_high_str)
    median_low, bw_low = parse_uvmbench_output(out_low_str)

    print(f"\nHigh priority (PID {high_pid}): Median={median_high:.2f}ms, BW={bw_high:.2f}GB/s")
    print(f"Low priority  (PID {low_pid}):  Median={median_low:.2f}ms, BW={bw_low:.2f}GB/s")
    print(f"Total time: {end_time - start_time:.2f}s")

    # Parse and print policy statistics
    print("\n" + "-"*40)
    print("Policy Statistics:")
    print("-"*40)

    # Parse per-PID statistics from the final output block
    def parse_pid_stats(output, pid):
        """Extract stats for a specific PID from policy output."""
        stats = {}
        # Find the section for this PID in the last statistics block
        pattern = rf'(?:High|Low) priority PID {pid}:\s*\n((?:\s+.*\n)*)'
        matches = list(re.finditer(pattern, output))
        if not matches:
            return stats
        # Use the last match (final statistics)
        section = matches[-1].group(1)

        # Parse individual stats
        current_match = re.search(r'Current active chunks:\s+(\d+)', section)
        if current_match:
            stats['current_count'] = int(current_match.group(1))

        activate_match = re.search(r'Total activated:\s+(\d+)', section)
        if activate_match:
            stats['total_activate'] = int(activate_match.group(1))

        used_match = re.search(r'Total used calls:\s+(\d+)', section)
        if used_match:
            stats['total_used'] = int(used_match.group(1))

        # Policy allow (handle both "moved" and "saved" variants)
        allow_match = re.search(r'Policy allow \([^)]+\):\s+(\d+)', section)
        if allow_match:
            stats['policy_allow'] = int(allow_match.group(1))

        # Policy deny (handle both "not moved" and "evicted" variants)
        deny_match = re.search(r'Policy deny \([^)]+\):\s+(\d+)', section)
        if deny_match:
            stats['policy_deny'] = int(deny_match.group(1))

        return stats

    high_stats = parse_pid_stats(policy_output, high_pid)
    low_stats = parse_pid_stats(policy_output, low_pid)

    if high_stats or low_stats:
        print(f"\n  High priority PID {high_pid}:")
        if high_stats:
            print(f"    Current active chunks: {high_stats.get('current_count', 0)}")
            print(f"    Total activated: {high_stats.get('total_activate', 0)}")
            total = high_stats.get('policy_allow', 0) + high_stats.get('policy_deny', 0)
            if total > 0:
                print(f"    Policy allow (moved): {high_stats.get('policy_allow', 0)} ({100*high_stats.get('policy_allow', 0)/total:.1f}%)")
                print(f"    Policy deny (not moved): {high_stats.get('policy_deny', 0)} ({100*high_stats.get('policy_deny', 0)/total:.1f}%)")
        else:
            print("    (No data)")

        print(f"\n  Low priority PID {low_pid}:")
        if low_stats:
            print(f"    Current active chunks: {low_stats.get('current_count', 0)}")
            print(f"    Total activated: {low_stats.get('total_activate', 0)}")
            total = low_stats.get('policy_allow', 0) + low_stats.get('policy_deny', 0)
            if total > 0:
                print(f"    Policy allow (moved): {low_stats.get('policy_allow', 0)} ({100*low_stats.get('policy_allow', 0)/total:.1f}%)")
                print(f"    Policy deny (not moved): {low_stats.get('policy_deny', 0)} ({100*low_stats.get('policy_deny', 0)/total:.1f}%)")
        else:
            print("    (No data)")
    else:
        print("  (No statistics captured)")
        print(f"  Raw output: {policy_output[:500]}")

    # Combine stats for return value
    final_stats = {
        'high': high_stats,
        'low': low_stats
    }

    return {
        'high_median': median_high, 'high_bw': bw_high,
        'low_median': median_low, 'low_bw': bw_low,
        'total_time': end_time - start_time,
        'policy_stats': final_stats
    }

def run_warmup(kernel='rand_stream', size_factor=0.6, iterations=3, stride_bytes=4096):
    """Run a complete baseline as warmup to initialize GPU context."""
    print("\n" + "="*60)
    print("WARMUP (running complete baseline to initialize GPU)")
    print("="*60)

    cleanup()

    print("Running two uvmbench processes for warmup...")
    uvm1 = subprocess.Popen(
        [UVMBENCH, f"--size_factor={size_factor}", "--mode=uvm", f"--iterations={iterations}",
         f"--kernel={kernel}", f"--stride_bytes={stride_bytes}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    uvm2 = subprocess.Popen(
        [UVMBENCH, f"--size_factor={size_factor}", "--mode=uvm", f"--iterations={iterations}",
         f"--kernel={kernel}", f"--stride_bytes={stride_bytes}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    print(f"  Process A (PID {uvm1.pid})")
    print(f"  Process B (PID {uvm2.pid})")
    uvm1.communicate()
    uvm2.communicate()
    print("Warmup complete.")
    time.sleep(2)

def main():
    import argparse

    # Get available policies dynamically
    available_policies = get_available_policies()

    parser = argparse.ArgumentParser(
        description='Test PID-based eviction policies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available policies (from {BASE_PATH}):
  {chr(10).join(f'  {p}' for p in available_policies) if available_policies else '  (none found - run make first)'}

Examples:
  # Test quota-based policy (high=80%, low=20%)
  %(prog)s --policy eviction_pid_quota -P 80 -L 20

  # Test frequency decay policy (high=every access, low=every 10 accesses)
  %(prog)s --policy eviction_freq_pid_decay -P 1 -L 10

  # Run multiple rounds with baseline comparison
  %(prog)s --policy eviction_pid_quota -P 80 -L 20 -r 3
""")
    parser.add_argument('--policy', type=str, default='eviction_pid_quota',
                        help=f'Eviction policy executable name (default: eviction_pid_quota)')
    parser.add_argument('-P', '--high-param', type=int, default=80,
                        help='High priority parameter (default: 80)')
    parser.add_argument('-L', '--low-param', type=int, default=20,
                        help='Low priority parameter (default: 20)')
    parser.add_argument('-k', '--kernel', type=str, default='rand_stream',
                        choices=['seq_stream', 'rand_stream', 'pointer_chase'],
                        help='Access pattern kernel (default: rand_stream)')
    parser.add_argument('-s', '--size-factor', type=float, default=0.6,
                        help='Size factor for uvmbench (default: 0.6)')
    parser.add_argument('-i', '--iterations', type=int, default=3,
                        help='Number of iterations per test (default: 3)')
    parser.add_argument('--stride-bytes', type=int, default=4096,
                        help='Access stride in bytes (default: 4096)')
    parser.add_argument('-r', '--rounds', type=int, default=1,
                        help='Number of rounds to run (interleaved baseline/policy) (default: 1)')
    parser.add_argument('--baseline-no-policy', action='store_true',
                        help='Run baseline without any policy (default: baseline uses equal params)')
    parser.add_argument('--list-policies', action='store_true',
                        help='List available policies and exit')
    args = parser.parse_args()

    # List policies and exit if requested
    if args.list_policies:
        print(f"Available policies in {BASE_PATH}:")
        for p in available_policies:
            print(f"  {p}")
        if not available_policies:
            print("  (none found - run make first)")
        return 0

    policy = args.policy
    high_param = args.high_param
    low_param = args.low_param
    size_factor = args.size_factor
    iterations = args.iterations
    stride_bytes = args.stride_bytes
    rounds = args.rounds
    baseline_no_policy = args.baseline_no_policy

    # Validate policy exists
    policy_path = os.path.join(BASE_PATH, policy)
    if not os.path.exists(policy_path):
        print(f"Error: Policy '{policy}' not found at {policy_path}")
        print(f"\nAvailable policies:")
        for p in available_policies:
            print(f"  {p}")
        return 1

    print("="*60)
    print(f"PID-BASED EVICTION POLICY EXPERIMENT")
    print("="*60)
    print(f"Policy: {policy}")
    print(f"Parameters:")
    print(f"  High priority param (-P): {high_param}")
    print(f"  Low priority param (-L):  {low_param}")
    print(f"  Access pattern:           {args.kernel}")
    print(f"  Size factor:              {size_factor}")
    print(f"  Iterations:               {iterations}")
    print(f"  Stride bytes:             {stride_bytes}")
    print(f"  Rounds:                   {rounds}")
    print(f"  Baseline mode:            {'no policy' if baseline_no_policy else 'equal params'}")

    if os.geteuid() != 0:
        print("WARNING: Run with sudo for full functionality.")

    # Warmup first
    print("\n[Phase 0] Warmup...")
    run_warmup(kernel=args.kernel, size_factor=size_factor, iterations=iterations, stride_bytes=stride_bytes)

    # Run interleaved rounds: baseline, policy, baseline, policy, ...
    baseline_results = []
    policy_results = []

    # Baseline uses equal params
    baseline_param = (high_param + low_param) // 2

    for r in range(rounds):
        print(f"\n{'#'*60}")
        print(f"# ROUND {r+1}/{rounds}")
        print(f"{'#'*60}")

        # Run baseline
        if baseline_no_policy:
            baseline = run_baseline(kernel=args.kernel, size_factor=size_factor,
                                    iterations=iterations, stride_bytes=stride_bytes, round_num=r+1)
        else:
            # Run baseline with equal params
            print(f"\n[Baseline with equal params: {baseline_param}-{baseline_param}]")
            baseline_policy = run_with_policy(policy, baseline_param, baseline_param,
                                              kernel=args.kernel, size_factor=size_factor, iterations=iterations,
                                              stride_bytes=stride_bytes, round_num=r+1)
            # Convert policy result format to baseline format
            baseline = {
                'a_median': baseline_policy['high_median'],
                'a_bw': baseline_policy['high_bw'],
                'b_median': baseline_policy['low_median'],
                'b_bw': baseline_policy['low_bw'],
                'total_time': baseline_policy['total_time']
            }
        baseline_results.append(baseline)

        # Run with policy
        print(f"\n[Policy with {high_param}-{low_param} params]")
        policy_result = run_with_policy(policy, high_param, low_param,
                                        kernel=args.kernel, size_factor=size_factor, iterations=iterations,
                                        stride_bytes=stride_bytes, round_num=r+1)
        policy_results.append(policy_result)

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)

    # Calculate averages across all rounds
    avg_baseline_a = sum(r['a_median'] for r in baseline_results) / len(baseline_results)
    avg_baseline_b = sum(r['b_median'] for r in baseline_results) / len(baseline_results)
    avg_baseline_a_bw = sum(r['a_bw'] for r in baseline_results) / len(baseline_results)
    avg_baseline_b_bw = sum(r['b_bw'] for r in baseline_results) / len(baseline_results)

    avg_policy_high = sum(r['high_median'] for r in policy_results) / len(policy_results)
    avg_policy_low = sum(r['low_median'] for r in policy_results) / len(policy_results)
    avg_policy_high_bw = sum(r['high_bw'] for r in policy_results) / len(policy_results)
    avg_policy_low_bw = sum(r['low_bw'] for r in policy_results) / len(policy_results)

    baseline_desc = "no policy" if baseline_no_policy else f"equal params ({baseline_param}-{baseline_param})"
    print(f"\nBaseline ({baseline_desc}) - avg of {rounds} rounds:")
    print(f"  Process A: {avg_baseline_a:.2f}ms, {avg_baseline_a_bw:.2f}GB/s")
    print(f"  Process B: {avg_baseline_b:.2f}ms, {avg_baseline_b_bw:.2f}GB/s")

    print(f"\nWith {policy} (high={high_param}, low={low_param}) - avg of {rounds} rounds:")
    print(f"  High priority (param={high_param}): {avg_policy_high:.2f}ms, {avg_policy_high_bw:.2f}GB/s")
    print(f"  Low priority (param={low_param}):  {avg_policy_low:.2f}ms, {avg_policy_low_bw:.2f}GB/s")

    # Print per-round details
    if rounds > 1:
        print(f"\nPer-round details:")
        print(f"  {'Round':<6} {'Baseline A':<12} {'Baseline B':<12} {'Policy High':<12} {'Policy Low':<12}")
        for r in range(rounds):
            print(f"  {r+1:<6} {baseline_results[r]['a_median']:<12.2f} {baseline_results[r]['b_median']:<12.2f} "
                  f"{policy_results[r]['high_median']:<12.2f} {policy_results[r]['low_median']:<12.2f}")

    # Calculate speedup
    avg_baseline = (avg_baseline_a + avg_baseline_b) / 2
    if avg_baseline and avg_policy_high:
        high_speedup = avg_baseline / avg_policy_high
        low_speedup = avg_baseline / avg_policy_low
        print(f"\nSpeedup vs baseline avg:")
        print(f"  High priority: {high_speedup:.2f}x ({avg_policy_high:.1f}ms vs {avg_baseline:.1f}ms)")
        print(f"  Low priority:  {low_speedup:.2f}x ({avg_policy_low:.1f}ms vs {avg_baseline:.1f}ms)")

        # Compare high vs low priority
        if avg_policy_high > 0 and avg_policy_low > 0:
            priority_ratio = avg_policy_low / avg_policy_high
            print(f"\nPriority differentiation:")
            print(f"  High/Low ratio: {1/priority_ratio:.2f}x (high priority is {'faster' if priority_ratio > 1 else 'slower'})")

    cleanup()
    return 0

if __name__ == "__main__":
    sys.exit(main())
