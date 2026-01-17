#!/usr/bin/env python3
"""
分析 launch pairs 之间的 context switch 影响
"""
import pandas as pd
import numpy as np
import sys

csv_file = sys.argv[1] if len(sys.argv) > 1 else '/tmp/qwen3_trace_clean.csv'

print(f"Loading {csv_file}...")
df = pd.read_csv(csv_file)

# 只看主进程 (最多 launches 的那个)
pid_counts = df[df['event_type'].str.contains('LaunchKernel', na=False)].groupby('pid').size()
main_pid = pid_counts.idxmax()
print(f"\nMain PID: {main_pid} ({pid_counts[main_pid]} launches)")

events = df[df['pid'] == main_pid].copy()
events = events.sort_values('timestamp_ns').reset_index(drop=True)

# 提取 launches
launches = events[events['event_type'].str.contains('LaunchKernel', na=False)].copy()
print(f"Total launches: {len(launches)}")

# 分析 launch pairs
pairs = []
for i in range(len(launches) - 1):
    launch_i = launches.iloc[i]
    launch_j = launches.iloc[i + 1]

    t_start = launch_i['timestamp_ns']
    t_end = launch_j['timestamp_ns']
    interval_us = (t_end - t_start) / 1000

    # 检查之间是否有 OFF-CPU
    switches_between = events[
        (events['timestamp_ns'] > t_start) &
        (events['timestamp_ns'] < t_end) &
        (events['event_type'] == 'schedSwitch')
    ]

    has_offcpu = False
    total_offcpu_us = 0
    num_switches = len(switches_between)

    if num_switches > 0:
        offcpu_start = None
        for _, sw in switches_between.iterrows():
            if sw['last_offcpu_ns'] > 0:  # Going OFF-CPU
                offcpu_start = sw['timestamp_ns']
                has_offcpu = True
            elif sw['last_oncpu_ns'] > 0 and offcpu_start is not None:  # Coming ON-CPU
                total_offcpu_us += (sw['timestamp_ns'] - offcpu_start) / 1000
                offcpu_start = None

    pairs.append({
        'interval_us': interval_us,
        'has_switch': has_offcpu,
        'num_switches': num_switches,
        'offcpu_us': total_offcpu_us
    })

pairs_df = pd.DataFrame(pairs)

# 分组统计
group_no_switch = pairs_df[~pairs_df['has_switch']]
group_with_switch = pairs_df[pairs_df['has_switch']]

print(f"\n=== Launch Pair Analysis ===")
print(f"Total pairs: {len(pairs_df)}")
print(f"Pairs without context switch: {len(group_no_switch)} ({len(group_no_switch)/len(pairs_df)*100:.1f}%)")
print(f"Pairs with context switch: {len(group_with_switch)} ({len(group_with_switch)/len(pairs_df)*100:.1f}%)")

print(f"\n=== Interval Distribution (without switch) ===")
intervals_no = group_no_switch['interval_us']
print(f"Count: {len(intervals_no)}")
print(f"P50:  {np.percentile(intervals_no, 50):.0f} µs")
print(f"P90:  {np.percentile(intervals_no, 90):.0f} µs")
print(f"P95:  {np.percentile(intervals_no, 95):.0f} µs")
print(f"P99:  {np.percentile(intervals_no, 99):.0f} µs")
print(f"Max:  {intervals_no.max():.0f} µs")
print(f"Mean: {intervals_no.mean():.0f} µs")
print(f"Std:  {intervals_no.std():.0f} µs")

if len(group_with_switch) > 0:
    print(f"\n=== Interval Distribution (with switch) ===")
    intervals_with = group_with_switch['interval_us']
    print(f"Count: {len(intervals_with)}")
    print(f"P50:  {np.percentile(intervals_with, 50):.0f} µs")
    print(f"P90:  {np.percentile(intervals_with, 90):.0f} µs")
    print(f"P95:  {np.percentile(intervals_with, 95):.0f} µs")
    print(f"P99:  {np.percentile(intervals_with, 99):.0f} µs")
    print(f"Max:  {intervals_with.max():.0f} µs")
    print(f"Mean: {intervals_with.mean():.0f} µs")
    print(f"Std:  {intervals_with.std():.0f} µs")

    print(f"\n=== Scheduler Impact ===")
    penalty_median = np.percentile(intervals_with, 50) - np.percentile(intervals_no, 50)
    penalty_p90 = np.percentile(intervals_with, 90) - np.percentile(intervals_no, 90)
    print(f"Preemption Penalty (Median): {penalty_median:.0f} µs ({penalty_median/1000:.2f} ms)")
    print(f"Preemption Penalty (P90):    {penalty_p90:.0f} µs ({penalty_p90/1000:.2f} ms)")

    # OFF-CPU 时间统计
    print(f"\n=== OFF-CPU Time During Preemption ===")
    offcpu_times = group_with_switch[group_with_switch['offcpu_us'] > 0]['offcpu_us']
    if len(offcpu_times) > 0:
        print(f"P50:  {np.percentile(offcpu_times, 50):.0f} µs")
        print(f"P90:  {np.percentile(offcpu_times, 90):.0f} µs")
        print(f"Max:  {offcpu_times.max():.0f} µs")

# Tail latency attribution
print(f"\n=== Tail Latency Attribution ===")
p95_threshold = np.percentile(pairs_df['interval_us'], 95)
p99_threshold = np.percentile(pairs_df['interval_us'], 99)

p95_pairs = pairs_df[pairs_df['interval_us'] >= p95_threshold]
p99_pairs = pairs_df[pairs_df['interval_us'] >= p99_threshold]

p95_with_switch = p95_pairs[p95_pairs['has_switch']]
p99_with_switch = p99_pairs[p99_pairs['has_switch']]

print(f"P95+ pairs: {len(p95_pairs)}, with switch: {len(p95_with_switch)} ({len(p95_with_switch)/len(p95_pairs)*100:.1f}%)")
print(f"P99+ pairs: {len(p99_pairs)}, with switch: {len(p99_with_switch)} ({len(p99_with_switch)/len(p99_pairs)*100:.1f}%)")

# Burst analysis
print(f"\n=== Burst Analysis ===")
burst_threshold_us = 100
bursts = []
current_burst = []

for i, pair in pairs_df.iterrows():
    if pair['interval_us'] < burst_threshold_us:
        current_burst.append(i)
    else:
        if len(current_burst) >= 2:  # 至少 3 个连续 launch
            bursts.append(current_burst)
        current_burst = []

if len(current_burst) >= 2:
    bursts.append(current_burst)

print(f"Number of bursts (3+ launches within {burst_threshold_us}µs): {len(bursts)}")
if len(bursts) > 0:
    burst_sizes = [len(b) + 1 for b in bursts]  # +1 because pairs count
    print(f"Burst size: min={min(burst_sizes)}, max={max(burst_sizes)}, avg={np.mean(burst_sizes):.1f}")

    # 检查 burst 内部被打断
    burst_preempted = 0
    for burst_indices in bursts:
        for idx in burst_indices:
            if pairs_df.iloc[idx]['has_switch']:
                burst_preempted += 1
                break
    print(f"Bursts with preemption: {burst_preempted}/{len(bursts)} ({burst_preempted/len(bursts)*100:.1f}%)")
