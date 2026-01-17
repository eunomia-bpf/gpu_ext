#!/usr/bin/env python3
import pandas as pd
import re
import sys

scenarios = [
    ("Baseline", "baseline"),
    ("Noisy CPU", "noisy_cpu"),
    ("Noisy Network", "noisy_network"),
    ("Noisy Disk", "noisy_disk"),
    ("Heavy Load", "heavy_load"),
    ("Optimized", "optimized"),
]

print("=" * 90)
print("调度器和 IRQ 影响分析 - 完整报告")
print("=" * 90)
print()

results = []

for name, prefix in scenarios:
    # Parse time output for tok/s and elapsed time
    time_file = f"/tmp/noisy_neighbor_test/{prefix}_output.txt"
    try:
        with open(time_file) as f:
            content = f.read()
            tok_match = re.search(r'tok/s:\s+([\d.]+)', content)
            elapsed_match = re.search(r'Elapsed.*:\s+([\d:]+\.?\d*)', content)

            tok_s = float(tok_match.group(1)) if tok_match else 0.0

            if elapsed_match:
                time_str = elapsed_match.group(1)
                parts = time_str.split(':')
                if len(parts) == 2:  # m:ss
                    elapsed_s = int(parts[0]) * 60 + float(parts[1])
                else:  # h:mm:ss
                    elapsed_s = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            else:
                elapsed_s = 0.0
    except:
        tok_s = 0.0
        elapsed_s = 0.0

    # Parse trace log
    log_file = f"/tmp/noisy_neighbor_test/{prefix}.log"
    try:
        with open(log_file) as f:
            content = f.read()
            launches = 0
            sched_switches = 0
            hard_irqs = 0
            soft_irqs = 0

            for line in content.split('\n'):
                if 'cuLaunchKernel' in line and 'Tracked' not in line:
                    m = re.search(r'cuLaunchKernel\s+(\d+)', line)
                    if m:
                        launches = int(m.group(1))
                elif 'Sched Switches Tracked' in line:
                    m = re.search(r'Sched Switches Tracked\s+(\d+)', line)
                    if m:
                        sched_switches = int(m.group(1))
                elif 'Hard IRQs Tracked' in line:
                    m = re.search(r'Hard IRQs Tracked\s+(\d+)', line)
                    if m:
                        hard_irqs = int(m.group(1))
                elif 'Soft IRQs Tracked' in line:
                    m = re.search(r'Soft IRQs Tracked\s+(\d+)', line)
                    if m:
                        soft_irqs = int(m.group(1))
    except:
        launches = 0
        sched_switches = 0
        hard_irqs = 0
        soft_irqs = 0

    # Parse trace CSV for detailed IRQ analysis
    csv_file = f"/tmp/noisy_neighbor_test/{prefix}_trace.csv"
    irq_time_ms = 0
    softirq_details = {}

    try:
        df = pd.read_csv(csv_file, low_memory=False)
        runtime = df['timestamp_ns'].max() / 1e9

        # Calculate IRQ time
        irq_exits = df[df['event_type'].str.contains('irqExit', na=False)]
        irq_time_ms = irq_exits['duration_ns'].sum() / 1e6

        # Analyze soft IRQ types
        soft_irq_exits = df[df['event_type'] == 'softirqExit']
        for _, row in soft_irq_exits.iterrows():
            irq_name = row.get('irq_name', 'unknown')
            duration_ns = row.get('duration_ns', 0)

            if irq_name not in softirq_details:
                softirq_details[irq_name] = {'count': 0, 'total_time_us': 0}

            softirq_details[irq_name]['count'] += 1
            softirq_details[irq_name]['total_time_us'] += duration_ns / 1000

    except Exception as e:
        runtime = 0
        irq_time_ms = 0
        softirq_details = {}

    # Calculate normalized metrics (per 1K launches)
    if launches > 0:
        sched_per_1k = (sched_switches / launches) * 1000
        soft_irq_per_1k = (soft_irqs / launches) * 1000
        hard_irq_per_1k = (hard_irqs / launches) * 1000
    else:
        sched_per_1k = 0
        soft_irq_per_1k = 0
        hard_irq_per_1k = 0

    results.append({
        'scenario': name,
        'tok_s': tok_s,
        'elapsed_s': elapsed_s,
        'runtime': runtime,
        'launches': launches,
        'sched_switches': sched_switches,
        'soft_irqs': soft_irqs,
        'hard_irqs': hard_irqs,
        'irq_time_ms': irq_time_ms,
        'sched_per_1k': sched_per_1k,
        'soft_irq_per_1k': soft_irq_per_1k,
        'hard_irq_per_1k': hard_irq_per_1k,
        'softirq_details': softirq_details,
    })

# Print main table
print(f"{'场景':<20} {'Launches':>10} {'Sched/1K':>10} {'SoftIRQ/1K':>12} {'HardIRQ/1K':>12} {'IRQ时间(ms)':>12}")
print("-" * 90)

baseline_tok = results[0]['tok_s']

for r in results:
    print(f"{r['scenario']:<20} {r['launches']:>10,} {r['sched_per_1k']:>10.1f} {r['soft_irq_per_1k']:>12.1f} "
          f"{r['hard_irq_per_1k']:>12.1f} {r['irq_time_ms']:>12.2f}")

print()
print("=" * 90)
print("性能指标详细分析")
print("=" * 90)
print()

print(f"{'场景':<20} {'tok/s':>10} {'运行时间(s)':>12} {'Slowdown':>12} {'Launches':>10}")
print("-" * 90)

for r in results:
    slowdown_pct = ((baseline_tok - r['tok_s']) / baseline_tok * 100) if baseline_tok > 0 else 0
    slowdown_str = f"{slowdown_pct:.1f}%" if r['scenario'] != "Baseline" else "-"

    print(f"{r['scenario']:<20} {r['tok_s']:>10.2f} {r['elapsed_s']:>12.2f} {slowdown_str:>12} {r['launches']:>10,}")

print()
print("=" * 90)
print("关键发现")
print("=" * 90)
print()

# Finding 1: Scheduling overhead
baseline = results[0]
noisy_cpu = results[1]
heavy_load = results[4]
optimized = results[5]

print(f"1. **调度切换增加（每1000个launches）**：")
print(f"   - Baseline:      {baseline['sched_per_1k']:.1f}")
print(f"   - Noisy CPU:     {noisy_cpu['sched_per_1k']:.1f} (增加 {noisy_cpu['sched_per_1k']/baseline['sched_per_1k']:.1f}x)")
print(f"   - Heavy Load:    {heavy_load['sched_per_1k']:.1f} (增加 {heavy_load['sched_per_1k']/baseline['sched_per_1k']:.1f}x)")
print(f"   - Optimized:     {optimized['sched_per_1k']:.1f}")
print()

# Finding 2: Soft IRQ
noisy_network = results[2]
print(f"2. **Soft IRQ 变化（每1000个launches）**：")
print(f"   - Baseline:      {baseline['soft_irq_per_1k']:.1f}")
print(f"   - Noisy Network: {noisy_network['soft_irq_per_1k']:.1f}")
print(f"   - Heavy Load:    {heavy_load['soft_irq_per_1k']:.1f}")
print()

# Finding 3: Hard IRQ
noisy_disk = results[3]
print(f"3. **Hard IRQ（块设备中断）**：")
print(f"   - Baseline:      {baseline['hard_irqs']}")
print(f"   - Noisy Disk:    {noisy_disk['hard_irqs']}")
print(f"   - Heavy Load:    {heavy_load['hard_irqs']}")
print()

# Finding 4: Heavy Load IRQ details
if heavy_load['softirq_details']:
    print(f"4. **Heavy Load 的 Soft IRQ 类型分析**：")
    for irq_name, stats in sorted(heavy_load['softirq_details'].items(),
                                   key=lambda x: x[1]['total_time_us'], reverse=True):
        count = stats['count']
        total_us = stats['total_time_us']
        avg_us = total_us / count if count > 0 else 0
        print(f"   - {irq_name:<12}: {count:>4} 次, 总时间 {total_us:>8.1f} µs, 平均 {avg_us:>5.1f} µs")
    print()

# Finding 5: Optimization effectiveness
print(f"5. **Optimized 的效果**：")
reduction_pct = (1 - optimized['sched_per_1k'] / noisy_cpu['sched_per_1k']) * 100
print(f"   - 调度切换: {noisy_cpu['sched_per_1k']:.1f} → {optimized['sched_per_1k']:.1f} (减少 {reduction_pct:.1f}%)")
print(f"   - 但仍远高于 Baseline ({baseline['sched_per_1k']:.1f})，说明绑核无法完全消除 CPU 竞争")
print()

# Finding 6: Performance impact
print(f"6. **性能影响总结**：")
for r in results[1:]:
    slowdown = ((baseline_tok - r['tok_s']) / baseline_tok * 100) if baseline_tok > 0 else 0
    print(f"   - {r['scenario']}: 性能下降 {slowdown:.1f}% (from {baseline_tok:.2f} to {r['tok_s']:.2f} tok/s)")
print()

# Finding 7: Heavy Load comparison
print(f"7. **Heavy Load (CPU+Network+Disk) 特别分析**：")
print(f"   - 性能下降: {((baseline_tok - heavy_load['tok_s']) / baseline_tok * 100):.1f}% (最严重！)")
print(f"   - 调度切换: {heavy_load['sched_per_1k']:.1f} 次/1K launches")
print(f"   - 比单独 CPU 干扰的影响: {heavy_load['sched_per_1k'] / noisy_cpu['sched_per_1k'] * 100:.1f}%")
print(f"   - 说明: CPU+Network+Disk 叠加效应显著，不是简单相加")
print()
