#!/bin/bash
# Noisy Neighbor 测试脚本
# 对比干净环境 vs 嘈杂环境 vs 优化后的性能

set -e

QWEN_DIR="/home/yunwei37/workspace/gpu/xpu-perf/test/qwen3.cu"
TRACE_TOOL="/home/yunwei37/workspace/gpu/co-processor-demo/gpu_ext_policy/tools/cuda_sched_trace"
OUTPUT_DIR="/tmp/noisy_neighbor_test"

mkdir -p "$OUTPUT_DIR"

echo "==================================================================="
echo "Noisy Neighbor 测试 - GPU 调度器影响分析"
echo "==================================================================="
echo ""

# 1. Baseline: 干净环境
echo "[1/6] 测试 Baseline (干净环境)..."
sudo "$TRACE_TOOL" > "$OUTPUT_DIR/baseline_trace.csv" 2> "$OUTPUT_DIR/baseline.log" &
TRACE_PID=$!
sleep 2

cd "$QWEN_DIR"
/usr/bin/time -v ./runcu Qwen3-0.6B-FP32.gguf -q "What is eBPF?" -r 1 > "$OUTPUT_DIR/baseline_output.txt" 2>&1

sudo kill -SIGINT $TRACE_PID
wait $TRACE_PID 2>/dev/null || true
sleep 1

echo "✓ Baseline 完成"
echo ""

# 2. Noisy Neighbor: CPU 密集型
echo "[2/6] 测试 Noisy Neighbor (CPU 密集型)..."
# 启动 CPU 干扰：在所有核心上运行计算密集型任务
stress-ng --cpu 0 --cpu-method fft --metrics-brief &
STRESS_PID=$!
sleep 2

sudo "$TRACE_TOOL" > "$OUTPUT_DIR/noisy_cpu_trace.csv" 2> "$OUTPUT_DIR/noisy_cpu.log" &
TRACE_PID=$!
sleep 2

cd "$QWEN_DIR"
/usr/bin/time -v ./runcu Qwen3-0.6B-FP32.gguf -q "What is eBPF?" -r 1 > "$OUTPUT_DIR/noisy_cpu_output.txt" 2>&1

sudo kill -SIGINT $TRACE_PID
kill $STRESS_PID
wait $TRACE_PID 2>/dev/null || true
wait $STRESS_PID 2>/dev/null || true
sleep 1

echo "✓ Noisy CPU 完成"
echo ""

# 3. Noisy Neighbor: 网络 I/O
echo "[3/6] 测试 Noisy Neighbor (网络 I/O)..."
# 启动网络干扰：大量 TCP 连接
iperf3 -s -p 5201 &
IPERF_SERVER=$!
sleep 1
iperf3 -c 127.0.0.1 -p 5201 -t 60 -P 10 &
IPERF_CLIENT=$!
sleep 2

sudo "$TRACE_TOOL" > "$OUTPUT_DIR/noisy_network_trace.csv" 2> "$OUTPUT_DIR/noisy_network.log" &
TRACE_PID=$!
sleep 2

cd "$QWEN_DIR"
/usr/bin/time -v ./runcu Qwen3-0.6B-FP32.gguf -q "What is eBPF?" -r 1 > "$OUTPUT_DIR/noisy_network_output.txt" 2>&1

sudo kill -SIGINT $TRACE_PID
kill $IPERF_CLIENT $IPERF_SERVER 2>/dev/null || true
wait $TRACE_PID 2>/dev/null || true
sleep 1

echo "✓ Noisy Network 完成"
echo ""

# 4. Noisy Neighbor: 磁盘 I/O
echo "[4/6] 测试 Noisy Neighbor (磁盘 I/O)..."
# 启动磁盘干扰：随机读写
fio --name=randwrite --ioengine=libaio --iodepth=32 --rw=randwrite --bs=4k --direct=1 \
    --size=1G --numjobs=4 --runtime=60 --time_based --filename=/tmp/fio_test &
FIO_PID=$!
sleep 2

sudo "$TRACE_TOOL" > "$OUTPUT_DIR/noisy_disk_trace.csv" 2> "$OUTPUT_DIR/noisy_disk.log" &
TRACE_PID=$!
sleep 2

cd "$QWEN_DIR"
/usr/bin/time -v ./runcu Qwen3-0.6B-FP32.gguf -q "What is eBPF?" -r 1 > "$OUTPUT_DIR/noisy_disk_output.txt" 2>&1

sudo kill -SIGINT $TRACE_PID
kill $FIO_PID 2>/dev/null || true
wait $TRACE_PID 2>/dev/null || true
wait $FIO_PID 2>/dev/null || true
rm -f /tmp/fio_test
sleep 1

echo "✓ Noisy Disk 完成"
echo ""

# 5. 重度负载：CPU + Network + Disk 同时运行
echo "[5/6] 测试 Heavy Load (CPU + Network + Disk)..."
# 启动所有三种干扰
stress-ng --cpu 0 --cpu-method fft --metrics-brief &
STRESS_PID=$!
sleep 1

iperf3 -s -p 5201 &
IPERF_SERVER=$!
sleep 1
iperf3 -c 127.0.0.1 -p 5201 -t 60 -P 10 &
IPERF_CLIENT=$!
sleep 1

fio --name=randwrite --ioengine=libaio --iodepth=32 --rw=randwrite --bs=4k --direct=1 \
    --size=1G --numjobs=4 --runtime=60 --time_based --filename=/tmp/fio_test &
FIO_PID=$!
sleep 2

sudo "$TRACE_TOOL" > "$OUTPUT_DIR/heavy_load_trace.csv" 2> "$OUTPUT_DIR/heavy_load.log" &
TRACE_PID=$!
sleep 2

cd "$QWEN_DIR"
/usr/bin/time -v ./runcu Qwen3-0.6B-FP32.gguf -q "What is eBPF?" -r 1 > "$OUTPUT_DIR/heavy_load_output.txt" 2>&1

sudo kill -SIGINT $TRACE_PID
kill $STRESS_PID $IPERF_CLIENT $IPERF_SERVER $FIO_PID 2>/dev/null || true
wait $TRACE_PID 2>/dev/null || true
wait $STRESS_PID 2>/dev/null || true
wait $FIO_PID 2>/dev/null || true
rm -f /tmp/fio_test
sleep 1

echo "✓ Heavy Load 完成"
echo ""

# 6. 优化：CPU 绑核 + 高优先级
echo "[6/6] 测试优化 (绑核 + 高优先级 + noisy CPU)..."
stress-ng --cpu 0 --cpu-method fft --metrics-brief &
STRESS_PID=$!
sleep 2

sudo "$TRACE_TOOL" > "$OUTPUT_DIR/optimized_trace.csv" 2> "$OUTPUT_DIR/optimized.log" &
TRACE_PID=$!
sleep 2

cd "$QWEN_DIR"
# 绑定到 CPU 0-3，高优先级
sudo taskset -c 0-3 nice -n -10 /usr/bin/time -v ./runcu Qwen3-0.6B-FP32.gguf -q "What is eBPF?" -r 1 > "$OUTPUT_DIR/optimized_output.txt" 2>&1

sudo kill -SIGINT $TRACE_PID
kill $STRESS_PID
wait $TRACE_PID 2>/dev/null || true
wait $STRESS_PID 2>/dev/null || true
sleep 1

echo "✓ 优化测试完成"
echo ""

# 生成分析报告
echo "==================================================================="
echo "生成分析报告..."
echo "==================================================================="

cat > "$OUTPUT_DIR/analyze_all.py" << 'PYEOF'
import pandas as pd
import re
import sys

scenarios = [
    ("Baseline (干净)", "baseline"),
    ("Noisy CPU", "noisy_cpu"),
    ("Noisy Network", "noisy_network"),
    ("Noisy Disk", "noisy_disk"),
    ("Heavy Load", "heavy_load"),
    ("Optimized (绑核)", "optimized"),
]

print("=" * 80)
print("Noisy Neighbor 测试结果对比")
print("=" * 80)
print()

results = []

for name, prefix in scenarios:
    # Parse time output
    time_file = f"/tmp/noisy_neighbor_test/{prefix}_output.txt"
    try:
        with open(time_file) as f:
            content = f.read()
            # Extract elapsed time
            elapsed_match = re.search(r'tok/s: ([\d.]+)', content)
            if elapsed_match:
                tok_s = float(elapsed_match.group(1))
            else:
                tok_s = 0.0
    except:
        tok_s = 0.0

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

    # Parse trace CSV for runtime
    csv_file = f"/tmp/noisy_neighbor_test/{prefix}_trace.csv"
    try:
        df = pd.read_csv(csv_file)
        runtime = df['timestamp_ns'].max() / 1e9

        # Calculate IRQ time
        irq_exits = df[df['event_type'].str.contains('irqExit', na=False)]
        irq_time_ms = irq_exits['duration_ns'].sum() / 1e6
        irq_pct = (irq_time_ms / (runtime * 1000)) * 100 if runtime > 0 else 0
    except:
        runtime = 0
        irq_time_ms = 0
        irq_pct = 0

    results.append({
        'scenario': name,
        'tok_s': tok_s,
        'runtime': runtime,
        'launches': launches,
        'sched_switches': sched_switches,
        'soft_irqs': soft_irqs,
        'hard_irqs': hard_irqs,
        'irq_time_ms': irq_time_ms,
        'irq_pct': irq_pct,
    })

# Print table
print(f"{'场景':<20} {'tok/s':>10} {'运行时间':>10} {'Launches':>10} {'Sched':>8} {'Soft IRQ':>10} {'Hard IRQ':>10} {'IRQ时间':>10} {'IRQ%':>8}")
print("-" * 120)

baseline_tok = results[0]['tok_s']

for r in results:
    degradation = ((baseline_tok - r['tok_s']) / baseline_tok * 100) if baseline_tok > 0 else 0
    degradation_str = f"({degradation:+.1f}%)" if r['scenario'] != "Baseline (干净)" else ""

    print(f"{r['scenario']:<20} {r['tok_s']:>10.2f} {degradation_str:>8} "
          f"{r['runtime']:>10.2f}s {r['launches']:>10} {r['sched_switches']:>8} "
          f"{r['soft_irqs']:>10} {r['hard_irqs']:>10} "
          f"{r['irq_time_ms']:>8.2f}ms {r['irq_pct']:>7.3f}%")

print()
print("=" * 80)
print("关键发现:")
print("=" * 80)

# Compare baseline vs worst noisy
worst = max(results[1:4], key=lambda x: x['runtime'])
optimized = results[4]

print(f"1. 最差 noisy neighbor ({worst['scenario']}): 性能下降 {((baseline_tok - worst['tok_s']) / baseline_tok * 100):.1f}%")
print(f"2. 优化后恢复: {((optimized['tok_s'] - worst['tok_s']) / worst['tok_s'] * 100):+.1f}%")
print(f"3. IRQ 影响: Baseline {results[0]['irq_pct']:.3f}% → Worst {worst['irq_pct']:.3f}%")
print(f"4. 调度切换: Baseline {results[0]['sched_switches']} → Worst {worst['sched_switches']}")
PYEOF

python3 "$OUTPUT_DIR/analyze_all.py"

echo ""
echo "详细追踪数据位于: $OUTPUT_DIR"
echo "==================================================================="
