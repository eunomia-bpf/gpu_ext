# UVM Kernel Module 调用图生成指南

这个指南介绍如何为NVIDIA UVM kernel module生成调用图(call graph)，帮助理解代码流程和policy decision points。

---

## 方法1: cflow (最简单，推荐新手)

### 安装
```bash
sudo apt-get install cflow
```

### 生成文本格式调用图
```bash
cd /home/yunwei37/open-gpu-kernel-modules/kernel-open/nvidia-uvm

# 生成特定函数的调用图
cflow --tree --depth=5 uvm_gpu_replayable_faults.c > callgraph_fault_service.txt

# 生成完整的调用图（会很大）
cflow --omit-arguments --tree *.c > callgraph_full.txt

# 只看指定函数被谁调用（reverse模式）
cflow --reverse --tree uvm_gpu_replayable_faults.c | grep "uvm_parent_gpu_service_replayable_faults"

# 生成特定函数的调用链
cflow --tree --main=uvm_parent_gpu_service_replayable_faults uvm_gpu_replayable_faults.c
```

### 示例输出
```
uvm_parent_gpu_service_replayable_faults() <void uvm_parent_gpu_service_replayable_faults (uvm_parent_gpu_t *parent_gpu) at uvm_gpu_replayable_faults.c:2906>:
    fetch_fault_buffer_entries()
    preprocess_fault_batch()
    service_fault_batch()
        service_managed_fault_in_block()
            uvm_va_block_make_resident()
                block_copy_resident_pages()
                    block_copy_resident_pages_between()
    push_replay_on_parent_gpu()
        parent_gpu->host_hal->replay_faults()  # HAL回调点！
```

---

## 方法2: Doxygen (最详细，带HTML可视化)

### 安装
```bash
sudo apt-get install doxygen graphviz
```

### 配置文件
创建 `Doxyfile.uvm`:

```bash
cd /home/yunwei37/open-gpu-kernel-modules/kernel-open/nvidia-uvm

cat > Doxyfile.uvm << 'EOF'
PROJECT_NAME           = "NVIDIA UVM Driver"
OUTPUT_DIRECTORY       = doxygen_output
INPUT                  = .
FILE_PATTERNS          = *.c *.h
RECURSIVE              = NO
EXTRACT_ALL            = YES
EXTRACT_STATIC         = YES
HAVE_DOT               = YES
CALL_GRAPH             = YES
CALLER_GRAPH           = YES
DOT_IMAGE_FORMAT       = svg
INTERACTIVE_SVG        = YES
DOT_GRAPH_MAX_NODES    = 100
MAX_DOT_GRAPH_DEPTH    = 5
GENERATE_LATEX         = NO
EOF
```

### 生成调用图
```bash
doxygen Doxyfile.uvm

# 查看结果
firefox doxygen_output/html/index.html
# 或
google-chrome doxygen_output/html/index.html
```

### 查看特定函数
在HTML中搜索 `uvm_parent_gpu_service_replayable_faults`，点击函数名，会看到：
- **Calls** (调用了哪些函数)
- **Called by** (被哪些函数调用)
- 交互式SVG图表

---

## 方法3: egypt + graphviz (生成图片)

### 安装
```bash
sudo apt-get install egypt graphviz gcc
```

### 生成调用图
```bash
cd /home/yunwei37/open-gpu-kernel-modules/kernel-open/nvidia-uvm

# Step 1: 使用GCC生成RTL文件
gcc -fdump-rtl-expand -c uvm_gpu_replayable_faults.c \
    -I. -I../common/inc -I../../kernel-open/common/inc \
    -D__KERNEL__ -DMODULE \
    -o /tmp/uvm_faults.o 2>/dev/null || true

# Step 2: 使用egypt生成dot文件
egypt *.expand > callgraph.dot

# Step 3: 转换为图片
dot -Tpng callgraph.dot -o callgraph.png
dot -Tsvg callgraph.dot -o callgraph.svg  # SVG可缩放

# 只生成特定函数周围的调用图
egypt *.expand | \
  grep -A 20 -B 20 "uvm_parent_gpu_service_replayable_faults" | \
  dot -Tpng -o callgraph_fault_service.png
```

---

## 方法4: 从内核符号表生成 (运行时)

### 查看已加载的UVM函数
```bash
# 加载UVM模块
sudo modprobe nvidia-uvm

# 查看所有UVM导出的符号
cat /proc/kallsyms | grep '\[nvidia_uvm\]' | head -20

# 查看特定函数地址
sudo cat /proc/kallsyms | grep uvm_parent_gpu_service_replayable_faults

# 输出示例：
# ffffffffc0a12340 t uvm_parent_gpu_service_replayable_faults [nvidia_uvm]
```

### 使用nm分析.ko文件
```bash
cd /home/yunwei37/open-gpu-kernel-modules

# 找到编译好的.ko
find . -name "nvidia-uvm.ko"

# 查看所有函数符号
nm -C kernel-open/nvidia-uvm/nvidia-uvm.ko | grep " T " | head -20

# 查看特定函数
nm -C kernel-open/nvidia-uvm/nvidia-uvm.ko | grep service_replayable_faults

# 输出示例：
# 0000000000012340 T uvm_parent_gpu_service_replayable_faults
```

---

## 方法5: ftrace 动态追踪 (运行时调用栈)

### 启用ftrace
```bash
# 加载UVM模块
sudo modprobe nvidia-uvm

# 启用ftrace
sudo su
cd /sys/kernel/debug/tracing

# 设置追踪器
echo function_graph > current_tracer

# 过滤只看UVM函数
echo '*uvm*' > set_ftrace_filter

# 开始追踪
echo 1 > tracing_on

# 运行触发GPU fault的程序
# (例如运行CUDA程序)

# 停止追踪
echo 0 > tracing_on

# 查看结果
cat trace | head -100

# 保存结果
cat trace > /tmp/uvm_trace.txt
```

### ftrace输出示例
```
  1)               |  uvm_parent_gpu_service_replayable_faults() {
  1)               |    fetch_fault_buffer_entries() {
  1)   0.234 us    |      uvm_spin_lock();
  1)   0.123 us    |      fault_buffer_read_get();
  1)   0.456 us    |      fault_buffer_read_put();
  1)   2.345 us    |    }
  1)               |    service_fault_batch() {
  1)               |      service_managed_fault_in_block() {
  1)               |        uvm_va_block_make_resident() {
  1)  12.345 us    |        }
  1)  15.678 us    |      }
  1)  18.901 us    |    }
  1)               |    push_replay_on_parent_gpu() {
  1)   1.234 us    |    }
  1)  25.678 us    |  }
```

---

## 方法6: perf + FlameGraph (性能热点可视化)

### 安装
```bash
sudo apt-get install linux-tools-common linux-tools-generic
git clone https://github.com/brendangregg/FlameGraph.git ~/FlameGraph
```

### 生成火焰图
```bash
# 记录GPU fault处理的性能数据
sudo perf record -e cycles -g --call-graph dwarf -a -- sleep 10
# (在这10秒内运行触发GPU fault的程序)

# 生成perf报告
sudo perf script > out.perf

# 转换为火焰图
~/FlameGraph/stackcollapse-perf.pl out.perf | \
~/FlameGraph/flamegraph.pl > flamegraph.svg

# 查看
firefox flamegraph.svg
```

---

## 方法7: BPF/bpftrace 追踪 (最灵活)

### 追踪特定函数
```bash
# 追踪fault service函数的调用
sudo bpftrace -e '
kprobe:uvm_parent_gpu_service_replayable_faults {
    printf("=== Fault Service Called ===\n");
    printf("Stack trace:\n%s\n", kstack);
}
'

# 追踪replay policy决策点
sudo bpftrace -e '
kprobe:push_replay_on_parent_gpu {
    printf("Replay triggered at %lld ns\n", nsecs);
    printf("Stack: %s\n", kstack);
}
'

# 统计函数调用次数
sudo bpftrace -e '
kprobe:uvm_* {
    @calls[func] = count();
}
interval:s:5 {
    print(@calls);
    clear(@calls);
}
'
```

---

## 方法8: 使用Clang Static Analyzer

### 安装
```bash
sudo apt-get install clang clang-tools
```

### 生成调用图
```bash
cd /home/yunwei37/open-gpu-kernel-modules/kernel-open/nvidia-uvm

# 使用clang生成IR
clang -S -emit-llvm -I. -I../common/inc -D__KERNEL__ -DMODULE \
      uvm_gpu_replayable_faults.c -o uvm_faults.ll

# 分析IR中的函数调用
grep "call.*@uvm" uvm_faults.ll | sort | uniq

# 使用opt生成调用图
opt -dot-callgraph uvm_faults.ll -o /dev/null
# 会生成 callgraph.dot 文件
dot -Tpng callgraph.dot -o callgraph_clang.png
```

---

## 方法9: 自定义脚本分析源码

### Python脚本提取函数调用
```python
#!/usr/bin/env python3
# callgraph_extractor.py

import re
import sys
from collections import defaultdict

def extract_calls(filename):
    """从C文件提取函数调用关系"""
    calls = defaultdict(list)
    current_func = None

    with open(filename, 'r') as f:
        lines = f.readlines()

    # 匹配函数定义: type func_name(
    func_def_re = re.compile(r'^[\w\s\*]+\s+(\w+)\s*\(')
    # 匹配函数调用: func_name(
    func_call_re = re.compile(r'\b(\w+)\s*\(')

    in_function = False
    brace_count = 0

    for line in lines:
        # 跳过注释
        if line.strip().startswith('//') or line.strip().startswith('/*'):
            continue

        # 检测函数定义
        match = func_def_re.match(line)
        if match and '{' in line:
            current_func = match.group(1)
            in_function = True
            brace_count = line.count('{') - line.count('}')
            continue

        if in_function:
            brace_count += line.count('{') - line.count('}')

            # 提取函数调用
            for match in func_call_re.finditer(line):
                called_func = match.group(1)
                # 过滤关键字和宏
                if called_func not in ['if', 'for', 'while', 'switch', 'sizeof', 'return']:
                    if called_func != current_func:  # 避免自递归
                        calls[current_func].append(called_func)

            # 函数结束
            if brace_count == 0:
                in_function = False
                current_func = None

    return calls

def generate_dot(calls, focus_func=None):
    """生成Graphviz DOT格式"""
    print("digraph callgraph {")
    print("  node [shape=box];")

    if focus_func:
        # 只显示与focus_func相关的调用
        visited = set()

        def traverse(func, depth=0):
            if depth > 3 or func in visited:
                return
            visited.add(func)

            if func in calls:
                for callee in set(calls[func]):
                    print(f'  "{func}" -> "{callee}";')
                    traverse(callee, depth + 1)

        traverse(focus_func)
    else:
        # 显示所有调用
        for caller, callees in calls.items():
            for callee in set(callees):
                print(f'  "{caller}" -> "{callee}";')

    print("}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 callgraph_extractor.py <file.c> [focus_function]")
        sys.exit(1)

    filename = sys.argv[1]
    focus = sys.argv[2] if len(sys.argv) > 2 else None

    calls = extract_calls(filename)
    generate_dot(calls, focus)
```

### 使用脚本
```bash
cd /home/yunwei37/open-gpu-kernel-modules/kernel-open/nvidia-uvm

# 保存上面的Python脚本
cat > /tmp/callgraph_extractor.py << 'EOFPY'
[上面的Python代码]
EOFPY

chmod +x /tmp/callgraph_extractor.py

# 生成fault service的调用图
python3 /tmp/callgraph_extractor.py uvm_gpu_replayable_faults.c \
    uvm_parent_gpu_service_replayable_faults | \
    dot -Tpng -o fault_service_calls.png

# 生成完整调用图
python3 /tmp/callgraph_extractor.py uvm_gpu_replayable_faults.c | \
    dot -Tsvg -o full_callgraph.svg
```

---

## 推荐工作流程

### 场景1: 快速查看特定函数的调用关系

```bash
# 最快速 - 使用cflow
cd kernel-open/nvidia-uvm
cflow --tree --main=uvm_parent_gpu_service_replayable_faults \
      --depth=3 uvm_gpu_replayable_faults.c | less
```

### 场景2: 生成可视化的完整调用图

```bash
# 使用doxygen (最详细)
cd kernel-open/nvidia-uvm
doxygen Doxyfile.uvm
firefox doxygen_output/html/index.html
```

### 场景3: 追踪运行时实际调用路径

```bash
# 使用ftrace (最准确)
sudo su
cd /sys/kernel/debug/tracing
echo function_graph > current_tracer
echo '*uvm_parent_gpu_service_replayable_faults*' > set_ftrace_filter
echo 1 > tracing_on

# 运行触发fault的程序
cuda_app

# 查看结果
cat trace > /tmp/uvm_runtime_trace.txt
```

### 场景4: 找到所有policy决策点

```bash
# 结合grep和调用图
cd kernel-open/nvidia-uvm

# 找到所有policy相关的调用
cflow --tree *.c | grep -i policy > policy_calls.txt

# 找到所有if判断（潜在决策点）
grep -n "if.*policy\|if.*mode\|if.*strategy" *.c > policy_decisions.txt

# 生成policy决策函数的调用图
for func in $(grep "policy" policy_decisions.txt | cut -d: -f1 | sort | uniq); do
    cflow --tree --main=$func *.c > callgraph_$func.txt
done
```

---

## 针对WIC论文的调用图分析

### 找到Interrupter (replay)相关调用链
```bash
cd kernel-open/nvidia-uvm

# 方法1: cflow查看replay函数
cflow --tree --reverse uvm_gpu_replayable_faults.c | \
  grep -A 5 "push_replay_on_gpu"

# 方法2: 查看HAL replay实现
grep -rn "\.replay_faults.*=" . | grep -v ".o:"

# 方法3: 追踪运行时replay调用
sudo bpftrace -e '
kprobe:push_replay_on_parent_gpu {
    printf("Replay at %lld: ", nsecs);
    printf("%s\n", kstack);
}
'
```

### 找到Monitor (thrashing)相关调用链
```bash
# 查看thrashing hint调用者
cflow --reverse --tree uvm_perf_thrashing.c | \
  grep -A 10 "uvm_perf_thrashing_get_hint"

# 追踪运行时thrashing检测
sudo bpftrace -e '
kprobe:uvm_perf_thrashing_get_hint {
    printf("Thrashing check at addr %llx\n", arg2);
}
kretprobe:uvm_perf_thrashing_get_hint {
    printf("  -> hint type: %d\n", retval);
}
'
```

### 找到Activator (prefetch)相关调用链
```bash
# 查看prefetch决策
cflow --tree uvm_perf_prefetch.c | grep -A 5 "prefetch"

# 追踪运行时prefetch
sudo bpftrace -e '
kprobe:uvm_perf_prefetch* {
    @prefetch[func] = count();
}
interval:s:1 {
    print(@prefetch);
}
'
```

---

## 输出结果示例

### cflow输出
```
uvm_parent_gpu_service_replayable_faults() <void at uvm_gpu_replayable_faults.c:2906> (R):
    uvm_tracker_init()
    fetch_fault_buffer_entries() <int at uvm_gpu_replayable_faults.c:844>:
        fault_buffer_read_get()
        fault_buffer_read_put()
        parse_fault_entry()
    preprocess_fault_batch() <int at uvm_gpu_replayable_faults.c:1234>:
        classify_fault()
        check_throttling()
    service_fault_batch() <int at uvm_gpu_replayable_faults.c:2232>:
        service_managed_fault_in_block()
            uvm_va_block_make_resident() <--- 策略决策点
                block_copy_resident_pages()
    push_replay_on_parent_gpu() <int at uvm_gpu_replayable_faults.c:503> (R):
        parent_gpu->host_hal->replay_faults() <--- HAL回调点！策略可扩展
```

### Graphviz可视化
生成的PNG/SVG图会显示：
- 矩形框 = 函数
- 箭头 = 调用关系
- 颜色/大小 = 调用频率（如果有运行时数据）

---

## 总结

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **cflow** | 快速、简单 | 文本输出 | 快速查看调用关系 |
| **doxygen** | 详细、可交互 | 生成慢、文件大 | 完整文档+可视化 |
| **egypt** | 图形化 | 需要编译 | 生成图片 |
| **ftrace** | 运行时实际路径 | 需root权限 | 调试、性能分析 |
| **bpftrace** | 灵活、低开销 | 需BPF支持 | 动态追踪 |
| **nm/kallsyms** | 快速查符号 | 无调用关系 | 查找函数地址 |
| **自定义脚本** | 可定制 | 可能不准确 | 特定需求 |

**推荐组合**：
1. 先用 `cflow` 快速了解
2. 用 `doxygen` 生成详细文档
3. 用 `ftrace/bpftrace` 验证运行时行为
