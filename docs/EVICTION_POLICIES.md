# GPU Memory Eviction Policies

本文档介绍 GPU 内存驱逐策略的 BPF 实现及测试脚本。

## 策略概览

| 策略 | 文件名 | 特点 | 适用场景 |
|------|--------|------|----------|
| **FIFO** | `eviction_fifo` | 先进先出 | 简单基准 |
| **LRU** | (内核默认) | 最近最少使用 | 通用 |
| **MRU** | `eviction_mru` | 最近最多使用 | 顺序扫描 |
| **LFU** | `eviction_lfu` | 最少频率使用 | 热点数据 |
| **PID Quota** | `eviction_pid_quota` | 基于配额的 PID 优先级 | 进程隔离 |
| **Freq Decay** | `eviction_freq_pid_decay` | 基于访问频率衰减 | 访问模式感知 |
| **FIFO Chance** | `eviction_fifo_chance` | FIFO + 二次机会 + PID | 平衡保护 |

## PID 优先级策略详解

### 1. eviction_pid_quota (配额策略)

基于配额百分比的驱逐策略：
- 高优先级进程：大配额（如 80%），chunk 更难被驱逐
- 低优先级进程：小配额（如 20%），超出配额的 chunk 容易被驱逐

```bash
sudo ./eviction_pid_quota -p <HIGH_PID> -P 80 -l <LOW_PID> -L 20
```

**参数含义**：
- `-P N`: 高优先级配额百分比 (0-100, 0=无限制)
- `-L N`: 低优先级配额百分比

### 2. eviction_freq_pid_decay (频率衰减策略)

基于访问频率的驱逐策略：
- 高优先级：每 N 次访问移动到 tail（N=1 表示每次访问都保护）
- 低优先级：每 N 次访问才移动（N 越大保护越少）

```bash
sudo ./eviction_freq_pid_decay -p <HIGH_PID> -P 1 -l <LOW_PID> -L 10
```

**参数含义**：
- `-P N`: 高优先级衰减因子 (N=1 表示每次访问都 move_tail)
- `-L N`: 低优先级衰减因子 (N=10 表示每 10 次访问才 move_tail)

### 3. eviction_fifo_chance (FIFO + 二次机会)

结合 FIFO 顺序和访问感知的二次机会策略：
- 保持 FIFO 的基本顺序（先进入的先被考虑驱逐）
- 每个 chunk 有"机会次数"，被访问时重置
- 驱逐检查时：机会 > 0 则减 1 并给第二次机会；机会 = 0 则驱逐

```bash
sudo ./eviction_fifo_chance -p <HIGH_PID> -P 3 -l <LOW_PID> -L 0
```

**参数含义**：
- `-P N`: 高优先级初始机会次数 (N=3 表示 3 次机会)
- `-L N`: 低优先级初始机会次数 (N=0 表示立即驱逐)

**算法流程**：

| Hook | 行为 |
|------|------|
| `activate` | 根据 PID 设置初始 chance_count |
| `chunk_used` | 重置 chance_count（被访问=有价值），保持 FIFO 顺序不变 |
| `eviction_prepare` | 检查 HEAD：chance>0 则 chance--，move_tail；chance=0 则驱逐 |

**示例**：

```
高优先级 (chance=3):
  [访问] → chance=3
  [驱逐检查1] → chance=2, move_tail (saved)
  [驱逐检查2] → chance=1, move_tail (saved)
  [驱逐检查3] → chance=0, move_tail (saved)
  [驱逐检查4] → evicted

低优先级 (chance=0):
  [访问] → chance=0
  [驱逐检查1] → evicted (立即)
```

## 测试脚本

### test_pid_lfu.py

统一的测试脚本，支持测试所有 PID 优先级 eviction 策略。

**位置**: `/home/yunwei37/workspace/gpu/co-processor-demo/memory/micro/test_pid_lfu.py`

#### 基本用法

```bash
# 列出可用策略
python3 test_pid_lfu.py --list-policies

# 测试 quota 策略
sudo python3 test_pid_lfu.py --policy eviction_pid_quota -P 80 -L 20

# 测试 frequency decay 策略
sudo python3 test_pid_lfu.py --policy eviction_freq_pid_decay -P 1 -L 10

# 测试 FIFO chance 策略
sudo python3 test_pid_lfu.py --policy eviction_fifo_chance -P 3 -L 0
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--policy NAME` | 策略可执行文件名 | `eviction_pid_quota` |
| `-P N` | 高优先级参数 | 80 |
| `-L N` | 低优先级参数 | 20 |
| `-k KERNEL` | 访问模式 (seq_stream, rand_stream, pointer_chase) | `rand_stream` |
| `-s FACTOR` | uvmbench size_factor | 0.6 |
| `-i N` | 每轮迭代次数 | 3 |
| `--stride-bytes N` | 访问步长 | 4096 |
| `-r N` | 测试轮数 | 1 |
| `--baseline-no-policy` | 基准测试不使用策略 | False |

#### 输出格式

脚本输出包含：

1. **配置信息**：策略名称、参数、访问模式等
2. **Warmup 阶段**：预热 GPU
3. **Baseline 测试**：使用相等参数作为基准
4. **Policy 测试**：使用指定参数测试
5. **统计信息**：
   - Current active chunks: 当前活跃 chunk 数
   - Total activated: 总激活次数
   - Total used calls: 总使用调用次数
   - Policy allow (moved/saved): 策略允许保护的次数
   - Policy deny (not moved/evicted): 策略拒绝保护的次数
6. **Summary**：性能对比和加速比

#### 示例输出

```
============================================================
PID-BASED EVICTION POLICY EXPERIMENT
============================================================
Policy: eviction_fifo_chance
Parameters:
  High priority param (-P): 3
  Low priority param (-L):  0
  Access pattern:           rand_stream
  ...

=== Per-PID Statistics ===
  High priority PID 12345:
    Current active chunks: 8192
    Total activated: 15000
    Policy allow (saved): 12000 (80.0%)
    Policy deny (evicted): 3000 (20.0%)

  Low priority PID 67890:
    Current active chunks: 2048
    Total activated: 15000
    Policy allow (saved): 1500 (10.0%)
    Policy deny (evicted): 13500 (90.0%)

=== Summary ===
  High priority: 723ms, 45.2 GB/s
  Low priority:  1229ms, 26.6 GB/s
  Speedup: 1.7x
```

## 共享数据结构

所有 PID 优先级策略使用统一的数据结构（定义在 `eviction_common.h`）：

### 配置 Key

```c
#define CONFIG_PRIORITY_PID 0       /* 高优先级 PID */
#define CONFIG_PRIORITY_PARAM 1     /* 高优先级参数 */
#define CONFIG_LOW_PRIORITY_PID 2   /* 低优先级 PID */
#define CONFIG_LOW_PRIORITY_PARAM 3 /* 低优先级参数 */
#define CONFIG_DEFAULT_PARAM 4      /* 默认参数 */
```

### 统计结构

```c
struct pid_chunk_stats {
    __u64 current_count;    /* 当前活跃 chunk 数 */
    __u64 total_activate;   /* 总激活次数 */
    __u64 total_used;       /* 总使用调用次数 */
    __u64 policy_allow;     /* 策略允许次数 (保护/移动) */
    __u64 policy_deny;      /* 策略拒绝次数 (不保护/驱逐) */
};
```

## 文件列表

| 文件 | 说明 |
|------|------|
| `src/eviction_common.h` | 共享头文件 |
| `src/eviction_pid_quota.bpf.c` | 配额策略 BPF |
| `src/eviction_pid_quota.c` | 配额策略用户态 |
| `src/eviction_freq_pid_decay.bpf.c` | 频率衰减策略 BPF |
| `src/eviction_freq_pid_decay.c` | 频率衰减策略用户态 |
| `src/eviction_fifo_chance.bpf.c` | FIFO 二次机会策略 BPF |
| `src/eviction_fifo_chance.c` | FIFO 二次机会策略用户态 |
| `memory/micro/test_pid_lfu.py` | 统一测试脚本 |

## 编译

```bash
cd gpu_ext_policy/src
make eviction_pid_quota eviction_freq_pid_decay eviction_fifo_chance
```

## 选择建议

| 场景 | 推荐策略 | 参数建议 |
|------|----------|----------|
| 简单优先级区分 | `eviction_pid_quota` | 高=80%, 低=20% |
| 访问频率敏感 | `eviction_freq_pid_decay` | 高=1, 低=10 |
| 平衡 FIFO + 访问感知 | `eviction_fifo_chance` | 高=3, 低=0 |
| 顺序扫描工作负载 | `eviction_mru` | - |
| 热点数据工作负载 | `eviction_lfu` | - |
