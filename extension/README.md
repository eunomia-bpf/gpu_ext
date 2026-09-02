# GPU Memory Management BPF Policies

NVIDIA UVM (Unified Virtual Memory) BPF struct_ops 策略实现，用于优化 GPU 内存管理。

Pages: 4KB, VA block: 2MB (512 pages), Chunks: 2MB

## 系统架构

### 核心概念

- **Physical Chunk**: GPU 内存管理基本单位 (2MB)。Eviction policy 操作的对象。
- **VA Block**: 虚拟地址范围 `[va_start, va_end)`, 2MB，映射到物理 chunks。
- **Prefetch**: page fault 时将 CPU 内存迁移到 GPU，减少后续 fault 延迟。
- **Eviction**: GPU 内存不足时回收 chunks。UVM 从 list HEAD evict，TAIL 最安全。

```
Virtual Address Space              Physical Memory (GPU VRAM)
+---------------------+           +--------------+
|   VA Block 1        |           |  Chunk A     |
|   [va_start, va_end]|----+----->|  (2MB)       |
|   2MB               |    |      |  [ACTIVE]    |
+---------------------+    |      +--------------+
                            |      +--------------+
+---------------------+    +----->|  Chunk B     |
|   VA Block 2        |           |  (2MB)       |
|   2MB               |---------->|  [ACTIVE]    |
+---------------------+           +--------------+
```

### Chunk 状态转换

```
Unused Pool --> [assign] --> Active (gpu_block_activate)
Active <-- [gpu_block_access] --> Active (更新 LRU 位置)
Active --> [evict] --> In-Eviction (gpu_evict_prepare) --> Unused Pool
```

- **Unused**: chunk 在 free pool 中，等待被分配
- **Active**: chunk 被分配给 VA block，处于 evictable 状态
- **In-Eviction**: 正在从 VA block 解除映射，即将回到 unused pool

---

## BPF Hook 点

NVIDIA UVM 提供 6 个 BPF struct_ops hook 点：

### Eviction 相关

#### `gpu_block_activate`

chunk 被分配给 VA block 后，unpin 进入 evictable 状态时触发。

```c
int gpu_block_activate(uvm_pmm_gpu_t *pmm,
                       uvm_gpu_chunk_t *chunk,
                       uvm_bpf_pmm_decision_ctx_t *decision_ctx);
```

**重要**: 这是 eviction policy 的主要 hook。所有访问计数、频率统计、list 位置调整都应在此实现。策略只能通过 `bpf_gpu_request_reorder(decision_ctx, destination, position)` 提交 callback-local 请求；kfunc 不接受 list 指针且不直接修改链表。driver 在 callback 后验证 PMM/root 身份、source state、generation、destination 和 position，再至多提交一次移动。

activate 在原生移动到 used list 后执行。无请求会保留该位置；合法 HEAD/TAIL 请求只在 callback 返回后由 driver 提交。callback 返回 action 不控制 activate 的原生路由。

#### `gpu_block_access`

chunk 被访问时触发。

```c
int gpu_block_access(uvm_pmm_gpu_t *pmm,
                     uvm_gpu_chunk_t *chunk,
                     uvm_bpf_pmm_decision_ctx_t *decision_ctx);
```

无请求且返回 DEFAULT 时保留原生 LRU tail move；无请求且返回 BYPASS 时保持 callback-entry 位置。一个合法请求配合 DEFAULT 或 BYPASS 时由 driver 提交一次并抑制原生移动；invalid/conflict/stale 请求及非法 action 均保持 callback-entry 位置。

该回调在原生 access tail move 之前执行。需要 access-time 更新的策略可在这里记录 callback-local reorder；不需要覆盖原生 LRU 的策略应返回 DEFAULT 且不提交请求。

#### `gpu_evict_prepare`

内存压力触发 eviction 时调用。

```c
int gpu_evict_prepare(uvm_pmm_gpu_t *pmm, struct list_head *va_block_used, struct list_head *va_block_unused);
```

### Prefetch 相关

#### `gpu_page_prefetch`

page fault 时决定预取范围。只能在当前 VA block (2MB) 内操作。

```c
int gpu_page_prefetch(uvm_page_index_t page_index,
    uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
    uvm_va_block_region_t *max_prefetch_region,
    uvm_bpf_prefetch_decision_t *decision_ctx);
```

通过 `bpf_gpu_set_prefetch_region(decision_ctx, first, outer)` 请求绝对
`[first, outer)` 区间。返回值: 0 (DEFAULT), 1 (BYPASS，验证并使用请求),
2 (ENTER_LOOP，逐区域迭代)。直接写 context 会被 verifier 拒绝。

#### `gpu_page_prefetch_iter`

`gpu_page_prefetch` 返回 ENTER_LOOP 时，对每个候选区域调用。

```c
int gpu_page_prefetch_iter(uvm_perf_prefetch_bitmap_tree_t *bitmap_tree,
    uvm_va_block_region_t *max_prefetch_region,
    uvm_va_block_region_t *current_region,
    unsigned int counter,
    uvm_bpf_prefetch_decision_t *decision_ctx);
```

#### `gpu_test_trigger`

测试/调试用，通过 proc 文件触发。

### 跨 VA Block Prefetch

Prefetch hook 只能在当前 VA block (2MB) 内操作。原型曾把整数
`va_space` handle 转回内核指针并通过 `bpf_gpu_migrate_range()` 跨 block
迁移，但该接口没有原生对象生命周期保证，现已从注册 kfunc 集移除。
依赖它的策略保留为源码参考，并从默认验证构建中排除；恢复这些策略
需要另行评审带所有权、代际、失效与 teardown 同步的原生 lookup 设计。

---

## Eviction 策略

| 文件 | 策略 | 实现 |
|------|------|------|
| `eviction_fifo.bpf.c` | FIFO | 不移动 chunk，保持插入顺序，bypass 所有 access |
| `eviction_mru.bpf.c` | MRU | activate 时 move_head，最近使用的优先 evict |
| `eviction_lfu.bpf.c` | LFU | per-CPU array 频率计数，低频在 head |
| `eviction_2q_approx.bpf.c` | 近似 2Q / segmented LRU | 新 chunk 放 HEAD probation；不同 list-generation episode 达到二次命中门槛后移到 TAIL protected；同 generation 的 activate/access 只计一次 |
| `eviction_cycle_moe.bpf.c` | T1 频率保护 | activate 时计数，access_count >= 3 则 move_tail 保护 |
| `eviction_fifo_chance.bpf.c` | PID FIFO + 二次机会 | FIFO 基础上，被 evict 时减 chance 值，有 chance 则移到 tail |
| `eviction_freq_pid_decay.bpf.c` | PID 频率衰减 | per-PID 频率计数，频率随时间指数衰减 |
| `eviction_pid_quota.bpf.c` | PID 配额 | 按进程配额分配 GPU 内存，超额 chunk 不保护 |
| `eviction_lfu_xcoord.bpf.c` | LFU + xCoord | LFU + GPU 共享状态 map，multi-tenant 协调 |

---

## Prefetch 策略

### 基础 Prefetch

| 文件 | 策略 | 实现 |
|------|------|------|
| `prefetch_none.bpf.c` | 禁用 | 设置空 region，纯 demand paging |
| `prefetch_always_max.bpf.c` | 全量预取 | 每次 fault 预取整个 max_prefetch_region (2MB)，最基础也最有效的策略 |
| `prefetch_adaptive_sequential.bpf.c` | 自适应顺序 | 按访问密度阈值决定是否预取子区域，使用 ENTER_LOOP 模式 |
| `prefetch_adaptive_tree_iter.bpf.c` | 自适应树迭代 | 遍历 bitmap tree，按每个子区域的 access count 决定 |
| `prefetch_stride.bpf.c` | 步长检测 | 检测固定步长访问模式，预取下一个预测位置 |
| `prefetch_delta_markov.bpf.c` | Block-local delta/Markov | 按 VA block 学习 `delta[n] -> delta[n+1]`，置信度门控并严格裁剪到当前 block；只读 kprobe 补充 block identity |
| `prefetch_pid_tree.bpf.c` | PID 树 | PID-aware 预取带宽分配，高优先级进程获得更多预取 |
| `prefetch_serving_adaptive.bpf.c` | 服务自适应 | fault-rate 门控的 always_max，低 fault rate 时禁用预取减少干扰 |

### 组合策略（Prefetch + Eviction）

| 文件 | 策略 | 实现 |
|------|------|------|
| `prefetch_always_max_cycle_moe.bpf.c` | always_max + T1 保护 | always_max 预取 + cycle_moe T1 频率保护 eviction。per-CPU array 计数，access >= 3 则 move_tail。当前所有 workload 的默认最佳策略 |
| `prefetch_template_belady.bpf.c` | always_max + Belady OPT | always_max 预取 + Belady 最优驱逐。基于层循环距离预测 future use，需要 profiling 数据填充 VA→layer 映射表 |
| `prefetch_max_passive_mru.bpf.c` | always_max + Passive MRU | always_max 预取 + T1 保护 + 非 T1 chunk 冻结 LRU 位置 |
| `prefetch_max_mru_expert.bpf.c` | always_max + MRU Expert | always_max 预取 + expert 感知 MRU 驱逐，对 expert 层和 attention 层分级保护 |
| `prefetch_eviction_pid.bpf.c` | PID Prefetch+Eviction | PID-aware 预取阈值 + probabilistic LRU 驱逐 |
| `prefetch_always_max_qos.bpf.c` | always_max + QoS 反馈 | always_max + cycle_moe + fault-rate 反馈控制 eviction 激进度 |
| `prefetch_cooperative.bpf.c` | 协作 Prefetch-Eviction | 预取和驱逐协同：预取前检查 eviction 压力，压力高时减少预取 |
| `prefetch_reuse_dist.bpf.c` | 重用距离 Eviction | always_max 预取 + reuse distance 估算驱逐。用 stack distance 近似 Belady |
| `prefetch_always_max_xcoord.bpf.c` | always_max + xCoord | always_max + cycle_moe + GPU 共享状态 map，multi-tenant 场景 |

### 跨 VA Block 预取

下列策略当前为 `UNAVAILABLE`，原因是其 raw VA-space handle 迁移接口已
按安全评审移除；它们不在默认 `BPF_APPS` 验证构建中。

| 文件 | 策略 | 实现 |
|------|------|------|
| `prefetch_cross_block_v2.bpf.c` | 跨 block 方向预取 | always_max (intra-block) + bpf_wq 异步预取相邻 VA block。kprobe 捕获 va_space 上下文，支持 4 种模式：方向感知 (mode 0)、盲目相邻 (mode 1)、stride (mode 2)、adjacent-stride (mode 3)。可配置 eviction mode 和 prefetch 大小。CLI: `./prefetch_cross_block_v2 [evict_mode] [prefetch_kb] [prefetch_mode]` |
| `prefetch_stride_multiblock.bpf.c` | 多 block 步长预取 | 步长检测 + 同时预取 K 个连续 VA block (K=1-6)。bpf_wq 异步迁移 |
| `prefetch_throttled_xb.bpf.c` | 节流跨 block | fault-rate 门控的跨 block 预取，高 fault rate 时限制跨 block 预取量 |

### Phase Detection（阶段检测）

通过 uprobe 挂载用户态函数，检测 workload 运行阶段（prefill/decode、build/search），根据阶段切换预取策略。

| 文件 | 策略 | 实现 |
|------|------|------|
| `prefetch_llama_phase.bpf.c` | llama.cpp 阶段检测 | uprobe 挂载 `libllama.so` 的 `llama_decode/llama_encode`，检测 prefill vs decode 阶段。可配置每个阶段的 prefetch 范围和是否启用跨 block |
| `prefetch_vllm_phase.bpf.c` | vLLM 阶段检测 | uprobe 挂载 `libcudart.so` 的 `cudaStreamSynchronize` + `paged_attention` 符号。检测 prefill vs decode，per-phase 预取配置 |
| `prefetch_vllm_phase_transparent.bpf.c` | vLLM 透明阶段 | 类似 vllm_phase 但无需指定 uprobe 目标函数，通过 cudaStreamSynchronize 间隔推断阶段 |
| `prefetch_faiss_phase.bpf.c` | FAISS 阶段自适应 | heuristic 阶段检测：在 window=32 次 fault 中统计 +1 stride 数量。stride >= 16 判定为 BUILD 阶段（顺序 prefetch），<= 8 判定为 SEARCH 阶段（保守 prefetch）|
| `prefetch_faiss_uprobe.bpf.c` | FAISS uprobe 阶段 | uprobe 挂载 `_swigfaiss.so` 的 `search_preassigned` 函数，精确检测 search 阶段 |

### Proactive Prefetch（主动预取）

在 fault 发生前主动迁移数据。

| 文件 | 策略 | 实现 |
|------|------|------|
| `prefetch_proactive_layer.bpf.c` | 层级前瞻迁移 | always_max + cycle_moe + bpf_wq 层前瞻。O(1) 层转换检测（无循环），在当前层完成前预迁移下一层的 VA block |
| `prefetch_gnn_proactive.bpf.c` | GNN 主动预取 | uprobe 挂载 `libcudart.so::cudaMallocManaged`，检测 PyTorch 分配。在分配完成后立即用 bpf_wq 预取整个 allocation 到 GPU |
| `prefetch_moe_expert.bpf.c` | MoE bitmap fault replay | uprobe 挂载 `cudaMallocManaged`（检测模型 buffer）+ `cudaStreamSynchronize`（token 边界）。在 gpu_block_activate 中记录每个 token 的 fault bitmap，token 边界 swap bitmap，下一个 token 第一次 fault 时用 bpf_wq replay 上一个 token 的 fault pattern。注意：这是盲目 replay，不是预测性 prefetch |

---

## CPU Scheduling 策略 (sched_ext)

基于 Linux sched_ext 的 GPU 进程感知 CPU 调度。

| 文件 | 策略 | 实现 |
|------|------|------|
| `sched_gpu_serving.bpf.c` | GPU 进程优先调度 | 注册 GPU 进程 PID，放入 GPU_BOOST_DSQ (FIFO, 40ms timeslice)，非 GPU 进程放入 SHARED_DSQ |
| `sched_gpu_xcoord.bpf.c` | GPU 状态自适应 | 读取 GPU 共享状态 map，根据 GPU fault pressure 动态调整 GPU 进程优先级 |
| `sched_gpu_xcoord_noad.bpf.c` | 无自动检测 | 仅对 `-p` 注册的 PID 做优先提升，不自动检测 GPU 进程 |
| `sched_gpu_coord.bpf.c` | FPRS 反馈控制 | Fault-Pressure Regulated Scheduling: LC 进程 fault_rate 作为反馈信号，integral controller 节流 BE 进程 CPU 时间 |
| `sched_gpu_baseline.bpf.c` | 盲目优先提升 | 所有进程无差别优先提升（对照实验用） |
| `sched_gpu_minimal.bpf.c` | 最小 GPU 感知 | 最简 sched_ext baseline，仅实现必要 hook |

---

## Trace 工具

| 文件 | 用途 |
|------|------|
| `prefetch_trace.bpf.c` | 追踪 prefetch 决策过程 (page_index, max_region, pages_accessed) |
| `chunk_trace.bpf.c` | 追踪 chunk 生命周期 (activate, used, eviction_prepare, VA info) |
| `gpu_sched_trace.bpf.c` | 追踪 GPU scheduling 事件 |

运行:
```bash
sudo timeout 30 ./chunk_trace > /tmp/trace.csv
sudo timeout 5 ./prefetch_trace > /tmp/prefetch.csv
```

---

## 使用方法

### 编译

```bash
cd extension
make        # 编译所有 BPF 策略和工具
```

### 加载 Policy

每个策略编译为独立的 userspace loader 二进制文件：

```bash
# 加载组合策略（推荐）
sudo ./prefetch_always_max_cycle_moe

# 跨 block 预取（带参数）
sudo ./prefetch_cross_block_v2 1 2048 0    # evict_mode=1 prefetch_kb=2048 mode=0(direction)

# PID-aware 策略
sudo ./prefetch_pid_tree -p 1234 -P 20 -l 5678 -L 80

# 带 uprobe 的策略
sudo ./prefetch_llama_phase /path/to/libllama.so 0 32 0
```

Loader 自动清理旧 struct_ops、加载 BPF、attach struct_ops。Ctrl-C 退出时自动 detach。

新增的安全 ABI 策略使用 ownership-scoped loader，只销毁本进程创建的
link，不自动清理其他进程的 struct_ops。两条策略的可表达性边界、参数和
engagement metrics 见 [`SAFE_POLICY_ROUTES.md`](SAFE_POLICY_ROUTES.md)。

### 验证

```bash
# 查看已加载的 struct_ops
sudo bpftool map list | grep struct_ops

# 清理残留 struct_ops
sudo ./cleanup_struct_ops_tool
```

---

## BPF 编程注意事项

### 必须遵守

1. **eviction 逻辑放在 `gpu_block_activate`，不是 `gpu_block_access`**。后者从未被调用。
2. **只能用 callback 收到的 `decision_ctx` 请求 reorder**。不要保存该指针，也不能把裸 list 指针传给策略接口。
3. **使用 PERCPU_ARRAY 而非 HASH map** 做频率计数。HASH map 在高 eviction 压力下超时导致 Xid 31。
4. **pointer 不能做算术运算**。BPF verifier 禁止。用 `bpf_probe_read_kernel(&scalar, sizeof(scalar), &ptr)` 转换为标量。
5. **struct_ops/kprobe 在 UVM 内核线程上下文运行**，`bpf_get_current_pid_tgid()` 不返回应用 PID。不能在 struct_ops 中用 PID 过滤。

### Stale Struct_ops 清理

`bpftool struct_ops unregister` 在 kernel 6.15.11 上 segfault。使用 `cleanup_struct_ops_tool` 或 BPF_MAP_DELETE_ELEM syscall：

```python
import ctypes, struct, os, subprocess
libc = ctypes.CDLL("libc.so.6")
result = subprocess.run(["bpftool", "map", "list"], capture_output=True, text=True)
for line in result.stdout.split(chr(10)):
    if "struct_ops" in line:
        map_id = int(line.split(":")[0])
        attr = bytearray(120); struct.pack_into("I", attr, 0, map_id)
        buf = ctypes.create_string_buffer(bytes(attr))
        fd = libc.syscall(321, 14, buf, 120)
        if fd >= 0:
            key = ctypes.c_uint32(0); attr2 = bytearray(120)
            struct.pack_into("I", attr2, 0, fd)
            struct.pack_into("Q", attr2, 8, ctypes.addressof(key))
            buf2 = ctypes.create_string_buffer(bytes(attr2))
            libc.syscall(321, 3, buf2, 120)
            os.close(fd); print(f"Cleaned map {map_id}")
```

### 添加新策略

1. 创建 `extension/my_policy.bpf.c`，实现所有 6 个 struct_ops hook
2. eviction 逻辑放在 `gpu_block_activate`，`gpu_block_access` 返回 0
3. Makefile 中 `BPF_APPS` 已自动发现所有 `.bpf.c` 文件
4. `make my_policy` 编译
5. `sudo ./my_policy` 运行
