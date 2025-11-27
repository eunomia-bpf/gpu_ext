# eBPF 实现 GPU Preempt 控制的可行性分析

## 1. GPreempt Patch 功能分解

GPreempt patch 主要实现了以下功能：

### 1.1 查询 TSG Handles (`NV_ESC_RM_QUERY_GROUP`)

**原始实现**: 新增 ioctl 命令，遍历 RM client 列表查找 TSG

**eBPF 替代方案**:
- **Hook 点**: `kchangrpapiConstruct_IMPL` (TSG 创建时)
- **方案**: 用 kprobe/fentry hook TSG 创建，记录 `hClient`, `hTsg`, `threadId` 到 BPF map

```c
// eBPF 程序
SEC("kprobe/kchangrpapiConstruct_IMPL")
int hook_tsg_create(struct pt_regs *ctx) {
    struct tsg_info info = {};

    // 从函数参数获取 KernelChannelGroupApi 指针
    // 提取 hClient, hTsg
    // 获取当前 threadId
    info.thread_id = bpf_get_current_pid_tgid();

    // 存入 BPF map
    bpf_map_update_elem(&tsg_map, &key, &info, BPF_ANY);
    return 0;
}
```

**可行性**: ✅ **可行**，但需要解析内核数据结构

### 1.2 记录 ThreadId

**原始实现**: 修改 `KernelChannelGroupApi` 结构体添加 `threadId` 字段

**eBPF 替代方案**:
- 不修改结构体，而是用 BPF map 外部记录关联关系

```c
// BPF map: tsg_handle -> thread_id
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, __u64);    // tsg_handle
    __type(value, __u64);  // thread_id
} tsg_thread_map SEC(".maps");
```

**可行性**: ✅ **可行**

### 1.3 触发 Preempt

**现有接口**: `NVA06C_CTRL_CMD_PREEMPT` 已经存在！

**问题**: 需要 valid `hClient` 和 `hTsg`

**eBPF 方案**: 通过上面的 hook 获取 handles 后，用户态直接调用现有 ioctl

**可行性**: ✅ **可行**

### 1.4 RPC 异步化

**原始实现**: 注释掉 `rpcRecvPoll()` 等待

**eBPF 替代方案**: ❌ **不可行**
- eBPF 不能修改函数执行流程
- 不能跳过函数调用

## 2. eBPF 实现方案

### 2.1 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      用户态程序                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ CUDA 应用   │    │ 调度控制器  │    │ BPF 用户态  │     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          │ CUDA API         │ ioctl            │ BPF syscall
          │                  │                  │
┌─────────┼──────────────────┼──────────────────┼─────────────┐
│         ▼                  ▼                  ▼   内核态    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ nvidia.ko   │───▶│ eBPF hooks  │───▶│  BPF maps   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                                     ▲             │
│         │ TSG create/destroy                  │             │
│         └─────────────────────────────────────┘             │
│                     记录 handle 信息                        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 需要 Hook 的函数

```c
// 1. TSG 创建 - 记录 handles
kchangrpapiConstruct_IMPL(KernelChannelGroupApi *pApi, ...)
  → 记录: hClient, hTsg, thread_id, pid

// 2. TSG 销毁 - 清理记录
kchangrpapiDestruct_IMPL(KernelChannelGroupApi *pApi)
  → 删除 map entry

// 3. Channel 加入 TSG - 记录 channel
kchannelCtrlCmdBind_IMPL(...)
  → 记录 channel handle

// 4. TSG 调度 - 可选，用于监控
kchangrpapiCtrlCmdGpFifoSchedule_IMPL(...)
  → 记录调度事件
```

### 2.3 BPF Maps 设计

```c
// Map 1: 按 PID 查找 TSG
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, __u32);           // pid
    __type(value, struct tsg_handles);
    __uint(max_entries, 1024);
} pid_tsg_map SEC(".maps");

struct tsg_handles {
    __u32 hClient;
    __u32 hTsg;
    __u64 thread_id;
    __u32 num_channels;
    __u32 channels[16];
};

// Map 2: 所有活跃 TSG 列表
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, __u64);           // (hClient << 32) | hTsg
    __type(value, struct tsg_info);
    __uint(max_entries, 4096);
} all_tsg_map SEC(".maps");

// Map 3: Ring buffer 用于事件通知
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");
```

### 2.4 用户态控制流程

```c
// 1. 加载 eBPF 程序
struct bpf_object *obj = bpf_object__open_file("gpu_sched.bpf.o", NULL);
bpf_object__load(obj);

// 2. Attach hooks
bpf_program__attach_kprobe(prog_tsg_create, false, "kchangrpapiConstruct_IMPL");
bpf_program__attach_kprobe(prog_tsg_destroy, false, "kchangrpapiDestruct_IMPL");

// 3. 查询 TSG handles (从 BPF map)
int pid_tsg_fd = bpf_object__find_map_fd_by_name(obj, "pid_tsg_map");
struct tsg_handles handles;
bpf_map_lookup_elem(pid_tsg_fd, &target_pid, &handles);

// 4. 触发 preempt (使用现有 ioctl)
int nv_fd = open("/dev/nvidiactl", O_RDWR);
NVA06C_CTRL_PREEMPT_PARAMS preempt = { .bWait = NV_TRUE };
NVOS54_PARAMETERS ctrl = {
    .hClient = handles.hClient,
    .hObject = handles.hTsg,
    .cmd = NVA06C_CTRL_CMD_PREEMPT,
    .params = &preempt,
    .paramsSize = sizeof(preempt),
};
ioctl(nv_fd, NV_ESC_RM_CONTROL, &ctrl);
```

## 3. 挑战与限制

### 3.1 结构体偏移问题

NVIDIA 驱动是闭源的（虽然有 open-gpu-kernel-modules），eBPF 需要知道结构体偏移：

```c
// 需要手动确定偏移量
#define KCHANGRPAPI_HCLIENT_OFFSET  0x??
#define KCHANGRPAPI_HTSG_OFFSET     0x??
```

**解决方案**:
1. 从 open-gpu-kernel-modules 源码分析偏移
2. 使用 BTF (如果驱动有 BTF 信息)
3. 运行时探测

### 3.2 符号访问

NVIDIA 驱动的函数可能不是导出符号：

```bash
# 检查符号是否可用
cat /proc/kallsyms | grep kchangrpapiConstruct
```

**解决方案**:
- 使用 kprobe 按地址 hook
- 或 hook 更上层的 ioctl 入口

### 3.3 安全性

eBPF 程序需要 root 权限或 CAP_BPF。

## 4. 与 GPreempt Patch 对比

| 方面 | GPreempt Patch | eBPF 方案 |
|------|---------------|-----------|
| 侵入性 | 需修改驱动代码 | 无需修改 |
| 部署 | 需重新编译驱动 | 动态加载 |
| 维护 | 跟随驱动版本更新 | 需适配偏移量 |
| 功能完整性 | 100% | ~80% (无法异步 RPC) |
| 性能 | 最优 | 略有开销 |

## 5. 推荐实现步骤

1. **Phase 1**: Hook TSG 创建/销毁，建立 handle 映射
2. **Phase 2**: 实现用户态 preempt 控制
3. **Phase 3**: 添加调度监控和统计
4. **Phase 4**: 实现自动化调度策略

## 6. 示例代码框架

见 `gpu_sched_ebpf.c` 和 `gpu_sched_user.c`
