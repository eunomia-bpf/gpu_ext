# GPU 调度调用图详细分析

基于 cflow 生成的调用图，本文档详细分析 NVIDIA 驱动中 GPU 调度相关函数的调用关系和内部逻辑。

## 一、核心数据结构

### 1.1 KernelChannelGroup (TSG)

TSG (Time Slice Group) 是 GPU 调度的基本单位，包含一组共享调度参数的 channels。

```c
// 关键字段
struct KernelChannelGroup {
    NvU32 grpID;                    // TSG 硬件 ID
    NvU64 timesliceUs;              // 时间片（微秒）
    NvU32 *pInterleaveLevel;        // 交错级别数组（每个 subdevice 一个）
    CHANNEL_LIST *pChannelList;     // channel 链表
    NvU32 runlistId;                // 所属 runlist ID
    // ...
};
```

### 1.2 KernelChannel

Channel 是 GPU 命令提交的通道，每个 channel 属于一个 TSG。

```c
struct KernelChannel {
    NvU32 ChID;                     // Channel 硬件 ID
    KernelChannelGroup *pKernelChannelGroup;  // 所属 TSG
    RM_ENGINE_TYPE engineType;      // 引擎类型 (GR/CE/NVDEC 等)
    NvBool bRunlistSet[];           // 每个 subdevice 的 runlist 状态
    // ...
};
```

### 1.3 Runlist

Runlist 是硬件调度器使用的可调度 TSG 列表。

```
Runlist 结构:
┌─────────────────────────────────────┐
│ Runlist 0 (Graphics)                │
│   ├── TSG 0 (timeslice=1000us)      │
│   ├── TSG 1 (timeslice=2000us)      │
│   └── TSG 2 (timeslice=1000us)      │
├─────────────────────────────────────┤
│ Runlist 1 (Copy Engine)             │
│   └── TSG 3 (timeslice=500us)       │
├─────────────────────────────────────┤
│ Runlist 2 (NVDEC)                   │
│   └── TSG 4 (timeslice=1000us)      │
└─────────────────────────────────────┘
```

---

## 二、TSG 生命周期

### 2.1 TSG 创建 - kchangrpInit_IMPL

**源码位置**: `kernel_channel_group.c:112`

**调用图**:
```
kchangrpInit_IMPL(pGpu, pKernelChannelGroup, pVAS, gfid)
│
├─[1] GPU_GET_KERNEL_FIFO()
│     获取 KernelFifo 对象，后续所有 FIFO 操作都需要它
│
├─[2] gpumgrGetSubDeviceMaxValuePlus1()
│     获取 GPU 子设备数量，用于分配 per-subdevice 数组
│
├─[3] portMemAllocNonPaged()
│     分配 pInterleaveLevel 数组内存
│     大小 = sizeof(NvU32) * subDeviceCount
│
├─[4] kfifoGetDefaultRunlist_HAL()
│     根据引擎类型获取默认 runlist ID
│     Graphics → runlist 0, Copy Engine → runlist 1, etc.
│
├─[5] kfifoIsPerRunlistChramEnabled()
│     检查是否启用 per-runlist Channel RAM
│     影响后续 channel ID 分配策略
│
├─[6] kfifoEngineInfoXlate_HAL()
│     引擎信息转换，将 runlist ID 转换为其他引擎信息
│     例如: RUNLIST_ID → ENGINE_TYPE
│
├─[7] kfifoGetChidMgr()
│     获取 Channel ID Manager
│     每个 runlist 有独立的 ChidMgr
│
├─[8] kfifoChidMgrAllocChannelGroupHwID()
│     ★ 关键操作：分配硬件 TSG ID
│     从 ChidMgr 的空闲 ID 池中分配
│     返回值存入 pKernelChannelGroup->grpID
│
├─[9] kfifoChannelGroupGetDefaultTimeslice_HAL()
│     获取默认时间片值
│     通常为 1000-2000 微秒
│
├─[10] [设置初始调度参数]
│      pKernelChannelGroup->timesliceUs = defaultTimeslice
│      pKernelChannelGroup->pInterleaveLevel[i] = defaultInterleave
│
├─[11] kfifoChannelGroupSetTimeslice()
│      将时间片值写入内部数据结构
│      注意：此时还未写入硬件
│
├─[12] kfifoChannelListCreate()
│      创建空的 channel 链表
│      后续添加的 channel 会加入此链表
│
├─[13] kfifoChannelGroupGetLocalMaxSubcontext_HAL()
│      获取最大子上下文数量
│      用于 MIG (Multi-Instance GPU) 场景
│
├─[14] constructObjEHeap()
│      构建子上下文 ID 分配器（堆）
│
├─[15] mapInit()
│      初始化 channel 映射表
│
├─[16] kchangrpAllocFaultMethodBuffers_HAL()
│      分配 fault method buffers
│      用于处理 GPU 页面错误
│
└─[17] kchangrpMapFaultMethodBuffers_HAL()
       映射 fault method buffers 到 GPU 地址空间
```

**关键逻辑分析**:

1. **硬件 ID 分配** (步骤 8):
   - TSG 的硬件 ID 是唯一的，由 ChidMgr 管理
   - 每个 runlist 有独立的 ID 空间
   - ID 用于硬件调度器识别 TSG

2. **时间片设置** (步骤 9-11):
   - 默认时间片通常由架构决定
   - 时间片决定 TSG 在 GPU 上连续执行的最长时间
   - 时间片耗尽后，硬件调度器会切换到下一个 TSG

3. **内存分配** (步骤 3, 14-17):
   - 需要分配多个数据结构的内存
   - fault method buffers 用于 GPU 缺页处理

### 2.2 TSG 销毁 - kchangrpDestroy_IMPL

**源码位置**: `kernel_channel_group.c:409`

**调用图**:
```
kchangrpDestroy_IMPL(pGpu, pKernelChannelGroup)
│
├─[1] GPU_GET_KERNEL_FIFO()
│
├─[2] kfifoGetNumRunqueues_HAL()
│     获取 runqueue 数量
│
├─[3] kfifoIsPerRunlistChramEnabled()
│
├─[4] kfifoEngineInfoXlate_HAL()
│
├─[5] kfifoGetChidMgr()
│
├─[6] kfifoChannelGroupGetLocalMaxSubcontext_HAL()
│
├─[7] portMemFree()
│     释放 pInterleaveLevel 数组
│
├─[8] mapDestroy()
│     销毁 channel 映射表
│
├─[9] kfifoChannelListDestroy()
│     销毁 channel 链表
│     注意：此时 TSG 中不应有任何 channel
│
├─[10] mapRemove()
│      从全局 TSG 映射中移除
│
├─[11] kfifoChidMgrFreeChannelGroupHwID()
│      ★ 关键操作：释放硬件 TSG ID
│      ID 返回到空闲池，可被新 TSG 使用
│
├─[12] IS_GFID_PF() / gpuIsWarBug200577889SriovHeavyEnabled()
│      虚拟化相关检查
│
├─[13] kchangrpUnmapFaultMethodBuffers_HAL()
│      取消映射 fault method buffers
│
└─[14] kchangrpFreeFaultMethodBuffers_HAL()
       释放 fault method buffers 内存
```

**关键逻辑分析**:

1. **清理顺序**:
   - 先销毁 channel 相关结构
   - 再释放硬件 ID
   - 最后释放内存

2. **前置条件**:
   - TSG 中不能有活跃的 channel
   - TSG 必须已从 runlist 中移除

---

## 三、Channel 管理

### 3.1 Channel 添加到 TSG - kchangrpAddChannel_IMPL

**源码位置**: `kernel_channel_group.c:554`

**调用图**:
```
kchangrpAddChannel_IMPL(pGpu, pKernelChannelGroup, pKernelChannel)
│
├─[1] GPU_GET_KERNEL_FIFO()
│
├─[2] gpumgrGetSubDeviceInstanceFromGpu()
│     获取当前 GPU 的 subdevice 索引
│
├─[3] kfifoGetMaxChannelGroupSize_HAL()
│     获取 TSG 最大 channel 数量
│     通常为 256 或更多
│
├─[4] [检查 TSG 是否已满]
│     if (channelCount >= maxSize) return ERROR
│
├─[5] kchannelIsRunlistSet()
│     检查 channel 是否已设置 runlist
│
├─[6] kchannelGetRunlistId()
│     获取 channel 的 runlist ID
│
├─[7] [检查 runlist 兼容性]
│     channel 的 runlist 必须与 TSG 的 runlist 匹配
│
├─[8] kfifoChannelListAppend()
│     ★ 将 channel 添加到 TSG 的 channel 链表
│
└─[9] kchangrpSetInterleaveLevel()
       设置 channel 的交错级别（继承自 TSG）
```

### 3.2 Channel 从 TSG 移除 - kchangrpRemoveChannel_IMPL

**源码位置**: `kernel_channel_group.c:629`

**调用图**:
```
kchangrpRemoveChannel_IMPL(pGpu, pKernelChannelGroup, pKernelChannel)
│
├─[1] kfifoChannelListRemove()
│     从 TSG 的 channel 链表中移除
│
├─[2] GPU_GET_KERNEL_FIFO()
│
├─[3] kfifoGetNumRunqueues_HAL()
│
└─[4] kchangrpUnmapFaultMethodBuffers_HAL()
       取消映射该 channel 相关的 fault method buffers
```

---

## 四、TSG 调度

### 4.1 TSG 调度启用 - kchangrpapiCtrlCmdGpFifoSchedule_IMPL

**源码位置**: `kernel_channel_group_api.c:1067`

**触发 ioctl**: `NVA06C_CTRL_CMD_GPFIFO_SCHEDULE` (0xa06c0101)

**调用图**:
```
kchangrpapiCtrlCmdGpFifoSchedule_IMPL(pKernelChannelGroupApi, pSchedParams)
│
├─[1] GPU_RES_GET_GPU()
│     从资源句柄获取 GPU 对象
│
├─[2] RES_GET_REF()
│     获取资源引用
│
├─[3] GPU_GET_PHYSICAL_RMAPI()
│     获取物理 RM API 接口
│
├─[4] gpuGetClassByClassId()
│     验证 class ID 有效性
│
├─[5] gpumgrGetSubDeviceInstanceFromGpu()
│
├─[6] [准入控制检查]
│     可以在此决定是否允许调度
│
├─[7] kchannelIsSchedulable_HAL()
│     检查 channel 是否可调度
│     │
│     └── kchannelIsSchedulable_IMPL()
│         ├── kchannelGetGfid()
│         ├── IS_GFID_VF()           // 虚拟化检查
│         ├── kchannelGetEngine_HAL()
│         ├── gvaspaceIsExternallyOwned()
│         └── IS_GR()                // Graphics 引擎检查
│
├─[8] SLI_LOOP_START()
│     开始 SLI 循环（多 GPU 场景）
│
├─[9] GPU_GET_KERNEL_FIFO()
│
├─[10] kchannelIsRunlistSet()
│      检查 channel 的 runlist 是否已设置
│      │
│      └── kchannelIsRunlistSet()
│          └── gpumgrGetSubDeviceInstanceFromGpu()
│              return pKernelChannel->bRunlistSet[subdevInst]
│
├─[11] kchannelGetRunlistId()
│      获取当前 runlist ID
│
├─[12] [如果 runlist 未设置]
│      kfifoGetDefaultRunlist_HAL()
│      获取默认 runlist
│
├─[13] kfifoRunlistSetId_HAL()
│      ★★★ 关键操作：将 TSG 加入 runlist ★★★
│      这是真正启用调度的操作
│      │
│      └── [架构相关实现]
│          例如 Ampere: kfifoRunlistSetId_GA100()
│
├─[14] IS_VIRTUAL() / IS_GSP_CLIENT()
│      虚拟化/GSP 检查
│
└─[15] NV_RM_RPC_CONTROL()
       如果是虚拟化环境，通过 RPC 转发到 Host
```

**关键逻辑分析**:

1. **调度启用流程**:
   ```
   用户态 ioctl → 驱动检查 → 加入 runlist → 硬件开始调度
   ```

2. **kfifoRunlistSetId_HAL 的作用**:
   - 将 TSG 添加到指定 runlist
   - 更新硬件 runlist 数据结构
   - 硬件调度器会从 runlist 中选择 TSG 执行

3. **可调度性检查**:
   - 检查 channel 是否绑定到有效引擎
   - 检查地址空间是否正确配置
   - 虚拟化场景的额外检查

### 4.2 Runlist 设置详细流程

**调用链**:
```
kfifoRunlistSetId_HAL()
│
└── [Ampere 实现] kfifoRunlistSetId_GA100()
    │
    ├─ 获取 runlist 基地址
    │
    ├─ 构建 runlist entry:
    │   ├─ TSG ID
    │   ├─ 时间片
    │   └─ 交错级别
    │
    ├─ 写入 runlist 内存
    │
    └─ 触发 runlist 更新
        └─ GPU_REG_WR32(NV_PFIFO_RUNLIST_BASE, ...)
```

---

## 五、时间片和优先级控制

### 5.1 设置时间片 - kchangrpapiCtrlCmdSetTimeslice_IMPL

**源码位置**: `kernel_channel_group_api.c:1316`

**触发 ioctl**: `NVA06C_CTRL_CMD_SET_TIMESLICE` (0xa06c0103)

**调用图**:
```
kchangrpapiCtrlCmdSetTimeslice_IMPL(pKernelChannelGroupApi, pTsParams)
│
├─[1] GPU_RES_GET_GPU()
│
├─[2] RES_GET_REF()
│
├─[3] GPU_GET_PHYSICAL_RMAPI()
│
├─[4] gpuGetClassByClassId()
│
├─[5] [参数验证]
│     检查时间片值是否在有效范围内
│     通常: 100us - 100000us
│
├─[6] IS_VIRTUAL() / IS_GSP_CLIENT()
│
└─[7] NV_RM_RPC_CONTROL()
       RPC 到 GSP/Host 实际设置时间片
       │
       └── [GSP 端处理]
           ├─ 更新 TSG 的 timesliceUs 字段
           └─ 如果 TSG 在 runlist 中，更新 runlist entry
```

**时间片的影响**:
```
时间片 = 1000us (1ms):
TSG A ────[1ms]──┐    ┌────[1ms]──┐
                 │    │           │
TSG B ───────────┴[1ms]───────────┴[1ms]──

时间片 = 2000us (2ms):
TSG A ────────[2ms]─────────┐    ┌────────[2ms]───────
                            │    │
TSG B ──────────────────────┴[1ms]────────────────────
```

### 5.2 设置交错级别 - kchangrpapiCtrlCmdSetInterleaveLevel_IMPL

**源码位置**: `kernel_channel_group_api.c:1398`

**触发 ioctl**: `NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL` (0xa06c0107)

**调用图**:
```
kchangrpapiCtrlCmdSetInterleaveLevel_IMPL(pKernelChannelGroupApi, pParams)
│
├─[1] GPU_RES_GET_GPU()
│
├─[2] RES_GET_REF()
│
├─[3] gpuGetClassByClassId()
│
├─[4] [参数验证]
│     level 必须是 LOW(1), MEDIUM(2), 或 HIGH(3)
│
├─[5] IS_VIRTUAL() / IS_GSP_CLIENT()
│     │
│     └─ NV_RM_RPC_CONTROL()
│
├─[6] [本地路径]
│     NV_CHECK_OR_RETURN()
│     验证 level 有效性
│
└─[7] kchangrpSetInterleaveLevel()
       │
       └── kchangrpSetInterleaveLevel_IMPL()
           │
           ├─ SLI_LOOP_START()
           │  遍历所有 subdevice
           │
           ├─ gpumgrGetSubDeviceInstanceFromGpu()
           │
           ├─ pKernelChannelGroup->pInterleaveLevel[subdevInst] = value
           │  更新内存中的值
           │
           └─ kchangrpSetInterleaveLevelSched()
              通知调度器更新
```

**交错级别的影响**:
```
Runlist 中有 3 个 TSG，交错级别分别为 HIGH, MEDIUM, LOW:

HIGH   ──[执行]──[执行]──[执行]──────────────[执行]──[执行]──[执行]──
MEDIUM ──────────[执行]──[执行]──────────────────────[执行]──[执行]──
LOW    ────────────────────────[执行]────────────────────────────────

HIGH 获得更多执行机会，LOW 获得较少执行机会
```

---

## 六、Work Submit Token

### 6.1 获取 Token - kchannelCtrlCmdGpfifoGetWorkSubmitToken_IMPL

**源码位置**: `kernel_channel.c:3256`

**触发 ioctl**: `NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN` (0xc36f0108)

**调用图**:
```
kchannelCtrlCmdGpfifoGetWorkSubmitToken_IMPL(pKernelChannel, pTokenParams)
│
├─[1] GPU_RES_GET_GPU()
│
├─[2] GPU_GET_KERNEL_FIFO()
│
├─[3] resservGetTlsCallContext()
│     获取线程本地存储的调用上下文
│
├─[4] IS_MIG_ENABLED()
│     检查是否启用 MIG
│
├─[5] IS_VIRTUAL() / IS_VIRTUAL_WITH_SRIOV()
│     虚拟化检查
│     │
│     ├─ [虚拟化路径]
│     │  kfifoIsPerRunlistChramEnabled()
│     │  NV_RM_RPC_CONTROL()
│     │  RPC 到 Host 获取 token
│     │
│     └─ [本地路径] 继续执行
│
├─[6] kfifoGenerateWorkSubmitToken()
│     ★ 生成 work submit token
│     │
│     └── kfifoGenerateWorkSubmitTokenHal_GA100() [Ampere]
│         │
│         ├─ vgpuGetCallingContextGfid()
│         │
│         ├─ IS_GFID_VF()
│         │  虚拟化场景需要转换 channel ID
│         │
│         ├─ kfifoGetVChIdForSChId_HAL()
│         │  获取虚拟 channel ID
│         │
│         ├─ kchannelGetEngineType()
│         │
│         ├─ kchannelIsRunlistSet()
│         │  检查 runlist 是否已设置
│         │
│         ├─ kchannelGetRunlistId()
│         │
│         └─ FLD_SET_DRF_NUM()
│            组合 token:
│            token = (runlist_id << 16) | channel_id
│
└─[7] kchannelNotifyWorkSubmitToken()
       通知 token 已生成
       │
       ├─ FLD_SET_DRF() / FLD_SET_DRF_NUM()
       │  设置 notifier 字段
       │
       └─ kchannelUpdateNotifierMem()
          更新 notifier 内存
```

**Token 结构**:
```
Work Submit Token (32-bit):
┌─────────────────┬─────────────────┐
│  Runlist ID     │   Channel ID    │
│   (16 bits)     │    (16 bits)    │
└─────────────────┴─────────────────┘

例如: runlist=0, channel=5 → token = 0x00000005
      runlist=1, channel=3 → token = 0x00010003
```

**Token 用途**:
- 用于追踪 GPU 工作完成状态
- 用户态可以通过 notifier 内存查询 token 对应的工作是否完成
- 通常在需要同步（如 cudaDeviceSynchronize）前获取

---

## 七、硬件 Preemption

### 7.1 Channel Halt 启动 - kfifoStartChannelHalt_GA100

**源码位置**: `kernel_fifo_ga100.c:877`

**调用图**:
```
kfifoStartChannelHalt_GA100(pGpu, pKernelFifo, pKernelChannel)
│
├─[1] kchannelGetRunlistId()
│     获取 channel 所属的 runlist ID
│
├─[2] kfifoEngineInfoXlate_HAL()
│     将 runlist ID 转换为其他引擎信息
│     │
│     └── kfifoEngineInfoXlate_GA100()
│         │
│         ├─ GPU_GET_KERNEL_GRAPHICS_MANAGER()
│         │
│         ├─ IS_MIG_IN_USE()
│         │  MIG 模式下需要特殊处理
│         │
│         └─ kfifoEngineInfoXlate_GV100()
│            基础实现
│
├─[3] [计算寄存器地址]
│     chramPriBase = NV_CHRAM_BASE + (runlistId * stride)
│     runlistPriBase = NV_RUNLIST_BASE + (runlistId * stride)
│
├─[4] FLD_SET_DRF(_CHRAM, _CHANNEL, _ENABLE, _IN_USE, channelVal)
│     设置 channel 状态为 IN_USE（禁用）
│
├─[5] ★ GPU_REG_WR32(chramPriBase + NV_CHRAM_CHANNEL(ChID), channelVal)
│     写 Channel RAM 寄存器，禁用 channel
│
│     寄存器布局:
│     NV_CHRAM_CHANNEL:
│     ┌─────────────────────────────────────┐
│     │ ENABLE [1:0]                        │
│     │   0 = FALSE (禁用)                   │
│     │   1 = TRUE (启用)                    │
│     │   2 = IN_USE (正在禁用中)            │
│     └─────────────────────────────────────┘
│
├─[6] FLD_SET_DRF(_RUNLIST, _PREEMPT, _TYPE, _RUNLIST, 0)
│     设置 preempt 类型为 RUNLIST
│
└─[7] ★ GPU_REG_WR32(runlistPriBase + NV_RUNLIST_PREEMPT, runlistVal)
       写 Runlist Preempt 寄存器，触发硬件 preemption

       寄存器布局:
       NV_RUNLIST_PREEMPT:
       ┌─────────────────────────────────────┐
       │ TYPE [1:0]                          │
       │   0 = RUNLIST (preempt 整个 runlist) │
       │   1 = TSG (preempt 单个 TSG)         │
       │   2 = CHANNEL (preempt 单个 channel) │
       └─────────────────────────────────────┘
```

### 7.2 Channel Halt 完成检查 - kfifoCompleteChannelHalt_GA100

**源码位置**: `kernel_fifo_ga100.c:925`

**调用图**:
```
kfifoCompleteChannelHalt_GA100(pGpu, pKernelFifo, pKernelChannel, pTimeout)
│
├─[1] kchannelGetRunlistId()
│
├─[2] kfifoEngineInfoXlate_HAL()
│
├─[3] [轮询循环]
│     while (!timeout) {
│     │
│     ├─ gpuCheckTimeout()
│     │  检查是否超时
│     │
│     ├─ GPU_REG_RD32(runlistPriBase + NV_RUNLIST_PREEMPT)
│     │  读取 preempt 状态寄存器
│     │
│     └─ FLD_TEST_DRF(_RUNLIST, _PREEMPT, _PENDING, _FALSE, regVal)
│        检查 PENDING 位是否为 FALSE
│        │
│        ├─ PENDING = TRUE: preemption 仍在进行
│        │  继续轮询
│        │
│        └─ PENDING = FALSE: preemption 完成
│           退出循环
│     }
│
└─[4] [返回结果]
       成功: NV_OK
       超时: NV_ERR_TIMEOUT
```

**Preemption 时序图**:
```
时间 ──────────────────────────────────────────────────────────►

CPU:  写 CHRAM ──► 写 PREEMPT ──► 轮询 PENDING ─────────► 完成
                       │                  │
                       ▼                  │
GPU:            [收到 preempt 请求]       │
                       │                  │
                       ▼                  │
                [保存当前上下文]           │
                       │                  │
                       ▼                  │
                [停止当前 TSG]             │
                       │                  │
                       ▼                  │
                [清除 PENDING 位] ─────────┘
```

### 7.3 Preemption 级别

GPU 支持多种 preemption 级别：

```
┌─────────────────────────────────────────────────────────────┐
│                    Preemption 级别                          │
├─────────────┬───────────────────────────────────────────────┤
│ RUNLIST     │ Preempt 整个 runlist 上的所有 TSG             │
│             │ 最重量级，用于紧急情况                         │
├─────────────┼───────────────────────────────────────────────┤
│ TSG         │ Preempt 单个 TSG                              │
│             │ 常用于正常调度切换                             │
├─────────────┼───────────────────────────────────────────────┤
│ CHANNEL     │ Preempt 单个 channel                          │
│             │ 最细粒度控制                                   │
└─────────────┴───────────────────────────────────────────────┘
```

---

## 八、Doorbell 机制

### 8.1 用户态 Doorbell（Kernel Launch 路径）

用户态 kernel launch 直接 bypass 内核：

```
用户态 (libcuda.so):
┌─────────────────────────────────────────────────────────────┐
│ cudaLaunchKernel() {                                        │
│     // 1. 准备 GPU 命令                                      │
│     build_gpu_commands(&commands);                          │
│                                                             │
│     // 2. 写入 pushbuffer（GPU 映射的用户态内存）             │
│     memcpy(pushbuffer_ptr, &commands, size);                │
│                                                             │
│     // 3. 写 doorbell 寄存器（用户态直接 MMIO）              │
│     *doorbell_ptr = work_submit_token;                      │
│                                                             │
│     // 以上操作完全不经过内核！                               │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
            │
            │ (直接访问 GPU 映射内存)
            ▼
┌─────────────────────────────────────────────────────────────┐
│ GPU 硬件                                                     │
│                                                             │
│   Doorbell 寄存器 ──► FIFO 引擎 ──► 读取 Pushbuffer         │
│                                          │                  │
│                                          ▼                  │
│                                    执行 GPU 命令             │
└─────────────────────────────────────────────────────────────┘
```

**关键点**:
- Pushbuffer 是 GPU 可访问的内存，用户态直接写入
- Doorbell 是 GPU 寄存器，用户态通过 MMIO 直接写入
- GPU 收到 doorbell 后从 pushbuffer 读取命令执行
- **整个过程不经过内核**，因此无法在驱动层追踪

### 8.2 内核态 Doorbell（仅内核触发）

**源码位置**: `kernel_fifo_ga100.c:966`

**调用图**:
```
kfifoRingChannelDoorBell_GA100(pGpu, pKernelFifo, pKernelChannel)
│
├─[1] kfifoGenerateWorkSubmitToken()
│     生成 work submit token
│
├─[2] [用户模式路径]
│     │
│     ├─ kfifoUpdateUsermodeDoorbell_HAL()
│     │  │
│     │  └── kfifoUpdateUsermodeDoorbell_GA100()
│     │      │
│     │      └─ GPU_VREG_WR32()
│     │         写虚拟 doorbell 寄存器
│     │
│     └─ kfifoGenerateInternalWorkSubmitToken_HAL()
│
└─[3] [内部路径]
       │
       └─ kfifoUpdateInternalDoorbellForUsermode_HAL()
          更新内部 doorbell
```

**重要说明**:
- `kfifoRingChannelDoorBell_GA100` 只在**内核态**代码需要触发 GPU 工作时使用
- 用户态的 `cudaLaunchKernel` **不会调用**这个函数
- 用户态直接通过 MMIO 写 doorbell 寄存器

### 8.3 Usermode Doorbell 寄存器映射

```
kfifoGetUsermodeMapInfo_GV100() [kernel_fifo_gv100.c]
│
├─ 返回 usermode 寄存器的 offset 和 size
│
└─ 用户态 mmap 这个区域后可以直接访问 doorbell 寄存器

用户态地址空间:
┌─────────────────────────────────────────┐
│  ...                                    │
├─────────────────────────────────────────┤
│  USERD Memory (pushbuffer 区域)          │
│  用户态直接写 GPU 命令                    │
├─────────────────────────────────────────┤
│  Doorbell Register (mmap 的 MMIO)        │
│  用户态直接写触发 GPU                     │
├─────────────────────────────────────────┤
│  Notifier Memory                         │
│  GPU 写入完成状态，用户态读取             │
└─────────────────────────────────────────┘
```

---

## 九、Channel 可调度性检查

### 9.1 kchannelIsSchedulable_IMPL

**源码位置**: `kernel_channel.c:2149`

**调用图**:
```
kchannelIsSchedulable_IMPL(pGpu, pKernelChannel)
│
├─[1] kchannelGetGfid()
│     获取 Guest Function ID (虚拟化)
│
├─[2] IS_GFID_VF()
│     检查是否是虚拟机
│
├─[3] dynamicCast()
│     获取 TSG 对象
│
├─[4] IS_MODS_AMODEL()
│     检查是否是模拟器模式
│
├─[5] kchannelGetEngine_HAL()
│     获取 channel 绑定的引擎
│
├─[6] gvaspaceIsExternallyOwned()
│     检查地址空间是否外部管理
│
└─[7] IS_GR()
       检查是否是 Graphics 引擎

返回条件:
- channel 必须属于一个 TSG
- channel 必须绑定到有效引擎
- 地址空间必须正确配置
- 虚拟化场景有额外限制
```

---

## 十、FIFO 控制操作

### 10.1 禁用 Channels - subdeviceCtrlCmdFifoDisableChannels_IMPL

**源码位置**: `kernel_fifo_ctrl.c:713`

**触发 ioctl**: `NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS` (0x2080110b)

**调用图**:
```
subdeviceCtrlCmdFifoDisableChannels_IMPL(pSubdevice, pDisableChannelParams)
│
├─[1] GPU_RES_GET_GPU()
│
├─[2] resservGetTlsCallContext()
│
├─[3] [权限检查]
│     if (pRunlistPreemptEvent != NULL && privLevel < KERNEL)
│         return INSUFFICIENT_PERMISSIONS
│     只有内核级客户端可以使用 preempt event
│
├─[4] IS_VIRTUAL() / IS_GSP_CLIENT()
│
└─[5] NV_RM_RPC_CONTROL()
       RPC 到 GSP/Host 执行实际禁用
       │
       └── [GSP 端]
           ├─ 遍历 channel 列表
           ├─ 对每个 channel 执行 halt
           └─ 等待 preemption 完成
```

### 10.2 Idle Channels - deviceCtrlCmdFifoIdleChannels_IMPL

**源码位置**: `kernel_fifo_ctrl.c:103`

**调用图**:
```
deviceCtrlCmdFifoIdleChannels_IMPL(pDevice, pParams)
│
├─[1] gpuGetInstance()
│
├─[2] gpumgrGetGrpMaskFromGpuInst()
│
├─[3] rmGpuGroupLockAcquire()
│     获取 GPU 组锁
│
├─[4] gpumgrGetGpu()
│
├─[5] GPU_GET_KERNEL_FIFO()
│
├─[6] portMemAllocNonPaged()
│     分配临时内存
│
├─[7] kfifoIdleChannelsPerDevice_HAL()
│     实际执行 idle 操作
│
├─[8] portMemFree()
│
└─[9] rmGpuGroupLockRelease()
       释放 GPU 组锁
```

---

## 十一、关键寄存器总结

| 寄存器 | 功能 | 访问者 |
|-------|------|-------|
| `NV_CHRAM_CHANNEL` | Channel 使能状态 | 驱动写，禁用 channel |
| `NV_RUNLIST_PREEMPT` | 触发 preemption | 驱动写，触发 preempt |
| `NV_RUNLIST_BASE` | Runlist 基地址 | 驱动写，更新 runlist |
| `NV_PFIFO_RUNLIST_*` | FIFO runlist 控制 | 驱动写，配置调度 |
| Usermode Doorbell | 触发 GPU 工作 | 用户态直接写 |

---

## 十二、调度流程总结

```
                        TSG 生命周期
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
    [创建阶段]          [运行阶段]           [销毁阶段]
         │                   │                   │
         ▼                   ▼                   ▼
  kchangrpInit_IMPL   Schedule/Preempt    kchangrpDestroy_IMPL
         │                   │                   │
         ▼                   ▼                   ▼
  分配 TSG ID        加入/移出 Runlist      释放 TSG ID
         │                   │                   │
         ▼                   ▼                   ▼
  设置调度参数        硬件调度执行          清理资源
  (timeslice,              │
   interleave)             ▼
                    ┌──────────────┐
                    │ GPU 硬件调度  │
                    │              │
                    │ 轮询 Runlist  │
                    │ 选择 TSG     │
                    │ 分配时间片    │
                    │ 执行 channel │
                    └──────────────┘
```
