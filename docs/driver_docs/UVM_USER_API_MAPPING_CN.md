# UVM 用户态 API 到内核态的映射关系

本文档旨在阐述用于管理统一虚拟内存（UVM）的用户态 CUDA API（例如 `cudaMemAdvise` 和 `cudaMemPrefetchAsync`）是如何映射到 NVIDIA Open Kernel Modules 中的底层内核态实现的。

## 1. 实现机制: `ioctl`

用户态的 CUDA 驱动与内核态的 UVM 驱动（`nvidia-uvm.ko`）之间的通信主要通过 `ioctl` (输入/输出控制) 系统调用来处理。一个用户态 API 调用会导致 CUDA 驱动打开 `/dev/nvidia-uvm` 设备文件，并发出一个带有特定命令码和指向参数结构体的指针的 `ioctl` 请求。

这些命令在内核中的核心入口点是位于 `kernel-open/nvidia-uvm/uvm.c` 文件中的 `uvm_ioctl` 函数。该函数使用一个 `switch` 语句将命令分发到相应的内部内核函数。

---

## 2. `cudaMemAdvise` 的映射

`cudaMemAdvise` 是一个基于策略的 API，它向驱动提供关于某个内存范围将如何被使用的“提示”。它不会主动触发数据迁移，而是设置影响未来迁移决策的策略。不同的“建议”类型会映射到不同的 `ioctl` 命令。

### 2.1. 核心 `ioctl` 命令与参数

相关的 `ioctl` 命令及其对应的参数结构体定义在 `kernel-open/nvidia-uvm/uvm_ioctl.h` 中。

#### a) `cudaMemAdviseSetPreferredLocation`

*   **`ioctl` 命令**: `UVM_SET_PREFERRED_LOCATION`
*   **参数结构体**:
    ```c
    typedef struct
    {
        NvU64           requestedBase      NV_ALIGN_BYTES(8); // IN
        NvU64           length             NV_ALIGN_BYTES(8); // IN
        NvProcessorUuid preferredLocation;                    // IN
        NvS32           preferredCpuNumaNode;                 // IN
        NV_STATUS       rmStatus;                             // OUT
    } UVM_SET_PREFERRED_LOCATION_PARAMS;
    ```

#### b) `cudaMemAdviseSetReadMostly`

*   **`ioctl` 命令**: `UVM_ENABLE_READ_DUPLICATION`
*   **参数结构体**:
    ```c
    typedef struct
    {
        NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
        NvU64     length        NV_ALIGN_BYTES(8); // IN
        NV_STATUS rmStatus;                        // OUT
    } UVM_ENABLE_READ_DUPLICATION_PARAMS;
    ```

#### c) `cudaMemAdviseSetAccessedBy`

*   **`ioctl` 命令**: `UVM_SET_ACCESSED_BY`
*   **参数结构体**:
    ```c
    typedef struct
    {
        NvU64           requestedBase   NV_ALIGN_BYTES(8); // IN
        NvU64           length          NV_ALIGN_BYTES(8); // IN
        NvProcessorUuid accessedByUuid;                    // IN
        NV_STATUS       rmStatus;                          // OUT
    } UVM_SET_ACCESSED_BY_PARAMS;
    ```

### 2.2. 内核调用链与逻辑

这些操作的逻辑位于 `kernel-open/nvidia-uvm/uvm_policy.c` 中。

1.  `uvm.c` 中的 `uvm_ioctl` 函数将命令分发到对应的 `uvm_api_*` 函数（例如 `uvm_api_set_preferred_location`）。
2.  `uvm_api_*` 函数会验证参数并获取 VA 空间锁。
3.  **`split_span_as_needed`**: 这个关键的辅助函数被调用。它会遍历请求的地址范围内的现有 `uvm_va_range_managed_t` 对象。如果某个范围的当前策略与正在应用的新策略不匹配，它会分割该范围。这确保了新策略能够以页的粒度精确应用，而不会影响指定范围之外的内存。
4.  **策略应用**: 接着，代码会遍历新分割出的范围及其对应的 `uvm_va_block_t` 对象。
5.  它会更新 `uvm_va_range_managed_t` 结构体中的 `policy` 字段。
6.  最后，它可能会触发页表的更新，例如从远程处理器上取消内存映射（`preferred_location_unmap_remote_pages`）或添加新的映射（`uvm_va_block_add_mappings`）。

总而言之，`cudaMemAdvise` 被转换为在内核中设置一个策略对象，这可能进而导致驱动调整内存映射以强制执行该策略。

---

## 3. `cudaMemPrefetchAsync` 的映射

`cudaMemPrefetchAsync` 是一个主动操作，它明确告诉驱动将一个内存范围迁移到特定的目标处理器。

### 3.1. 核心 `ioctl` 命令与参数

*   **`ioctl` 命令**: `UVM_MIGRATE`
*   **参数结构体**: 定义在 `kernel-open/nvidia-uvm/uvm_ioctl.h` 中。
    ```c
    typedef struct
    {
        NvU64           base               NV_ALIGN_BYTES(8); // IN
        NvU64           length             NV_ALIGN_BYTES(8); // IN
        NvProcessorUuid destinationUuid;                      // IN
        NvU32           flags;                                // IN (例如 UVM_MIGRATE_FLAG_ASYNC)
        NvU64           semaphoreAddress   NV_ALIGN_BYTES(8); // IN (用于异步完成通知)
        NvU32           semaphorePayload;                     // IN
        NvS32           cpuNumaNode;                          // IN
        ...
        NV_STATUS       rmStatus;                             // OUT
    } UVM_MIGRATE_PARAMS;
    ```

### 3.2. 内核调用链与逻辑

其实现主要位于 `kernel-open/nvidia-uvm/uvm_migrate.c` 中。

1.  `uvm_ioctl` 函数将 `UVM_MIGRATE` 命令分发给 `uvm_api_migrate`。
2.  `uvm_api_migrate` 验证参数并识别目标处理器。然后它调用内部辅助函数 `uvm_migrate`。
3.  `uvm_migrate` 辅助函数负责协调整个迁移过程，可能会为了性能采用两阶段方法：首先使内存驻留，然后添加映射。它通过 `uvm_migrate_ranges` 遍历地址范围。
4.  调用链最终会下降到对每一个 `uvm_va_block_t` 调用 `uvm_va_block_migrate_locked`。
5.  **`uvm_va_block_make_resident`**: 这是“干活”的函数。它负责：
    *   在目标处理器上分配物理页面。
    *   为**拷贝引擎 (Copy Engine, CE)** 构建一个命令列表（一个 "push"）。
    *   将这个 push 提交到一个硬件通道，以执行从源到目标的物理内存拷贝。
6.  **映射更新**: 拷贝操作发起后，`block_migrate_add_mappings` 会被调用以更新页表，从而允许目标处理器访问新近驻留的内存。
7.  **异步通知**: 如果调用是异步的并且提供了一个信号量，那么这项工作会由一个 `uvm_tracker_t` 来跟踪。当 GPU 拷贝完成后，内核会调用 `semaphore_release`，该函数会向用户提供的信号量地址写入数据，以示完成。
