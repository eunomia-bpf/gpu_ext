# UVM 驱逐策略模块化改造与可扩展框架设计

**版本**: 1.0
**日期**: 2025-11-19

---

## 1. 概述

本文档旨在为 NVIDIA UVM 驱动设计一个全新的、可扩展的页面驱逐策略框架。

当前 UVM 中的驱逐策略（一种类似 FIFO 的 LRU 实现）是硬编码在物理内存管理器 (PMM) 中的，这限制了新算法的快速原型设计、测试和集成。为了解决这一问题，我们提出一个基于回调函数（函数指针）的模块化框架。该框架旨在将**策略逻辑**与**内存管理核心机制**完全解耦，从而允许开发者以“可插拔”的方式实现、切换和评估多种不同的驱逐算法。

## 2. 设计目标

*   **模块化 (Modularity)**: 将每种驱逐算法（如 LRU, FIFO, Clock, LFU 等）封装在独立的、自包含的模块中。
*   **可扩展性 (Extensibility)**: 能够轻松添加新的、未来可能出现的驱逐算法，而无需修改 PMM 的核心代码。
*   **灵活性 (Flexibility)**: 允许在系统运行时（通过内核模块参数）动态选择要使用的驱逐策略，便于性能分析和算法比较。
*   **最小侵入 (Minimal Intrusion)**: 框架的集成应尽可能少地改动现有的 PMM 核心流程。

## 3. 核心设计：可插拔的策略框架

框架的核心是定义一个标准的策略操作接口 (`uvm_eviction_policy_ops`)，所有具体的算法都必须实现这个接口。PMM 则通过一个指针来调用当前激活的策略。

### 3.1. 策略操作接口 (`uvm_eviction_policy_ops`)

我们建议在 `uvm_pmm_gpu.h` 中定义以下接口结构体：

```c
// In uvm_pmm_gpu.h

/**
 * @struct uvm_eviction_policy_ops
 * @brief 定义了驱逐策略必须实现的一组回调函数。
 *
 * 这个接口抽象了驱逐策略的所有核心操作，使得 PMM 可以调用任何
 * 实现了此接口的算法。
 */
struct uvm_eviction_policy_ops {
    /** @brief 策略的唯一名称 (例如, "lru", "fifo")，用于日志和选择。*/
    const char *name;

    /**
     * @brief 初始化策略。
     * 在 PMM 初始化时调用。可以用于分配策略所需的私有数据
     * (例如，为 Clock 算法分配时钟指针的存储空间)。
     *
     * @param pmm 当前 GPU 的 PMM 实例。
     * @return NV_OK 表示成功，否则返回错误码。
     */
    int (*init)(struct uvm_pmm_gpu_struct *pmm);

    /**
     * @brief 销毁策略。
     * 在 PMM 销毁时调用，用于释放 init 期间分配的资源。
     *
     * @param pmm 当前 GPU 的 PMM 实例。
     */
    void (*deinit)(struct uvm_pmm_gpu_struct *pmm);

    /**
     * @brief 当一个新的 root chunk 被 PMM 管理时调用。
     * 策略应使用此回调将 chunk 加入其内部跟踪数据结构中。
     *
     * @param pmm 当前 GPU 的 PMM 实例。
     * @param root_chunk 要添加的 root chunk。
     */
    void (*add)(struct uvm_pmm_gpu_struct *pmm, struct uvm_gpu_root_chunk_t *root_chunk);

    /**
     * @brief 当一个 root chunk 永久离开 PMM 管理时调用。
     * 策略应使用此回调将其从内部跟踪中移除。
     *
     * @param pmm 当前 GPU 的 PMM 实例。
     * @param root_chunk 要移除的 root chunk。
     */
    void (*remove)(struct uvm_pmm_gpu_struct *pmm, struct uvm_gpu_root_chunk_t *root_chunk);

    /**
     * @brief 当一个 chunk 被“触碰”时调用。
     * “触碰”通常指一次成功的分配或一次有效的 GPU 访问。
     * 这是实现 LRU、LFU 等访问敏感性策略的核心回调。
     *
     * @param pmm 当前 GPU 的 PMM 实例。
     * @param root_chunk 被触碰的 root chunk。
     */
    void (*touch)(struct uvm_pmm_gpu_struct *pmm, struct uvm_gpu_root_chunk_t *root_chunk);

    /**
     * @brief 当一个 chunk 变得不可驱逐时（例如被 Pinned）调用。
     * 策略需要将此 chunk 暂时从可驱逐集合中移除。
     *
     * @param pmm 当前 GPU 的 PMM 实例。
     * @param root_chunk 变得不可驱逐的 chunk。
     */
    void (*unevictable)(struct uvm_pmm_gpu_struct *pmm, struct uvm_gpu_root_chunk_t *root_chunk);

    /**
     * @brief 当一个 chunk 重新变为可驱逐时调用。
     *
     * @param pmm 当前 GPU 的 PMM 实例。
     * @param root_chunk 重新变为可驱逐的 chunk。
     */
    void (*evictable)(struct uvm_pmm_gpu_struct *pmm, struct uvm_gpu_root_chunk_t *root_chunk);

    /**
     * @brief 策略的核心：选择一个牺牲者进行驱逐。
     * PMM 在需要释放内存时调用此函数。
     *
     * @param pmm 当前 GPU 的 PMM 实例。
     * @return 返回被选中的 root chunk，如果没有可驱逐的 chunk，则返回 NULL。
     */
    struct uvm_gpu_root_chunk_struct* (*pick_to_evict)(struct uvm_pmm_gpu_struct *pmm);
};
```

### 3.2. 集成到 PMM

1.  **扩展 `uvm_pmm_gpu_t` 结构体**:
    在 `uvm_pmm_gpu.h` 中，修改 `uvm_pmm_gpu_t` 以包含对策略的引用。
    ```c
    struct uvm_pmm_gpu_struct {
        // ... 其他字段 ...

        /** @brief 指向当前激活的驱逐策略操作集。*/
        const struct uvm_eviction_policy_ops *policy_ops;

        /** @brief 指向策略的私有上下文数据 (例如 Clock 指针、LFU 的堆)。*/
        void *policy_context;
    };
    ```

2.  **重构 PMM 核心函数**:
    将 `uvm_pmm_gpu.c` 中硬编码的列表操作替换为对策略回调的调用。

    *   **`pick_root_chunk_to_evict()`**
        *   **旧逻辑**: 直接从 `pmm->root_chunks.va_block_used` 或 `va_block_unused` 列表头部获取。
        *   **新逻辑**: `return pmm->policy_ops->pick_to_evict(pmm);`

    *   **`chunk_update_lists_locked()`**
        *   **旧逻辑**: `list_move_tail(&root_chunk->chunk.list, &pmm->root_chunks.va_block_used);`
        *   **新逻辑**: 这个函数的大部分逻辑将被分散到各个回调中。例如，在 chunk 被分配后 (`uvm_pmm_gpu_unpin_allocated` 函数内)，应该调用 `pmm->policy_ops->touch(pmm, root_chunk);`

    *   **`uvm_pmm_gpu_init()` 和 `uvm_pmm_gpu_deinit()`**
        *   在初始化时，根据模块参数选择策略并调用其 `init` 函数。
        *   在销毁时，调用 `deinit` 函数。

    *   **Chunk 固定与解除**
        *   当 chunk 被 pin (`chunk_pin`) 或 unpin (`chunk_unpin`) 时，调用 `policy_ops->unevictable()` 和 `policy_ops->evictable()`，让策略动态调整其可驱逐集合。

### 3.3. 动态策略选择

通过内核模块参数，可以在加载 UVM 驱动时指定要使用的策略。

1.  **定义模块参数** (例如在 `uvm_main.c` 中):
    ```c
    static char *g_uvm_eviction_policy_name = "lru"; // 默认策略
    module_param(g_uvm_eviction_policy_name, charp, S_IRUGO);
    MODULE_PARM_DESC(g_uvm_eviction_policy_name, "Select eviction policy: 'lru', 'fifo', 'clock'.");
    ```

2.  **在 `uvm_pmm_gpu_init` 中选择并初始化策略**:
    ```c
    // 伪代码
    NV_STATUS uvm_pmm_gpu_init(uvm_pmm_gpu_t *pmm) {
        // ...
        // 定义一个全局的策略数组
        const struct uvm_eviction_policy_ops *available_policies[] = {
            &g_uvm_policy_lru,
            &g_uvm_policy_fifo,
            &g_uvm_policy_clock,
        };

        // 查找并设置策略
        pmm->policy_ops = NULL;
        for (i = 0; i < ARRAY_SIZE(available_policies); i++) {
            if (strcmp(g_uvm_eviction_policy_name, available_policies[i]->name) == 0) {
                pmm->policy_ops = available_policies[i];
                break;
            }
        }

        if (!pmm->policy_ops) {
            // 如果未找到，则使用默认策略
            pmm->policy_ops = &g_uvm_policy_lru;
            UVM_WARN_PRINT("Eviction policy '%s' not found, defaulting to '%s'.\n",
                           g_uvm_eviction_policy_name, pmm->policy_ops->name);
        }

        // 初始化选定的策略
        status = pmm->policy_ops->init(pmm);
        if (status != NV_OK) {
            // ... 错误处理 ...
        }

        UVM_INFO_PRINT("PMM GPU initialized with '%s' eviction policy.\n", pmm->policy_ops->name);
        // ...
        return NV_OK;
    }
    ```

## 4. 算法实现示例

在新框架下，实现一个新算法只需提供一个 `uvm_eviction_policy_ops` 结构体的实例。

### 4.1. LRU (当前实现的变体)
这个实现将封装当前代码的行为。

```c
// In uvm_policy_lru.c
static const struct uvm_eviction_policy_ops g_uvm_policy_lru = {
    .name          = "lru",
    .init          = lru_init,        // 初始化 va_block_used/unused 列表
    .deinit        = NULL,
    .add           = lru_add,         // 将新 chunk 加入 unused 列表
    .remove        = lru_remove,      // 从任一列表中删除
    .touch         = lru_touch,       // 实现: list_move_tail 到 va_block_used 尾部
    .unevictable   = lru_remove,      // 从列表中移除
    .evictable     = lru_touch,       // 重新变为可驱逐时，视为一次 touch
    .pick_to_evict = lru_pick_to_evict, // 实现: 优先从 unused 头部取，再从 used 头部取
};
```

### 4.2. Clock 算法

```c
// In uvm_policy_clock.c

// 私有数据结构
typedef struct {
    struct list_head list;       // 环形链表
    struct list_head *clock_hand; // 时钟指针
} clock_policy_context_t;

static const struct uvm_eviction_policy_ops g_uvm_policy_clock = {
    .name          = "clock",
    .init          = clock_init,      // 分配 clock_policy_context_t 并初始化
    .deinit        = clock_deinit,    // 释放上下文
    .add           = clock_add,       // 添加到环形链表中
    .remove        = clock_remove,    // 从环形链表中删除
    .touch         = clock_touch,     // 实现: 将 chunk->referenced 设为 true
    .unevictable   = clock_remove,
    .evictable     = clock_add,
    .pick_to_evict = clock_pick,      // 实现: 扫描环形链表，查找 referenced==false 的牺牲者
};
```

## 5. 结论

通过引入一个基于回调的策略框架，UVM 驱动的内存驱逐部分将变得高度模块化和可扩展。这种设计不仅使现有代码更清晰，还极大地简化了未来添加、测试和比较新驱逐算法的流程，为性能优化和学术研究提供了坚实的基础。
