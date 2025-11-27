# WIC (Warp-level Interrupt-based Communication) Integration Analysis
## NVIDIA Open GPU Kernel Modules - UVM Driver

**Analysis Date**: 2025-11-14
**Repository**: `/home/yunwei37/open-gpu-kernel-modules`
**Branch**: `uvm-print-test`

---

## Executive Summary

This document analyzes the NVIDIA open GPU kernel modules codebase to identify how the mechanisms described in the ATC'25 WIC paper are implemented in the existing UVM (Unified Virtual Memory) driver, and where WIC's enhancements would integrate.

**Key Finding**: The UVM driver already implements the core "warp stall → fault → service → replay" mechanism that WIC builds upon. WIC's contribution is adding **PCM (Producer-Consumer Communication Medium)** and **selective replay** logic on top of this existing infrastructure.

---

## Table of Contents

1. [Background: WIC Paper Summary](#1-background-wic-paper-summary)
2. [UVM Architecture Overview](#2-uvm-architecture-overview)
3. [Core Mechanisms Mapping](#3-core-mechanisms-mapping)
4. [Detailed Code Analysis](#4-detailed-code-analysis)
5. [WIC Integration Points](#5-wic-integration-points)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [Key Files Reference](#7-key-files-reference)

---

## 1. Background: WIC Paper Summary

### Problem
GPU consumer kernels waste compute resources busy-polling on synchronization flags while waiting for producer data in cross-device communication (multi-GPU or CPU-GPU).

### Solution
WIC replaces busy-wait polling with:
- **PCM (Producer-Consumer Communication Medium)**: UVM pages that intentionally trigger page faults
- **PAT (Producer Availability Tags)**: Flags indicating when producer data is ready
- **DAB (Data Availability Bitmap)**: Host-side bitmap tracking ready data
- **PFQ (Pending Faults Queue)**: Queue of faults waiting for producer data

### Three Modules
1. **Interrupter**: Consumer kernel accesses PCM page → triggers fault → warp stalls
2. **Monitor**: Checks DAB, enqueues faults in PFQ if data not ready
3. **Activator**: When data ready, copies to PCM, services fault, sends replay signal

### Result
Average 1.13× speedup, up to 1.3× on communication-intensive workloads.

---

## 2. UVM Architecture Overview

### What is UVM?

NVIDIA's Unified Virtual Memory allows CPU and GPU(s) to share a unified address space. When a GPU accesses a page not in its local memory:

1. **GPU hardware** detects the access and **automatically stalls the warp**
2. Writes fault entry to **GPU Fault Buffer** (ring buffer in GPU memory)
3. **GPU interrupt** fires, triggering host-side fault handler
4. **UVM driver** reads fault buffer, migrates data, updates page tables
5. **Replay signal** sent to GPU to resume stalled warps

### Key Insight

The **warp stall and replay mechanism already exists** in hardware and UVM driver. WIC leverages this existing mechanism by:
- Intentionally triggering faults (PCM access)
- Controlling when to service faults (DAB/PFQ)
- Issuing selective replays (only when data ready)

---

## 3. Core Mechanisms Mapping

| WIC Component | UVM Equivalent | File Location |
|---------------|----------------|---------------|
| **GPU Fault Buffer** | `uvm_replayable_fault_buffer_t` | `uvm_gpu.h:356-406` |
| **Fault Fetch** | `fetch_fault_buffer_entries()` | `uvm_gpu_replayable_faults.c` |
| **Fault Servicing** | `service_fault_batch()` | `uvm_gpu_replayable_faults.c` |
| **Warp Replay** | `push_replay_on_gpu()` | `uvm_gpu_replayable_faults.c:503-556` |
| **Hardware Replay** | `uvm_hal_pascal_replay_faults()` | `uvm_pascal_host.c:248-288` |
| **Replay Policies** | `uvm_perf_fault_replay_policy_t` | `uvm_gpu_replayable_faults.h:32-51` |

### WIC Additions (Not in Current UVM)

| WIC Component | Purpose | Where to Add |
|---------------|---------|--------------|
| **PCM** | Special UVM pages that trigger faults | New allocation in `uvm_gpu.h` |
| **PAT** | Producer availability tags | New allocation in `uvm_gpu.h` |
| **DAB** | Data availability bitmap | New field in fault buffer context |
| **PFQ** | Pending faults queue | New field in fault buffer context |
| **Segment Tree** | PCM page allocation | New allocator in Interrupter |

---

## 4. Detailed Code Analysis

### 4.1 GPU Fault Buffer

**File**: `kernel-open/nvidia-uvm/uvm_gpu.h:356-406`

```c
struct uvm_replayable_fault_buffer_struct
{
    // Maximum number of fault entries that can be stored in the buffer
    NvU32 max_faults;

    // Cached value of the GPU GET register to minimize the round-trips
    // over PCIe
    NvU32 cached_get;

    // Cached value of the GPU PUT register to minimize the round-trips over
    // PCIe
    NvU32 cached_put;

    // Policy that determines when GPU replays are issued during normal
    // fault servicing
    uvm_perf_fault_replay_policy_t replay_policy;

    // Tracker used to aggregate replay operations, needed for fault cancel
    // and GPU removal
    uvm_tracker_t replay_tracker;

    // Statistics
    struct {
        NvU64 num_read_faults;
        NvU64 num_write_faults;
        NvU64 num_duplicate_faults;
        atomic64_t num_pages_out;
        atomic64_t num_pages_in;
        NvU64 num_replays;
        NvU64 num_replays_ack_all;
    } stats;

    // Hardware details (max batch size, utlb count, etc.)
    NvU32 utlb_count;

    // Batch service context
    uvm_fault_service_batch_context_t batch_service_context;
};
```

**This is WIC's "Monitor" foundation** - it tracks all GPU page faults.

### 4.2 Fault Service Batch Context

**File**: `kernel-open/nvidia-uvm/uvm_gpu.h:286-339`

```c
struct uvm_fault_service_batch_context_struct
{
    // Array of elements fetched from the GPU fault buffer. The number of
    // elements in this array is exactly max_batch_size
    uvm_fault_buffer_entry_t *fault_cache;

    // Array of pointers to elements in fault cache used for fault
    // preprocessing. The number of elements in this array is exactly
    // max_batch_size
    uvm_fault_buffer_entry_t **ordered_fault_cache;

    // Per uTLB fault information. Used for replay policies and fault
    // cancellation on Pascal
    uvm_fault_utlb_info_t *utlbs;

    // Largest uTLB id seen in a GPU fault
    NvU32 max_utlb_id;

    NvU32 num_cached_faults;
    NvU32 num_coalesced_faults;

    // One of the VA spaces in this batch which had fatal faults
    uvm_va_space_t *fatal_va_space;
    uvm_gpu_t *fatal_gpu;

    bool has_throttled_faults;
    NvU32 num_invalid_prefetch_faults;
    NvU32 num_duplicate_faults;
    NvU32 num_replays;

    // Unique id (per-GPU) generated for tools events recording
    NvU32 batch_id;

    // Tracker for GPU operations
    uvm_tracker_t tracker;

    // Boolean used to avoid sorting the fault batch by instance_ptr if we
    // determine at fetch time that all the faults in the batch report the same
    // instance_ptr
    bool is_single_instance_ptr;

    // Last fetched fault. Used for fault filtering.
    uvm_fault_buffer_entry_t *last_fault;
};
```

**This manages fault batching** - similar to WIC's PFQ but currently processes all faults immediately.

### 4.3 Main Fault Servicing Loop

**File**: `kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c:2906-3015`

```c
void uvm_parent_gpu_service_replayable_faults(uvm_parent_gpu_t *parent_gpu)
{
    NvU32 num_replays = 0;
    NvU32 num_batches = 0;
    NvU32 num_throttled = 0;
    NV_STATUS status = NV_OK;
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;
    uvm_fault_service_batch_context_t *batch_context = &replayable_faults->batch_service_context;

    UVM_ASSERT(parent_gpu->replayable_faults_supported);

    uvm_tracker_init(&batch_context->tracker);

    // Process all faults in the buffer
    while (1) {
        if (num_throttled >= uvm_perf_fault_max_throttle_per_service ||
            num_batches >= uvm_perf_fault_max_batches_per_service) {
            break;
        }

        batch_context->num_invalid_prefetch_faults = 0;
        batch_context->num_duplicate_faults        = 0;
        batch_context->num_replays                 = 0;
        batch_context->fatal_va_space              = NULL;
        batch_context->fatal_gpu                   = NULL;
        batch_context->has_throttled_faults        = false;

        // Fetch faults from GPU buffer
        status = fetch_fault_buffer_entries(parent_gpu, batch_context,
                                            FAULT_FETCH_MODE_BATCH_READY);
        if (status != NV_OK)
            break;

        if (batch_context->num_cached_faults == 0)
            break;

        ++batch_context->batch_id;

        // Preprocess: sort, deduplicate, coalesce
        status = preprocess_fault_batch(parent_gpu, batch_context);

        num_replays += batch_context->num_replays;

        if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
            continue;
        else if (status != NV_OK)
            break;

        // Service faults: migrate pages, update page tables
        status = service_fault_batch(parent_gpu, FAULT_SERVICE_MODE_REGULAR,
                                     batch_context);

        num_replays += batch_context->num_replays;

        enable_disable_prefetch_faults(parent_gpu, batch_context);

        if (status != NV_OK) {
            cancel_fault_batch(parent_gpu, batch_context,
                              uvm_tools_status_to_fatal_fault_reason(status));
            break;
        }

        // Handle fatal faults
        if (batch_context->fatal_va_space) {
            status = uvm_tracker_wait(&batch_context->tracker);
            if (status == NV_OK) {
                status = cancel_faults_precise(batch_context);
                if (status == NV_OK) {
                    ++num_batches;
                    continue;
                }
            }
            break;
        }

        // Issue replay based on policy
        if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BATCH) {
            status = push_replay_on_parent_gpu(parent_gpu,
                                              UVM_FAULT_REPLAY_TYPE_START,
                                              batch_context);
            if (status != NV_OK)
                break;
            ++num_replays;
        }
        else if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH) {
            uvm_gpu_buffer_flush_mode_t flush_mode = UVM_GPU_BUFFER_FLUSH_MODE_CACHED_PUT;

            // Use UPDATE_PUT if too many duplicates
            if (batch_context->num_duplicate_faults * 100 >
                batch_context->num_cached_faults * replayable_faults->replay_update_put_ratio) {
                flush_mode = UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT;
            }

            status = fault_buffer_flush_locked(parent_gpu, NULL, flush_mode,
                                              UVM_FAULT_REPLAY_TYPE_START,
                                              batch_context);
            if (status != NV_OK)
                break;
            ++num_replays;
            status = uvm_tracker_wait(&replayable_faults->replay_tracker);
            if (status != NV_OK)
                break;
        }

        ++num_batches;

        if (batch_context->has_throttled_faults)
            ++num_throttled;
    }

    // Cleanup
    uvm_tracker_deinit(&batch_context->tracker);
}
```

**This is where WIC would add conditional servicing** - check DAB before calling `service_fault_batch()`.

### 4.4 Replay Mechanism

**File**: `kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c:503-556`

```c
static NV_STATUS push_replay_on_gpu(uvm_gpu_t *gpu,
                                    uvm_fault_replay_type_t type,
                                    uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status;
    uvm_push_t push;
    uvm_replayable_fault_buffer_t *replayable_faults = &gpu->parent->fault_buffer.replayable;
    uvm_tracker_t *tracker = NULL;

    if (batch_context)
        tracker = &batch_context->tracker;

    // Begin GPU command push
    status = uvm_push_begin_acquire(gpu->channel_manager, UVM_CHANNEL_TYPE_MEMOPS,
                                    tracker, &push, "Replaying faults");
    if (status != NV_OK)
        return status;

    // Call HAL function to push replay method to GPU
    gpu->parent->host_hal->replay_faults(&push, type);

    // Do not count REPLAY_TYPE_START_ACK_ALL's toward the replay count.
    // REPLAY_TYPE_START_ACK_ALL's are issued for cancels, and the cancel
    // algorithm checks to make sure that no REPLAY_TYPE_START's have been
    // issued using batch_context->replays.
    if (batch_context && type != UVM_FAULT_REPLAY_TYPE_START_ACK_ALL) {
        uvm_tools_broadcast_replay(gpu, &push, batch_context->batch_id,
                                  UVM_FAULT_CLIENT_TYPE_GPC);
        ++batch_context->num_replays;
    }

    uvm_push_end(&push);

    // Add this push to the GPU's replay_tracker so cancel can wait on it.
    status = uvm_tracker_add_push_safe(&replayable_faults->replay_tracker, &push);

    if (uvm_procfs_is_debug_enabled()) {
        if (type == UVM_FAULT_REPLAY_TYPE_START)
            ++replayable_faults->stats.num_replays;
        else
            ++replayable_faults->stats.num_replays_ack_all;
    }

    return status;
}

static NV_STATUS push_replay_on_parent_gpu(uvm_parent_gpu_t *parent_gpu,
                                           uvm_fault_replay_type_t type,
                                           uvm_fault_service_batch_context_t *batch_context)
{
    uvm_gpu_t *gpu = uvm_parent_gpu_find_first_valid_gpu(parent_gpu);

    if (gpu)
        return push_replay_on_gpu(gpu, type, batch_context);

    return NV_OK;
}
```

**This is WIC's "Activator" - sends replay signal** to resume stalled warps.

### 4.5 Hardware Replay Implementation

**File**: `kernel-open/nvidia-uvm/uvm_pascal_host.c:248-288`

```c
void uvm_hal_pascal_replay_faults(uvm_push_t *push, uvm_fault_replay_type_t type)
{
    NvU32 aperture_value;
    NvU32 replay_value = 0;
    uvm_gpu_t *gpu = uvm_push_get_gpu(push);
    uvm_gpu_phys_address_t pdb;
    NvU32 va_lo = 0;
    NvU32 va_hi = 0;
    NvU32 pdb_lo;
    NvU32 pdb_hi;

    // MMU will not forward the replay to the uTLBs if the PDB is not in the MMU PDB_ID cache.
    // To force a replay regardless of which faults happen to be in the uTLB replay lists,
    // we use the PDB of the channel used to push the replay, which is guaranteed to be in
    // the cache. We invalidate PTEs for address 0x0 to minimize performance impact.
    UVM_ASSERT_MSG(type == UVM_FAULT_REPLAY_TYPE_START ||
                   type == UVM_FAULT_REPLAY_TYPE_START_ACK_ALL,
                   "replay_type: %u\n", type);

    pdb = uvm_page_tree_pdb_address(&gpu->address_space_tree);

    if (pdb.aperture == UVM_APERTURE_VID)
        aperture_value = HWCONST(C06F, MEM_OP_C, TLB_INVALIDATE_PDB_APERTURE, VID_MEM);
    else
        aperture_value = HWCONST(C06F, MEM_OP_C, TLB_INVALIDATE_PDB_APERTURE, SYS_MEM_COHERENT);

    UVM_ASSERT_MSG(IS_ALIGNED(pdb.address, 1 << 12), "pdb 0x%llx\n", pdb.address);
    pdb.address >>= 12;

    pdb_lo = pdb.address & HWMASK(C06F, MEM_OP_C, TLB_INVALIDATE_PDB_ADDR_LO);
    pdb_hi = pdb.address >> HWSIZE(C06F, MEM_OP_C, TLB_INVALIDATE_PDB_ADDR_LO);

    if (type == UVM_FAULT_REPLAY_TYPE_START)
        replay_value = HWCONST(C06F, MEM_OP_C, TLB_INVALIDATE_REPLAY, START);
    else if (type == UVM_FAULT_REPLAY_TYPE_START_ACK_ALL)
        replay_value = HWCONST(C06F, MEM_OP_C, TLB_INVALIDATE_REPLAY, START_ACK_ALL);

    // Push hardware method to GPU (4 32-bit values)
    NV_PUSH_4U(C06F, MEM_OP_A, HWCONST(C06F, MEM_OP_A, TLB_INVALIDATE_SYSMEMBAR, DIS) |
                               HWVALUE(C06F, MEM_OP_A, TLB_INVALIDATE_TARGET_ADDR_LO, va_lo),
                     MEM_OP_B, HWVALUE(C06F, MEM_OP_B, TLB_INVALIDATE_TARGET_ADDR_HI, va_hi),
                     MEM_OP_C, HWVALUE(C06F, MEM_OP_C, TLB_INVALIDATE_PDB_ADDR_LO, pdb_lo) |
                               aperture_value |
                               replay_value,
                     MEM_OP_D, HWVALUE(C06F, MEM_OP_D, TLB_INVALIDATE_PDB_ADDR_HI, pdb_hi) |
                               HWCONST(C06F, MEM_OP_D, OPERATION, MMU_TLB_INVALIDATE_TARGETED));
}
```

**This sends the actual hardware command** to GPU's command FIFO to resume warps!

### 4.6 Replay Policies

**File**: `kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.h:32-51`

```c
typedef enum
{
    // Issue a fault replay after all faults for a block within a batch have been serviced
    UVM_PERF_FAULT_REPLAY_POLICY_BLOCK = 0,

    // Issue a fault replay after each fault batch has been serviced
    UVM_PERF_FAULT_REPLAY_POLICY_BATCH,

    // Like UVM_PERF_FAULT_REPLAY_POLICY_BATCH but only one batch of faults is serviced.
    // The fault buffer is flushed before issuing the replay. The potential benefit is
    // that we can resume execution of some SMs earlier, if SMs are faulting on different
    // sets of pages.
    UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH,

    // Issue a fault replay after all faults in the buffer have been serviced
    UVM_PERF_FAULT_REPLAY_POLICY_ONCE,

    // TODO: Bug 1768226: Implement uTLB-aware fault replay policy

    UVM_PERF_FAULT_REPLAY_POLICY_MAX,
} uvm_perf_fault_replay_policy_t;
```

**Default**: `UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH`

**WIC would add**: A new policy that uses DAB/PFQ to selectively replay only when producer data is ready.

### 4.7 Fault Buffer Flush

**File**: `kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c:674-693`

```c
NV_STATUS uvm_gpu_fault_buffer_flush(uvm_gpu_t *gpu)
{
    NV_STATUS status = NV_OK;

    UVM_ASSERT(gpu->parent->replayable_faults_supported);

    // Disables replayable fault interrupts and fault servicing
    uvm_parent_gpu_replayable_faults_isr_lock(gpu->parent);

    status = fault_buffer_flush_locked(gpu->parent,
                                       gpu,
                                       UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT,
                                       UVM_FAULT_REPLAY_TYPE_START,
                                       NULL);

    // This will trigger the top half to start servicing faults again, if the
    // replay brought any back in
    uvm_parent_gpu_replayable_faults_isr_unlock(gpu->parent);
    return status;
}
```

---

## 5. WIC Integration Points

### 5.1 Data Structures to Add

**File**: `kernel-open/nvidia-uvm/uvm_gpu.h` (add around line 347)

```c
// WIC: Producer-Consumer Communication Medium
typedef struct {
    // PCM region: Special UVM pages that intentionally trigger faults
    void *pcm_region;
    size_t pcm_region_size;
    NvU32 pcm_page_count;          // Paper suggests 16K pages

    // PAT region: Producer Availability Tags
    void *pat_region;
    size_t pat_region_size;

    // Segment tree for PCM page allocation
    bool *pcm_page_free;            // Free page bitmap
    struct {
        NvU32 sum;                  // Free pages in range
        NvU32 maxLen;               // Max consecutive free
        NvU32 leftLen;              // Left edge consecutive free
        NvU32 rightLen;             // Right edge consecutive free
    } *segment_tree;
    NvU32 segment_tree_size;

    // DAB: Data Availability Bitmap (host-side)
    NvU64 *dab_bits;
    NvU32 dab_size;
    spinlock_t dab_lock;

    // PFQ: Pending Faults Queue
    struct list_head pending_faults_queue;
    spinlock_t pfq_lock;

    // Registration table: maps channel_id -> PCM pages
    struct {
        NvU32 pcm_page_start;
        NvU32 num_pages;
        NvU32 tag_id;
        void *producer_data_addr;
        size_t producer_data_size;
    } *channel_registry;
    NvU32 max_channels;

} uvm_wic_context_t;

// Add to uvm_replayable_fault_buffer_t
struct uvm_replayable_fault_buffer_struct
{
    // ... existing fields ...

    // WIC context
    uvm_wic_context_t wic;
};
```

### 5.2 Initialization

**File**: `kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c`

Add to `fault_buffer_init_replayable_faults()`:

```c
static NV_STATUS fault_buffer_init_replayable_faults(uvm_parent_gpu_t *parent_gpu)
{
    // ... existing initialization ...

    // WIC: Initialize PCM/PAT regions
    status = wic_init_pcm_pat(parent_gpu);
    if (status != NV_OK)
        return status;

    return NV_OK;
}

static NV_STATUS wic_init_pcm_pat(uvm_parent_gpu_t *parent_gpu)
{
    uvm_wic_context_t *wic = &parent_gpu->fault_buffer.replayable.wic;
    NV_STATUS status;

    // Allocate PCM region (16K pages = ~64MB for 4KB pages)
    wic->pcm_page_count = 16384;
    wic->pcm_region_size = wic->pcm_page_count * PAGE_SIZE;

    // Use cudaMallocManaged equivalent in kernel
    status = uvm_rm_mem_alloc_and_map_cpu(parent_gpu, UVM_RM_MEM_TYPE_SYS,
                                         wic->pcm_region_size, 0,
                                         &wic->pcm_region);
    if (status != NV_OK)
        return status;

    // Set preferred location to host (ensures faults on first access)
    // Note: Need to disable UVM prefetcher for PCM region

    // Allocate PAT region
    wic->pat_region_size = wic->pcm_page_count * sizeof(NvU32);
    status = uvm_rm_mem_alloc_and_map_cpu(parent_gpu, UVM_RM_MEM_TYPE_SYS,
                                         wic->pat_region_size, 0,
                                         &wic->pat_region);
    if (status != NV_OK)
        goto error;

    // Initialize segment tree for PCM allocation
    wic->segment_tree_size = wic->pcm_page_count * 4; // Conservative
    wic->segment_tree = uvm_kvmalloc_zero(wic->segment_tree_size *
                                         sizeof(*wic->segment_tree));
    if (!wic->segment_tree)
        goto error;

    wic->pcm_page_free = uvm_kvmalloc_zero(wic->pcm_page_count *
                                          sizeof(*wic->pcm_page_free));
    if (!wic->pcm_page_free)
        goto error;

    // Mark all pages as free initially
    memset(wic->pcm_page_free, 1, wic->pcm_page_count);
    wic_segment_tree_build(wic->segment_tree, wic->pcm_page_free,
                          0, wic->pcm_page_count - 1, 0);

    // Initialize DAB
    wic->dab_size = (wic->pcm_page_count + 63) / 64; // Round up to NvU64
    wic->dab_bits = uvm_kvmalloc_zero(wic->dab_size * sizeof(NvU64));
    if (!wic->dab_bits)
        goto error;
    spin_lock_init(&wic->dab_lock);

    // Initialize PFQ
    INIT_LIST_HEAD(&wic->pending_faults_queue);
    spin_lock_init(&wic->pfq_lock);

    // Initialize channel registry
    wic->max_channels = 256;
    wic->channel_registry = uvm_kvmalloc_zero(wic->max_channels *
                                             sizeof(*wic->channel_registry));
    if (!wic->channel_registry)
        goto error;

    return NV_OK;

error:
    wic_deinit_pcm_pat(parent_gpu);
    return NV_ERR_NO_MEMORY;
}
```

### 5.3 Interrupter Module

**File**: `kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c`

Modify `fetch_fault_buffer_entries()` or add hook in `preprocess_fault_batch()`:

```c
static NV_STATUS preprocess_fault_batch(uvm_parent_gpu_t *parent_gpu,
                                       uvm_fault_service_batch_context_t *batch_context)
{
    // ... existing preprocessing ...

    // WIC: Check if any faults are for PCM/PAT regions
    for (i = 0; i < batch_context->num_cached_faults; i++) {
        uvm_fault_buffer_entry_t *entry = &batch_context->fault_cache[i];

        if (wic_is_pcm_fault(parent_gpu, entry->fault_address)) {
            // Mark as WIC fault for special handling
            entry->is_wic_pcm_fault = true;
        }
        else if (wic_is_pat_fault(parent_gpu, entry->fault_address)) {
            // PAT write - producer signaling data ready
            entry->is_wic_pat_fault = true;
        }
    }

    return status;
}

static bool wic_is_pcm_fault(uvm_parent_gpu_t *parent_gpu, NvU64 addr)
{
    uvm_wic_context_t *wic = &parent_gpu->fault_buffer.replayable.wic;
    NvU64 pcm_start = (NvU64)wic->pcm_region;
    NvU64 pcm_end = pcm_start + wic->pcm_region_size;

    return (addr >= pcm_start && addr < pcm_end);
}

static bool wic_is_pat_fault(uvm_parent_gpu_t *parent_gpu, NvU64 addr)
{
    uvm_wic_context_t *wic = &parent_gpu->fault_buffer.replayable.wic;
    NvU64 pat_start = (NvU64)wic->pat_region;
    NvU64 pat_end = pat_start + wic->pat_region_size;

    return (addr >= pat_start && addr < pat_end);
}
```

### 5.4 Monitor Module

**File**: `kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c`

Modify `service_fault_batch()`:

```c
static NV_STATUS service_fault_batch(uvm_parent_gpu_t *parent_gpu,
                                    fault_service_mode_t service_mode,
                                    uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status;

    // WIC: Separate PCM/PAT faults from regular faults
    uvm_fault_buffer_entry_t *regular_faults[batch_context->num_cached_faults];
    uvm_fault_buffer_entry_t *pcm_faults[batch_context->num_cached_faults];
    uvm_fault_buffer_entry_t *pat_faults[batch_context->num_cached_faults];
    NvU32 num_regular = 0, num_pcm = 0, num_pat = 0;

    for (i = 0; i < batch_context->num_cached_faults; i++) {
        uvm_fault_buffer_entry_t *entry = &batch_context->fault_cache[i];

        if (entry->is_wic_pat_fault) {
            pat_faults[num_pat++] = entry;
        }
        else if (entry->is_wic_pcm_fault) {
            pcm_faults[num_pcm++] = entry;
        }
        else {
            regular_faults[num_regular++] = entry;
        }
    }

    // Handle PAT faults first (producer signaling)
    if (num_pat > 0) {
        status = wic_handle_pat_faults(parent_gpu, pat_faults, num_pat, batch_context);
        if (status != NV_OK)
            return status;
    }

    // Handle PCM faults (consumer requests)
    if (num_pcm > 0) {
        status = wic_handle_pcm_faults(parent_gpu, pcm_faults, num_pcm, batch_context);
        if (status != NV_OK)
            return status;
    }

    // Handle regular UVM faults
    if (num_regular > 0) {
        status = service_fault_batch_regular(parent_gpu, regular_faults, num_regular,
                                            service_mode, batch_context);
        if (status != NV_OK)
            return status;
    }

    return NV_OK;
}

static NV_STATUS wic_handle_pat_faults(uvm_parent_gpu_t *parent_gpu,
                                       uvm_fault_buffer_entry_t **faults,
                                       NvU32 num_faults,
                                       uvm_fault_service_batch_context_t *batch_context)
{
    uvm_wic_context_t *wic = &parent_gpu->fault_buffer.replayable.wic;

    for (i = 0; i < num_faults; i++) {
        NvU64 addr = faults[i]->fault_address;
        NvU32 tag_id = wic_addr_to_tag_id(wic, addr);

        // Update DAB
        spin_lock(&wic->dab_lock);
        wic->dab_bits[tag_id / 64] |= (1ULL << (tag_id % 64));
        spin_unlock(&wic->dab_lock);

        // Check PFQ for faults waiting on this tag
        wic_wake_pending_faults(parent_gpu, tag_id, batch_context);
    }

    return NV_OK;
}

static NV_STATUS wic_handle_pcm_faults(uvm_parent_gpu_t *parent_gpu,
                                       uvm_fault_buffer_entry_t **faults,
                                       NvU32 num_faults,
                                       uvm_fault_service_batch_context_t *batch_context)
{
    uvm_wic_context_t *wic = &parent_gpu->fault_buffer.replayable.wic;

    for (i = 0; i < num_faults; i++) {
        NvU64 addr = faults[i]->fault_address;
        NvU32 pcm_page = wic_addr_to_pcm_page(wic, addr);

        // Look up channel info
        channel_info = wic_lookup_channel_by_pcm_page(wic, pcm_page);
        if (!channel_info) {
            // Error: PCM fault without registration
            continue;
        }

        // Check if producer data is ready
        spin_lock(&wic->dab_lock);
        bool data_ready = (wic->dab_bits[channel_info->tag_id / 64] >>
                          (channel_info->tag_id % 64)) & 1;
        spin_unlock(&wic->dab_lock);

        if (!data_ready) {
            // Enqueue to PFQ
            wic_enqueue_pending_fault(wic, faults[i], channel_info);
        }
        else {
            // Data ready - activate immediately
            wic_activate_fault(parent_gpu, faults[i], channel_info, batch_context);
        }
    }

    return NV_OK;
}
```

### 5.5 Activator Module

**File**: `kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c`

```c
static NV_STATUS wic_activate_fault(uvm_parent_gpu_t *parent_gpu,
                                   uvm_fault_buffer_entry_t *fault,
                                   channel_info_t *channel_info,
                                   uvm_fault_service_batch_context_t *batch_context)
{
    uvm_wic_context_t *wic = &parent_gpu->fault_buffer.replayable.wic;
    NV_STATUS status;
    void *pcm_addr;

    // Get PCM page address
    pcm_addr = wic->pcm_region + (channel_info->pcm_page_start * PAGE_SIZE);

    // Copy producer data to PCM pages
    status = wic_copy_producer_data(parent_gpu,
                                   channel_info->producer_data_addr,
                                   pcm_addr,
                                   channel_info->producer_data_size);
    if (status != NV_OK)
        return status;

    // Service the fault (update page tables to map PCM page to GPU)
    status = wic_service_pcm_fault(parent_gpu, fault, pcm_addr);
    if (status != NV_OK)
        return status;

    // Issue replay for this specific fault
    status = push_replay_on_parent_gpu(parent_gpu, UVM_FAULT_REPLAY_TYPE_START,
                                      batch_context);

    // Schedule PCM page invalidation (P/P' double buffering)
    wic_schedule_pcm_invalidation(wic, channel_info->pcm_page_start,
                                  channel_info->num_pages);

    return status;
}

static void wic_wake_pending_faults(uvm_parent_gpu_t *parent_gpu,
                                   NvU32 tag_id,
                                   uvm_fault_service_batch_context_t *batch_context)
{
    uvm_wic_context_t *wic = &parent_gpu->fault_buffer.replayable.wic;
    struct list_head *pos, *tmp;

    spin_lock(&wic->pfq_lock);

    list_for_each_safe(pos, tmp, &wic->pending_faults_queue) {
        pending_fault_entry_t *entry = list_entry(pos, pending_fault_entry_t, list);

        if (entry->channel_info->tag_id == tag_id) {
            // Remove from PFQ
            list_del(pos);

            // Activate
            wic_activate_fault(parent_gpu, entry->fault, entry->channel_info,
                             batch_context);

            kfree(entry);
        }
    }

    spin_unlock(&wic->pfq_lock);
}
```

### 5.6 User-Space API

**File**: `kernel-open/nvidia-uvm/uvm_ioctl.h` (add new ioctl)

```c
// WIC: Register a producer-consumer channel
typedef struct {
    void *producer_data_addr;     // IN: Producer buffer address
    size_t producer_data_size;    // IN: Size of producer buffer
    NvU32 tag_id;                 // IN: Tag ID for this channel

    void *pcm_addr;               // OUT: PCM address for consumer to access
    NvU32 channel_id;             // OUT: Channel ID for cleanup
} UVM_WIC_REGISTER_CHANNEL_PARAMS;

#define UVM_WIC_REGISTER_CHANNEL     UVM_IOCTL_BASE(100)

// WIC: Notify that producer data is ready
typedef struct {
    NvU32 channel_id;             // IN: Channel ID from register
    NvU32 tag_id;                 // IN: Tag ID to mark ready
} UVM_WIC_NOTIFY_READY_PARAMS;

#define UVM_WIC_NOTIFY_READY         UVM_IOCTL_BASE(101)
```

---

## 6. Implementation Roadmap

### Phase 1: Foundation (1-2 weeks)
1. Add WIC data structures to `uvm_gpu.h`
2. Implement PCM/PAT allocation in `fault_buffer_init_replayable_faults()`
3. Implement segment tree allocator for PCM pages
4. Add DAB and PFQ basic infrastructure

### Phase 2: Fault Detection (1 week)
1. Modify `preprocess_fault_batch()` to detect PCM/PAT faults
2. Add `wic_is_pcm_fault()` and `wic_is_pat_fault()` helper functions
3. Add flags to `uvm_fault_buffer_entry_t` for WIC fault types

### Phase 3: Monitor Module (2 weeks)
1. Implement `wic_handle_pat_faults()` - update DAB on producer signals
2. Implement `wic_handle_pcm_faults()` - check DAB and enqueue to PFQ
3. Implement PFQ management (enqueue, dequeue, lookup)
4. Add locking for DAB and PFQ (spinlocks for interrupt context)

### Phase 4: Activator Module (2 weeks)
1. Implement `wic_activate_fault()` - copy data to PCM and service fault
2. Implement `wic_wake_pending_faults()` - wake PFQ entries when data ready
3. Implement PCM page invalidation with P/P' double buffering
4. Add selective replay based on DAB state

### Phase 5: User-Space API (1 week)
1. Add ioctls for channel registration and data ready notification
2. Implement `wic_register_channel()` handler
3. Implement `wic_notify_ready()` handler
4. Add channel registry management

### Phase 6: Testing & Optimization (2-3 weeks)
1. Unit tests for segment tree allocator
2. Stress tests for DAB/PFQ with concurrent faults
3. Performance benchmarking with producer-consumer workloads
4. Tune batch sizes, PFQ limits, PCM page count
5. Add debugfs/procfs entries for WIC statistics

### Phase 7: Documentation & Cleanup (1 week)
1. Add inline documentation
2. Update UVM driver documentation
3. Code review and cleanup
4. Prepare patch series for upstream (if applicable)

**Total Estimated Time**: 10-12 weeks

---

## 7. Key Files Reference

### Core UVM Fault Handling

| File | Lines | Purpose |
|------|-------|---------|
| `kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c` | 2906-3015 | Main fault servicing loop |
| `kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c` | 503-556 | Replay mechanism (software) |
| `kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c` | 674-693 | Fault buffer flush |
| `kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.h` | 32-77 | Replay policies and API |
| `kernel-open/nvidia-uvm/uvm_pascal_host.c` | 248-288 | Replay hardware implementation |
| `kernel-open/nvidia-uvm/uvm_gpu.h` | 286-339 | Fault service batch context |
| `kernel-open/nvidia-uvm/uvm_gpu.h` | 356-406 | Replayable fault buffer structure |

### HAL (Hardware Abstraction Layer)

| File | Lines | Purpose |
|------|-------|---------|
| `kernel-open/nvidia-uvm/uvm_hal.h` | 779-804 | Host HAL structure definition |
| `kernel-open/nvidia-uvm/uvm_hal.h` | 624-655 | Replay/cancel fault function types |
| `kernel-open/nvidia-uvm/uvm_hal.c` | 238-283 | HAL initialization (Pascal-Ampere) |

### Architecture-Specific Implementations

| File | Purpose |
|------|---------|
| `kernel-open/nvidia-uvm/uvm_pascal_host.c` | Pascal GPU replay implementation |
| `kernel-open/nvidia-uvm/uvm_volta_host.c` | Volta GPU replay implementation |
| `kernel-open/nvidia-uvm/uvm_turing_*.c` | Turing GPU specific code |
| `kernel-open/nvidia-uvm/uvm_ampere_*.c` | Ampere GPU specific code |
| `kernel-open/nvidia-uvm/uvm_hopper_*.c` | Hopper GPU specific code |

### Fault Buffer Hardware Definitions

| File | Purpose |
|------|---------|
| `kernel-open/nvidia-uvm/uvm_pascal_fault_buffer.c` | Pascal fault buffer HAL |
| `kernel-open/nvidia-uvm/uvm_volta_fault_buffer.c` | Volta fault buffer HAL |
| `kernel-open/nvidia-uvm/uvm_ampere_fault_buffer.c` | Ampere fault buffer HAL |

---

## Appendix A: Module Parameters

Current UVM module parameters relevant to WIC:

| Parameter | Default | Description | WIC Impact |
|-----------|---------|-------------|------------|
| `uvm_perf_fault_batch_count` | 256 | Faults fetched per batch | Paper mentions this as bottleneck |
| `uvm_perf_fault_replay_policy` | BATCH_FLUSH | When to issue replays | WIC adds DAB-based policy |
| `uvm_perf_fault_replay_update_put_ratio` | 50 | Threshold for PUT update | Affects duplicate handling |
| `uvm_perf_fault_max_batches_per_service` | 20 | Max batches per bottom-half | Limits fault servicing work |
| `uvm_perf_fault_coalesce` | 1 | Enable fault coalescing | May interact with WIC |

---

## Appendix B: Hardware Replay Methods

### Pascal/Volta/Turing/Ampere

**Register**: `NVC06F_MEM_OP_C_TLB_INVALIDATE_REPLAY`

**Values**:
- `START` (0): Start replay of faulted operations
- `START_ACK_ALL` (1): Start replay and acknowledge all pending faults

**Method Sequence**:
```
MEM_OP_A: VA_LO | SYSMEMBAR_DIS
MEM_OP_B: VA_HI
MEM_OP_C: PDB_ADDR_LO | PDB_APERTURE | REPLAY_TYPE
MEM_OP_D: PDB_ADDR_HI | OPERATION_MMU_TLB_INVALIDATE_TARGETED
```

---

## Appendix C: WIC Paper Key Metrics

From the paper's experimental results:

| Metric | Value |
|--------|-------|
| Average speedup | 1.13× |
| Max speedup (C_H3D) | >1.3× |
| WIC overhead | ~1-3% |
| PCM pages | 16K (supports ~256MB data) |
| Fault buffer max batch | 256 entries |
| Scenario 2 dominance | >90% of applications |
| Polling overhead reduction | 60-90% consumer time saved |

---

## Appendix D: Questions for Further Investigation

1. **How to disable UVM prefetcher for PCM/PAT regions?**
   - Need to ensure first access always faults
   - May require new hint system or explicit prefetch control

2. **How to handle PCM page recycling atomically?**
   - P/P' double buffering requires careful synchronization
   - Need to coordinate with GPU scheduler

3. **How to batch PCM faults efficiently?**
   - Current batching is by arrival time
   - WIC needs to batch by tag_id dependency

4. **How to extend to GPU-GPU communication?**
   - PAT mechanism vs direct DAB write
   - Cross-GPU replay coordination

5. **How to handle fault buffer overflow with PFQ?**
   - PFQ adds memory pressure
   - May need adaptive batching

---

## Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-14 | 1.0 | Initial analysis |

---

**End of Document**
