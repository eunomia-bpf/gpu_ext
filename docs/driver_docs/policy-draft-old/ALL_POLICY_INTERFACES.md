# Complete List of ALL Policy Interfaces in NVIDIA UVM Driver

This document provides a **comprehensive catalog** of ALL policy interfaces, enums, hints, strategies, and decision points already exposed in the UVM driver.

---

## 1. Fault Replay Policy (Primary for WIC)

**File**: `uvm_gpu_replayable_faults.h:34-51`

```c
typedef enum {
    UVM_PERF_FAULT_REPLAY_POLICY_BLOCK = 0,        // Replay after each block
    UVM_PERF_FAULT_REPLAY_POLICY_BATCH,            // Replay after each batch
    UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH,      // Batch + flush (default)
    UVM_PERF_FAULT_REPLAY_POLICY_ONCE,             // Replay once at end
    UVM_PERF_FAULT_REPLAY_POLICY_MAX,
} uvm_perf_fault_replay_policy_t;
```

**Module Parameter**: `uvm_perf_fault_replay_policy` (runtime tunable via sysfs)
**Usage**: Read at line 196 of `uvm_gpu_replayable_faults.c`
**Decision Point**: Line 2986 in `uvm_parent_gpu_service_replayable_faults()`

---

## 2. VA Policy (User-Facing Memory Policy)

**File**: `uvm_va_policy.h:36-82`

### 2.1 Read Duplication Policy
```c
typedef enum {
    UVM_READ_DUPLICATION_UNSET = 0,
    UVM_READ_DUPLICATION_ENABLED,
    UVM_READ_DUPLICATION_DISABLED,
    UVM_READ_DUPLICATION_MAX
} uvm_read_duplication_policy_t;
```

**API Functions**:
- `uvm_api_enable_read_duplication()`
- `uvm_api_disable_read_duplication()`

### 2.2 VA Policy Structure
```c
struct uvm_va_policy_struct {
    uvm_read_duplication_policy_t read_duplication;
    uvm_processor_id_t preferred_location;      // Processor ID for preferred location
    int preferred_nid;                          // NUMA node ID for CPU
    uvm_processor_mask_t accessed_by;           // Processors that access this VA
};
```

**API Functions**:
- `uvm_api_set_preferred_location()` - Set preferred memory location
- `uvm_api_unset_preferred_location()` - Clear preferred location
- `uvm_api_set_accessed_by()` - Set accessing processors
- `uvm_api_unset_accessed_by()` - Clear accessing processors
- `uvm_api_enable_system_wide_atomics()` - Enable system-wide atomics
- `uvm_api_disable_system_wide_atomics()` - Disable system-wide atomics

### 2.3 VA Policy Type
```c
typedef enum {
    UVM_VA_POLICY_PREFERRED_LOCATION = 0,
    UVM_VA_POLICY_ACCESSED_BY,
    UVM_VA_POLICY_READ_DUPLICATION,
} uvm_va_policy_type_t;
```

---

## 3. Migration Cause Policy

**File**: `uvm_va_block_types.h:116-129`

```c
typedef enum {
    UVM_MAKE_RESIDENT_CAUSE_REPLAYABLE_FAULT,      // GPU replayable fault
    UVM_MAKE_RESIDENT_CAUSE_NON_REPLAYABLE_FAULT,  // GPU non-replayable fault
    UVM_MAKE_RESIDENT_CAUSE_ACCESS_COUNTER,        // Access counter notification
    UVM_MAKE_RESIDENT_CAUSE_PREFETCH,              // Prefetch operation
    UVM_MAKE_RESIDENT_CAUSE_EVICTION,              // Memory eviction
    UVM_MAKE_RESIDENT_CAUSE_API_TOOLS,             // Tools API call
    UVM_MAKE_RESIDENT_CAUSE_API_MIGRATE,           // Explicit migration API
    UVM_MAKE_RESIDENT_CAUSE_API_SET_RANGE_GROUP,   // Range group API
    UVM_MAKE_RESIDENT_CAUSE_API_HINT,              // Hint API
    UVM_MAKE_RESIDENT_CAUSE_MAX
} uvm_make_resident_cause_t;
```

**Usage**: Passed to all migration operations, tracked in perf events
**Exposed via**: `uvm_perf_event_data_t.migration.cause`

---

## 4. Thrashing Detection & Hints

**File**: `uvm_perf_thrashing.h:33-74`

### 4.1 Thrashing Hint Type
```c
typedef enum {
    UVM_PERF_THRASHING_HINT_TYPE_NONE     = 0,  // No thrashing detected
    UVM_PERF_THRASHING_HINT_TYPE_PIN      = 1,  // Pin pages remotely
    UVM_PERF_THRASHING_HINT_TYPE_THROTTLE = 2,  // Throttle processor execution
} uvm_perf_thrashing_hint_type_t;
```

### 4.2 Thrashing Hint Structure
```c
typedef struct {
    uvm_perf_thrashing_hint_type_t type;

    union {
        struct {
            uvm_processor_id_t residency;         // Where to pin
            uvm_processor_mask_t processors;      // Which processors to map
        } pin;

        struct {
            NvU64 end_time_stamp;                 // When to unthrottle (ns)
        } throttle;
    };
} uvm_perf_thrashing_hint_t;
```

**API Function**: `uvm_perf_thrashing_get_hint()` - Query thrashing hint for address
**Module Parameters** (line 307-318 of `uvm_perf_thrashing.c`):
```c
module_param(uvm_perf_thrashing_enable,        uint, S_IRUGO);  // 1 (default)
module_param(uvm_perf_thrashing_threshold,     uint, S_IRUGO);  // 3 (default)
module_param(uvm_perf_thrashing_pin_threshold, uint, S_IRUGO);  // 10 (default)
module_param(uvm_perf_thrashing_lapse_usec,    uint, S_IRUGO);  // 10000 (10ms)
module_param(uvm_perf_thrashing_nap,           uint, S_IRUGO);  // 100 (default)
module_param(uvm_perf_thrashing_epoch,         uint, S_IRUGO);  // 3 (default)
module_param(uvm_perf_thrashing_pin,           uint, S_IRUGO);  // 1 (default)
module_param(uvm_perf_thrashing_max_resets,    uint, S_IRUGO);  // 200 (default)
module_param(uvm_perf_map_remote_on_native_atomics_fault, uint, S_IRUGO);  // 1
```

---

## 5. Prefetch Policy

**File**: `uvm_perf_prefetch.h` and `uvm_perf_prefetch.c:59-61`

**Module Parameters**:
```c
module_param(uvm_perf_prefetch_enable,    uint, S_IRUGO);  // 1 (default)
module_param(uvm_perf_prefetch_threshold, uint, S_IRUGO);  // 51 (%)
module_param(uvm_perf_prefetch_min_faults, uint, S_IRUGO); // 2 (default)
```

---

## 6. Fault Batch Sizing & Service Policy

**File**: `uvm_gpu_replayable_faults.c:69-116`

```c
module_param(uvm_perf_fault_batch_count, uint, S_IRUGO);  // 256 (default)
MODULE_PARM_DESC(uvm_perf_fault_batch_count,
    "Number of GPU replayable faults to batch before servicing");

module_param(uvm_perf_fault_replay_update_put_ratio, uint, S_IRUGO);  // 50 (%)
MODULE_PARM_DESC(uvm_perf_fault_replay_update_put_ratio,
    "Ratio of duplicate faults to trigger PUT pointer update");

module_param(uvm_perf_fault_max_batches_per_service, uint, S_IRUGO);  // 20
MODULE_PARM_DESC(uvm_perf_fault_max_batches_per_service,
    "Maximum batches to service before exiting fault handler");

module_param(uvm_perf_fault_max_throttle_per_service, uint, S_IRUGO);  // 5
MODULE_PARM_DESC(uvm_perf_fault_max_throttle_per_service,
    "Maximum throttled batches before exiting");

module_param(uvm_perf_fault_coalesce, uint, S_IRUGO);  // 1 (default)
MODULE_PARM_DESC(uvm_perf_fault_coalesce,
    "Coalesce faults from different uTLBs within same page");

module_param(uvm_perf_reenable_prefetch_faults_lapse_msec, uint, S_IRUGO);  // 10000 (10s)
```

---

## 7. Fault Service Mode

**File**: `uvm_gpu_replayable_faults.c:566-574`

```c
typedef enum {
    // Service faults as normal
    UVM_FAULT_SERVICE_MODE_REGULAR = 0,

    // Cancel all faults (for fatal errors)
    UVM_FAULT_SERVICE_MODE_CANCEL,

    // Service faults for throttling check only
    UVM_FAULT_SERVICE_MODE_CANCEL_THROTTLING,
} fault_service_mode_t;
```

---

## 8. Fault Fetch Mode

**File**: `uvm_gpu_replayable_faults.c:824-834`

```c
typedef enum {
    // Only fetch faults that are ready (batch available)
    FAULT_FETCH_MODE_BATCH_READY,

    // Fetch all faults currently in buffer
    FAULT_FETCH_MODE_BATCH_ALL,

    // Fetch all faults (for shutdown/cancellation)
    FAULT_FETCH_MODE_ALL,
} fault_fetch_mode_t;
```

---

## 9. GPU Buffer Flush Mode

**File**: `uvm_gpu_replayable_faults.c:423-432`

```c
typedef enum {
    // Use cached PUT pointer (fast, may have stale faults)
    UVM_GPU_BUFFER_FLUSH_MODE_CACHED_PUT,

    // Update PUT pointer before flush (slower, fewer duplicates)
    UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
} uvm_gpu_buffer_flush_mode_t;
```

**Decision Point**: Line 2995 in `uvm_parent_gpu_service_replayable_faults()`
```c
if (batch_context->num_duplicate_faults * 100 >
    batch_context->num_cached_faults * replayable_faults->replay_update_put_ratio) {
    flush_mode = UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT;  // Policy decision!
}
```

---

## 10. Access Counter Policy

**File**: `uvm_gpu_access_counters.c:79-84`

```c
module_param(uvm_perf_access_counter_migration_enable, int, S_IRUGO);  // 1
MODULE_PARM_DESC(uvm_perf_access_counter_migration_enable,
    "Enable migrations based on access counter notifications");

module_param(uvm_perf_access_counter_batch_count, uint, S_IRUGO);  // 64
module_param(uvm_perf_access_counter_threshold, uint, S_IRUGO);    // 16
```

---

## 11. Migration Strategy Policy

**File**: `uvm_migrate.c:45-49, 52-57`

### 11.1 Migration Pass
```c
typedef enum {
    UVM_MIGRATE_PASS_FIRST,   // First pass: migrate pages
    UVM_MIGRATE_PASS_SECOND   // Second pass: handle special cases
} uvm_migrate_pass_t;
```

### 11.2 Migration Preunmap Policy
```c
module_param(uvm_perf_migrate_cpu_preunmap_enable, int, S_IRUGO);  // 1
MODULE_PARM_DESC(uvm_perf_migrate_cpu_preunmap_enable,
    "Unmap CPU pages before migration to avoid TLB shootdowns");

module_param(uvm_perf_migrate_cpu_preunmap_block_order, uint, S_IRUGO);  // 2
```

---

## 12. VA Block Transfer Mode

**File**: `uvm_va_block_types.h` (search for transfer_mode)

```c
typedef enum {
    UVM_VA_BLOCK_TRANSFER_MODE_MOVE,      // Move pages (source loses residency)
    UVM_VA_BLOCK_TRANSFER_MODE_COPY,      // Copy pages (source keeps residency)
} uvm_va_block_transfer_mode_t;
```

**Usage**: Tracked in migration events, affects residency bitmap updates

---

## 13. Memory Protection Policy

**File**: `uvm_hal_types.h:181-188`

```c
typedef enum {
    UVM_PROT_NONE,                 // No access
    UVM_PROT_READ_ONLY,            // Read-only access
    UVM_PROT_READ_WRITE,           // Read-write access
    UVM_PROT_READ_WRITE_ATOMIC,    // Read-write-atomic access
    UVM_PROT_MAX
} uvm_prot_t;
```

**Usage**: Page table mapping permissions, revocation policies

---

## 14. Memory Barrier Policy

**File**: `uvm_hal_types.h:192-197`

```c
typedef enum {
    UVM_MEMBAR_NONE,  // No memory barrier
    UVM_MEMBAR_GPU,   // GPU-scoped memory barrier
    UVM_MEMBAR_SYS,   // System-wide memory barrier
} uvm_membar_t;
```

---

## 15. Fault Access Type Policy

**File**: `uvm_hal_types.h:204-215`

```c
typedef enum {
    UVM_FAULT_ACCESS_TYPE_PREFETCH = 0,     // Prefetch fault
    UVM_FAULT_ACCESS_TYPE_READ,             // Read fault
    UVM_FAULT_ACCESS_TYPE_WRITE,            // Write fault
    UVM_FAULT_ACCESS_TYPE_ATOMIC_WEAK,      // Weak atomic
    UVM_FAULT_ACCESS_TYPE_ATOMIC_STRONG,    // Strong atomic
    UVM_FAULT_ACCESS_TYPE_COUNT
} uvm_fault_access_type_t;
```

**Usage**: Determines migration, mapping permissions

---

## 16. Fault Cancel Mode

**File**: `uvm_hal_types.h:262-271`

```c
typedef enum {
    UVM_FAULT_CANCEL_VA_MODE_ALL = 0,           // Cancel all accesses
    UVM_FAULT_CANCEL_VA_MODE_WRITE_AND_ATOMIC,  // Cancel write/atomic only
    UVM_FAULT_CANCEL_VA_MODE_COUNT
} uvm_fault_cancel_va_mode_t;
```

---

## 17. Fault Replay Type

**File**: `uvm_hal_types.h:455-467`

```c
typedef enum {
    // Replay starts (faults in-flight)
    UVM_FAULT_REPLAY_TYPE_START = 0,

    // Replay completes (all faults resolved/reissued)
    UVM_FAULT_REPLAY_TYPE_START_ACK_ALL,

    UVM_FAULT_REPLAY_TYPE_COUNT
} uvm_fault_replay_type_t;
```

**Usage**: Passed to `push_replay_on_gpu()` at decision points

---

## 18. GPU Caching Policy

**File**: `uvm_types.h:102-108, uvm_va_block.c:79-85`

```c
// From uvm_types.h
typedef enum {
    UvmGpuCachingTypeDefault = 0,
    UvmGpuCachingTypeForceUncached = 1,
    UvmGpuCachingTypeForceCached = 2,
    UvmGpuCachingTypeCount = 3
} UvmGpuCachingType;

// Module parameters for experimental GPU caching
module_param(uvm_exp_gpu_cache_peermem, uint, S_IRUGO);  // 0 (default)
MODULE_PARM_DESC(uvm_exp_gpu_cache_peermem,
    "Experimental: cache peer memory in GPU L2");

module_param(uvm_exp_gpu_cache_sysmem, uint, S_IRUGO);   // 0 (default)
MODULE_PARM_DESC(uvm_exp_gpu_cache_sysmem,
    "Experimental: cache system memory in GPU L2");
```

---

## 19. Channel & Copy Engine Policy

**File**: `uvm_channel.c:82-85`

```c
module_param(uvm_channel_num_gpfifo_entries, uint, S_IRUGO);  // 1024
module_param(uvm_channel_gpfifo_loc, charp, S_IRUGO);         // "auto"
module_param(uvm_channel_gpput_loc, charp, S_IRUGO);          // "auto"
module_param(uvm_channel_pushbuffer_loc, charp, S_IRUGO);     // "auto"
```

---

## 20. Memory Allocation Policy

**File**: `uvm_pmm_sysmem.c:31, uvm_pmm_gpu.c:183-192`

```c
// CPU chunk allocation sizes
module_param(uvm_cpu_chunk_allocation_sizes, uint, S_IRUGO | S_IWUSR);

// GPU memory management
module_param(uvm_global_oversubscription, int, S_IRUGO);      // 1 (allow)
MODULE_PARM_DESC(uvm_global_oversubscription,
    "Allow system memory oversubscription");

module_param(uvm_perf_pma_batch_nonpinned_order, uint, S_IRUGO);  // 0
```

---

## 21. ATS (Address Translation Service) Policy

**File**: `uvm_ats.c:30`

```c
module_param(uvm_ats_mode, int, S_IRUGO);  // 0 (disabled by default)
MODULE_PARM_DESC(uvm_ats_mode,
    "ATS mode: 0=disabled, 1=enabled");
```

---

## 22. Page Table Location Policy

**File**: `uvm_mmu.c:71`

```c
module_param(uvm_page_table_location, charp, S_IRUGO);  // "auto"
MODULE_PARM_DESC(uvm_page_table_location,
    "Location of page tables: auto, vidmem, sysmem");
```

---

## 23. Peer Copy Policy

**File**: `uvm_gpu.c:55`

```c
module_param(uvm_peer_copy, charp, S_IRUGO);  // "phys"
MODULE_PARM_DESC(uvm_peer_copy,
    "Peer copy mode: phys (physical), virt (virtual)");
```

---

## 24. VA Block Mapping Strategy

**File**: `uvm_va_block.c:63-70`

```c
module_param(uvm_fault_force_sysmem, int, S_IRUGO|S_IWUSR);  // 0
MODULE_PARM_DESC(uvm_fault_force_sysmem,
    "Force all fault migrations to system memory");

module_param(uvm_perf_map_remote_on_eviction, int, S_IRUGO);  // 0
MODULE_PARM_DESC(uvm_perf_map_remote_on_eviction,
    "Map remotely when pages are evicted");

module_param(uvm_block_cpu_to_cpu_copy_with_ce, int, S_IRUGO | S_IWUSR);  // 1
```

---

## 25. HMM (Heterogeneous Memory Management) Policy

**File**: `uvm_hmm.c:49`

```c
module_param(uvm_disable_hmm, bool, 0444);  // false (HMM enabled by default)
MODULE_PARM_DESC(uvm_disable_hmm,
    "Disable HMM (Heterogeneous Memory Management) support");
```

---

## 26. Perf Module System (Callback Framework)

**File**: `uvm_perf_module.h:50-68`

### 26.1 Perf Module Types
```c
typedef enum {
    UVM_PERF_MODULE_FIRST_TYPE     = 0,
    UVM_PERF_MODULE_TYPE_TEST      = UVM_PERF_MODULE_FIRST_TYPE,
    UVM_PERF_MODULE_TYPE_THRASHING,       // Thrashing detection
    UVM_PERF_MODULE_TYPE_ACCESS_COUNTERS, // Access counter-based migration
    UVM_PERF_MODULE_TYPE_COUNT,
} uvm_perf_module_type_t;
```

### 26.2 Perf Module Structure
```c
struct uvm_perf_module_struct {
    const char *name;
    uvm_perf_module_type_t type;
    uvm_perf_event_callback_t callbacks[UVM_PERF_EVENT_COUNT];  // 8 event types
};
```

**API Functions**:
- `uvm_perf_module_load()` - Load module into VA space
- `uvm_perf_module_unload()` - Unload module
- `uvm_perf_register_event_callback()` - Register callback
- `uvm_perf_unregister_event_callback()` - Unregister callback

---

## 27. Perf Event Types (Notification Points)

**File**: `uvm_perf_events.h:49-79`

```c
typedef enum {
    UVM_PERF_EVENT_BLOCK_DESTROY = 0,     // VA block destroyed
    UVM_PERF_EVENT_BLOCK_SHRINK,          // VA block shrunk
    UVM_PERF_EVENT_BLOCK_MUNMAP,          // VA block unmapped
    UVM_PERF_EVENT_RANGE_DESTROY,         // VA range destroyed
    UVM_PERF_EVENT_RANGE_SHRINK,          // VA range shrunk
    UVM_PERF_EVENT_MODULE_UNLOAD,         // Perf module unloaded
    UVM_PERF_EVENT_FAULT,                 // ⭐ GPU/CPU fault occurred
    UVM_PERF_EVENT_MIGRATION,             // ⭐ Page migration occurred
    UVM_PERF_EVENT_REVOCATION,            // ⭐ Permission revoked
    UVM_PERF_EVENT_COUNT,
} uvm_perf_event_t;
```

**Event Data**: Rich context provided in `uvm_perf_event_data_t` (see section 2 of MINIMAL_EBPF_INTEGRATION.md)

---

## 28. HAL Function Pointers (Hardware Abstraction)

**File**: `uvm_hal.h:779-811, 848-864`

### 28.1 Host HAL Structure
```c
struct uvm_host_hal_struct {
    uvm_hal_fault_buffer_replay_t replay_faults;            // ⭐ Replay hook
    uvm_hal_fault_cancel_global_t cancel_faults_global;     // Cancel all faults
    uvm_hal_fault_cancel_targeted_t cancel_faults_targeted; // Cancel specific faults
    uvm_hal_fault_cancel_va_t cancel_faults_va;             // Cancel VA faults
    // ... 60+ more function pointers
};
```

### 28.2 Fault Buffer HAL Structure
```c
struct uvm_fault_buffer_hal_struct {
    uvm_hal_fault_buffer_parse_entry_t parse_entry;         // Parse fault entry
    uvm_hal_fault_buffer_parse_non_replayable_entry_t parse_non_replayable_entry;
    uvm_hal_fault_buffer_init_t init_buffer;                // Initialize buffer
    uvm_hal_fault_buffer_write_get_t write_get;             // Update GET pointer
    uvm_hal_fault_buffer_write_put_t write_put;             // Update PUT pointer
    uvm_hal_fault_buffer_read_put_t read_put;               // Read PUT pointer
    uvm_hal_fault_buffer_read_get_t read_get;               // Read GET pointer
    uvm_hal_enable_replayable_faults_t enable_replayable_faults;
    uvm_hal_disable_replayable_faults_t disable_replayable_faults;
    uvm_hal_clear_replayable_faults_t clear_replayable_faults;
};
```

**Usage**: Architecture-specific implementations (Pascal, Volta, Ampere, Hopper, etc.)
**Potential**: Could add eBPF-aware HAL variant

---

## 29. Debug & Development Policies

**File**: `uvm_common.c:34-72, uvm_push.c:37-41`

```c
module_param(uvm_debug_prints, int, S_IRUGO|S_IWUSR);           // 0 (disabled)
module_param(uvm_release_asserts, int, S_IRUGO|S_IWUSR);        // 1 (enabled)
module_param(uvm_release_asserts_dump_stack, int, S_IRUGO|S_IWUSR);  // 0
module_param(uvm_release_asserts_set_global_error, int, S_IRUGO|S_IWUSR);  // 0
module_param(uvm_enable_builtin_tests, int, S_IRUGO);           // 0 (disabled)

// Push debugging
module_param(uvm_debug_enable_push_desc, uint, S_IRUGO|S_IWUSR);  // 0
module_param(uvm_debug_enable_push_acquire_info, uint, S_IRUGO|S_IWUSR);  // 0
```

---

## 30. Confidential Computing Policy

**File**: `uvm_conf_computing.c:66`

```c
module_param(uvm_conf_computing_channel_iv_rotation_limit, ulong, S_IRUGO);
MODULE_PARM_DESC(uvm_conf_computing_channel_iv_rotation_limit,
    "IV rotation limit for confidential computing channels");
```

---

## Summary: Total Policy Interfaces Exposed

| Category | Count | Examples |
|----------|-------|----------|
| **Fault Policies** | 8 | Replay policy, batch size, service mode, fetch mode |
| **VA Memory Policies** | 5 | Preferred location, accessed by, read duplication |
| **Migration Policies** | 9 | Cause types, preunmap strategy, transfer mode |
| **Thrashing Policies** | 9 | Hint types (PIN/THROTTLE), threshold parameters |
| **Prefetch Policies** | 3 | Enable, threshold, min faults |
| **Access Counter Policies** | 3 | Enable migration, batch count, threshold |
| **Memory Protection** | 4 | None, RO, RW, RWA |
| **HAL Function Pointers** | 60+ | Replay, cancel, parse, init callbacks |
| **Perf Event System** | 9 | Event types with full callback registration |
| **Module Parameters** | 50+ | Runtime-tunable sysfs parameters |

**Grand Total**: **150+ distinct policy decision points** already exposed!

---

## Key Takeaway for eBPF Integration

The UVM driver **already has extensive policy infrastructure**. Rather than creating new mechanisms, eBPF integration should:

1. ✅ **Leverage existing enums** - Add `UVM_PERF_FAULT_REPLAY_POLICY_EBPF` to existing `uvm_perf_fault_replay_policy_t`
2. ✅ **Use existing event system** - Attach to `uvm_perf_event_data_t` which already exposes fault/migration/revocation context
3. ✅ **Use existing perf module** - Add `UVM_PERF_MODULE_TYPE_EBPF` to existing callback framework
4. ✅ **Reuse existing hints** - `uvm_perf_thrashing_hint_t` already has PIN/THROTTLE types
5. ✅ **Preserve existing defaults** - All current policies remain as fallback behavior

**The minimal eBPF patch only needs to add ONE decision point at the fault replay logic (line 2986) and ONE callback to decide when to replay faults.**

All other policy interfaces are **already query-able** by eBPF programs via `uvm_perf_event_data_t` and module parameters!
