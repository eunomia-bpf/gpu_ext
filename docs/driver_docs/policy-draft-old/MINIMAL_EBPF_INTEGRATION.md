# Minimal eBPF Integration for UVM Fault Handling

## Summary

This document identifies **already exposed policy interfaces** in the NVIDIA UVM driver and proposes a **minimal struct_ops integration** that adds only one policy type and one callback function without deleting any existing code.

## Already Exposed Policy Interfaces

### 1. Module Parameters (Runtime Tunable via sysfs)

The UVM driver already exposes **50+ module parameters**. Key fault-handling policy parameters:

#### Fault Replay Policy (Most Critical)
```c
// File: uvm_gpu_replayable_faults.c:83
static uvm_perf_fault_replay_policy_t uvm_perf_fault_replay_policy = UVM_PERF_FAULT_REPLAY_POLICY_DEFAULT;
module_param(uvm_perf_fault_replay_policy, uint, S_IRUGO);

// Enum definition: uvm_gpu_replayable_faults.h:34-51
typedef enum {
    UVM_PERF_FAULT_REPLAY_POLICY_BLOCK = 0,        // Replay after each block
    UVM_PERF_FAULT_REPLAY_POLICY_BATCH,            // Replay after each batch
    UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH,      // Batch + flush (default)
    UVM_PERF_FAULT_REPLAY_POLICY_ONCE,             // Replay once at end
    UVM_PERF_FAULT_REPLAY_POLICY_MAX,
} uvm_perf_fault_replay_policy_t;
```

**Usage**: Read at initialization (uvm_gpu_replayable_faults.c:196) and stored in:
```c
replayable_faults->replay_policy = uvm_perf_fault_replay_policy;
```

#### Other Exposed Fault Policy Parameters
```c
// Batch sizing
static unsigned uvm_perf_fault_batch_count = 256;                    // Line 78
module_param(uvm_perf_fault_batch_count, uint, S_IRUGO);

// PUT pointer update timing
static unsigned uvm_perf_fault_replay_update_put_ratio = 50;         // Line 101
module_param(uvm_perf_fault_replay_update_put_ratio, uint, S_IRUGO);

// Service limits
static unsigned uvm_perf_fault_max_batches_per_service = 20;         // Line 109
module_param(uvm_perf_fault_max_batches_per_service, uint, S_IRUGO);

static unsigned uvm_perf_fault_max_throttle_per_service = 5;         // Line 113
module_param(uvm_perf_fault_max_throttle_per_service, uint, S_IRUGO);

// Coalescing behavior
static unsigned uvm_perf_fault_coalesce = 1;                         // Line 116
module_param(uvm_perf_fault_coalesce, uint, S_IRUGO);
```

#### Thrashing Policy Parameters
```c
// File: uvm_perf_thrashing.c:307-318
module_param(uvm_perf_thrashing_enable,        uint, S_IRUGO);       // Enable/disable
module_param(uvm_perf_thrashing_threshold,     uint, S_IRUGO);       // Detection threshold
module_param(uvm_perf_thrashing_pin_threshold, uint, S_IRUGO);       // Pin threshold
module_param(uvm_perf_thrashing_lapse_usec,    uint, S_IRUGO);       // Time window
module_param(uvm_perf_thrashing_nap,           uint, S_IRUGO);       // NAP policy
module_param(uvm_perf_thrashing_epoch,         uint, S_IRUGO);       // Epoch length
module_param(uvm_perf_thrashing_pin,           uint, S_IRUGO);       // Pinning policy
```

#### Prefetch Policy Parameters
```c
// File: uvm_perf_prefetch.c:59-61
module_param(uvm_perf_prefetch_enable,    uint, S_IRUGO);
module_param(uvm_perf_prefetch_threshold, uint, S_IRUGO);
module_param(uvm_perf_prefetch_min_faults, uint, S_IRUGO);
```

### 2. Perf Event Callback System (Already Implemented)

The UVM driver has a **complete callback registration framework** in `uvm_perf_events.h`:

#### Event Types (Line 49-79)
```c
typedef enum {
    UVM_PERF_EVENT_BLOCK_DESTROY = 0,
    UVM_PERF_EVENT_BLOCK_SHRINK,
    UVM_PERF_EVENT_BLOCK_MUNMAP,
    UVM_PERF_EVENT_RANGE_DESTROY,
    UVM_PERF_EVENT_RANGE_SHRINK,
    UVM_PERF_EVENT_MODULE_UNLOAD,
    UVM_PERF_EVENT_FAULT,           // ⭐ GPU/CPU fault events
    UVM_PERF_EVENT_MIGRATION,       // ⭐ Page migration events
    UVM_PERF_EVENT_REVOCATION,      // ⭐ Permission revocation
    UVM_PERF_EVENT_COUNT,
} uvm_perf_event_t;
```

#### Event Data Structure (Line 83-225)
Rich fault context already exposed via `uvm_perf_event_data_t`:
```c
struct {
    uvm_va_space_t *space;           // VA space
    uvm_va_block_t *block;           // VA block
    uvm_processor_id_t proc_id;      // Faulting GPU/CPU
    uvm_processor_id_t preferred_location;

    // GPU fault details
    struct {
        uvm_fault_buffer_entry_t *buffer_entry;  // Full fault entry
        NvU32 batch_id;
        bool is_duplicate;
    } gpu;

    // CPU fault details
    struct {
        NvU64 fault_va;
        NvU32 cpu_num;
        bool is_write;
        NvU64 pc;
    } cpu;
} fault;
```

#### Callback Registration API (Line 259-285)
```c
// Function pointer type
typedef void (*uvm_perf_event_callback_t)(uvm_perf_event_t event_id,
                                          uvm_perf_event_data_t *event_data);

// Registration functions (already implemented)
NV_STATUS uvm_perf_register_event_callback(uvm_perf_va_space_events_t *va_space_events,
                                           uvm_perf_event_t event_id,
                                           uvm_perf_event_callback_t callback);

void uvm_perf_unregister_event_callback(uvm_perf_va_space_events_t *va_space_events,
                                        uvm_perf_event_t event_id,
                                        uvm_perf_event_callback_t callback);

// Notification (already implemented)
void uvm_perf_event_notify(uvm_perf_va_space_events_t *va_space_events,
                           uvm_perf_event_t event_id,
                           uvm_perf_event_data_t *event_data);
```

#### Helper Functions Already Exposed (Line 357-434)
```c
// GPU fault notification
void uvm_perf_event_notify_gpu_fault(uvm_perf_va_space_events_t *va_space_events,
                                     uvm_va_block_t *va_block,
                                     uvm_gpu_id_t gpu_id,
                                     uvm_processor_id_t preferred_location,
                                     uvm_fault_buffer_entry_t *buffer_entry,
                                     NvU32 batch_id,
                                     bool is_duplicate);

// Migration notification
void uvm_perf_event_notify_migration(...);

// Revocation notification
void uvm_perf_event_notify_revocation(...);
```

### 3. Perf Module Type System

The driver has an **extensible module type enum** (uvm_perf_module.h:50-59):

```c
typedef enum {
    UVM_PERF_MODULE_FIRST_TYPE     = 0,
    UVM_PERF_MODULE_TYPE_TEST      = UVM_PERF_MODULE_FIRST_TYPE,
    UVM_PERF_MODULE_TYPE_THRASHING,
    UVM_PERF_MODULE_TYPE_ACCESS_COUNTERS,
    UVM_PERF_MODULE_TYPE_COUNT,
} uvm_perf_module_type_t;
```

**⭐ Can easily add**: `UVM_PERF_MODULE_TYPE_EBPF`

### 4. Thrashing Hint API (Already Exposed)

File: `uvm_perf_thrashing.h:33-74`

```c
typedef enum {
    UVM_PERF_THRASHING_HINT_TYPE_NONE     = 0,
    UVM_PERF_THRASHING_HINT_TYPE_PIN      = 1,   // Pin pages to avoid migration
    UVM_PERF_THRASHING_HINT_TYPE_THROTTLE = 2,   // Throttle migrations
} uvm_perf_thrashing_hint_type_t;

// Query function (already implemented)
uvm_perf_thrashing_hint_t uvm_perf_thrashing_get_hint(uvm_va_space_t *va_space,
                                                      uvm_va_block_t *va_block,
                                                      NvU64 address,
                                                      uvm_processor_id_t processor_id);
```

---

## Minimal eBPF struct_ops Integration

Based on the example pattern (`/home/yunwei37/bpf-developer-tutorial/src/features/struct_ops/module/hello.c`) and existing exposed interfaces, here's the minimal approach:

### Design Principles
1. ✅ **Add one policy type enum value**
2. ✅ **Add one callback function**
3. ✅ **Zero code deletion** - keep all existing code as default behavior
4. ✅ **Leverage existing `uvm_perf_fault_replay_policy` enum**
5. ✅ **Use existing `uvm_perf_event_data_t` structure**
6. ✅ **Follow RCU pattern from example**

### Implementation: Add ONE Enum Value

**File**: `kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.h`

```c
typedef enum {
    UVM_PERF_FAULT_REPLAY_POLICY_BLOCK = 0,
    UVM_PERF_FAULT_REPLAY_POLICY_BATCH,
    UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH,
    UVM_PERF_FAULT_REPLAY_POLICY_ONCE,
    UVM_PERF_FAULT_REPLAY_POLICY_EBPF,        // [+] Add this one line
    UVM_PERF_FAULT_REPLAY_POLICY_MAX,
} uvm_perf_fault_replay_policy_t;
```

### Implementation: Define ONE struct_ops

**New file**: `kernel-open/nvidia-uvm/uvm_fault_policy_ops.h`

```c
#ifndef __UVM_FAULT_POLICY_OPS_H__
#define __UVM_FAULT_POLICY_OPS_H__

#include "uvm_perf_events.h"

// eBPF struct_ops for fault replay policy
struct uvm_fault_policy_ops {
    // Returns: 0 = use default policy, 1 = replay now, -1 = defer replay
    int (*decide_replay)(uvm_perf_event_data_t *fault_data);
};

#endif
```

### Implementation: Add ONE Global Variable + Registration

**File**: `kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c`

```c
#include <linux/bpf.h>
#include <linux/btf.h>
#include "uvm_fault_policy_ops.h"

// [+] Add global RCU pointer (one line)
static struct uvm_fault_policy_ops __rcu *g_fault_policy_ops;

// [+] Add CFI stub (default behavior = return 0)
static int uvm_fault_policy_ops__decide_replay(uvm_perf_event_data_t *fault_data)
{
    return 0;  // Use default policy
}

// [+] Add CFI stubs structure
static struct uvm_fault_policy_ops __bpf_ops_uvm_fault_policy_ops = {
    .decide_replay = uvm_fault_policy_ops__decide_replay,
};

// [+] Add registration/unregistration
static int uvm_fault_policy_ops_reg(void *kdata, struct bpf_link *link)
{
    struct uvm_fault_policy_ops *ops = kdata;

    if (cmpxchg(&g_fault_policy_ops, NULL, ops) != NULL)
        return -EEXIST;

    pr_info("UVM: eBPF fault policy registered\n");
    return 0;
}

static void uvm_fault_policy_ops_unreg(void *kdata, struct bpf_link *link)
{
    struct uvm_fault_policy_ops *ops = kdata;

    if (cmpxchg(&g_fault_policy_ops, ops, NULL) != ops)
        return;

    pr_info("UVM: eBPF fault policy unregistered\n");
}

// [+] Add struct_ops definition
static struct bpf_struct_ops bpf_uvm_fault_policy_ops = {
    .verifier_ops = &uvm_fault_policy_verifier_ops,  // Define separately
    .init = uvm_fault_policy_ops_init,
    .init_member = uvm_fault_policy_ops_init_member,
    .reg = uvm_fault_policy_ops_reg,
    .unreg = uvm_fault_policy_ops_unreg,
    .cfi_stubs = &__bpf_ops_uvm_fault_policy_ops,
    .name = "uvm_fault_policy_ops",
    .owner = THIS_MODULE,
};

// [+] Register in module init
static int __init uvm_module_init(void)
{
    // ... existing init code ...

    ret = register_bpf_struct_ops(&bpf_uvm_fault_policy_ops, uvm_fault_policy_ops);
    if (ret)
        pr_warn("UVM: Failed to register eBPF struct_ops: %d\n", ret);

    return 0;
}
```

### Implementation: Call Hook at ONE Decision Point

**File**: `kernel-open/nvidia-uvm/uvm_gpu_replayable_faults.c:2906-3015`

Current code (simplified):
```c
static void uvm_parent_gpu_service_replayable_faults(uvm_parent_gpu_t *parent_gpu)
{
    // ... fetch faults ...
    // ... service faults ...

    // Decision point: when to replay?
    switch (replayable_faults->replay_policy) {
        case UVM_PERF_FAULT_REPLAY_POLICY_BLOCK:
            // Replay after each block
            break;
        case UVM_PERF_FAULT_REPLAY_POLICY_BATCH:
            // Replay after batch
            break;
        case UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH:
            // Replay with flush
            break;
        case UVM_PERF_FAULT_REPLAY_POLICY_ONCE:
            // Replay once at end
            break;
    }
}
```

**Modified code** (adds 12 lines, deletes 0 lines):
```c
static void uvm_parent_gpu_service_replayable_faults(uvm_parent_gpu_t *parent_gpu)
{
    // ... fetch faults ...
    // ... service faults ...

    // [+] Check if eBPF policy is enabled
    if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_EBPF) {
        struct uvm_fault_policy_ops *ops;
        int decision;

        rcu_read_lock();
        ops = rcu_dereference(g_fault_policy_ops);
        if (ops && ops->decide_replay) {
            decision = ops->decide_replay(&fault_event_data);
            rcu_read_unlock();

            if (decision > 0) {
                // eBPF says replay now
                push_replay_on_gpu(parent_gpu, ...);
                return;
            } else if (decision < 0) {
                // eBPF says defer
                return;
            }
            // decision == 0: fall through to default
        } else {
            rcu_read_unlock();
        }
    }

    // [=] Original code - unchanged, serves as default behavior
    switch (replayable_faults->replay_policy) {
        case UVM_PERF_FAULT_REPLAY_POLICY_BLOCK:
            // Replay after each block
            break;
        case UVM_PERF_FAULT_REPLAY_POLICY_BATCH:
            // Replay after batch
            break;
        case UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH:
            // Replay with flush
            break;
        case UVM_PERF_FAULT_REPLAY_POLICY_ONCE:
            // Replay once at end
            break;
    }
}
```

---

## Usage Example

### 1. Enable eBPF Policy via Module Parameter

```bash
# At module load time
sudo modprobe nvidia-uvm uvm_perf_fault_replay_policy=4  # 4 = EBPF policy

# Or at runtime (if parameter is writable)
echo 4 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_replay_policy
```

### 2. eBPF Program Example

```c
// user_ebpf_policy.bpf.c
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>

struct uvm_fault_policy_ops {
    int (*decide_replay)(void *fault_data);
};

SEC(".struct_ops.link")
int BPF_PROG(my_decide_replay, void *fault_data)
{
    // Custom policy logic
    // Return 1 to replay immediately
    // Return 0 to use default policy
    // Return -1 to defer replay

    // Example: always replay immediately for GPU 0
    // (real code would inspect fault_data)
    return 1;
}

SEC(".struct_ops")
struct uvm_fault_policy_ops my_uvm_policy = {
    .decide_replay = (void *)my_decide_replay,
};

char LICENSE[] SEC("license") = "GPL";
```

### 3. Load eBPF Program

```bash
# Compile
clang -O2 -target bpf -c user_ebpf_policy.bpf.c -o user_ebpf_policy.o

# Load (using libbpf)
bpftool struct_ops register user_ebpf_policy.o
```

---

## Summary of Changes

### Lines Added: ~50 lines total
- 1 enum value
- 1 struct definition (3 lines)
- 1 global variable
- ~15 lines of registration/CFI stubs
- ~12 lines of conditional check at decision point
- ~20 lines of boilerplate (verifier, init functions)

### Lines Deleted: 0 lines

### Files Modified: 2 files
1. `uvm_gpu_replayable_faults.h` - add enum value
2. `uvm_gpu_replayable_faults.c` - add struct_ops registration + conditional check

### Files Added: 1 file
1. `uvm_fault_policy_ops.h` - struct_ops definition

---

## Why This is Minimal

1. **Leverages existing module parameter infrastructure** - `uvm_perf_fault_replay_policy` already exists
2. **Leverages existing event data structure** - `uvm_perf_event_data_t` already exposes all fault context
3. **Zero code deletion** - all original policies remain intact as defaults
4. **Single callback function** - only `decide_replay()`, not 20+ hooks
5. **Single decision point** - fault replay timing (most critical for WIC paper)
6. **Follows kernel pattern** - same RCU pattern as the example struct_ops module

---

## Advantages Over Previous Design

| Aspect | Previous Design (EBPF_HOOKS_DESIGN.md) | Minimal Design |
|--------|----------------------------------------|----------------|
| Enum values added | 1 new enum type | 1 value in existing enum |
| Callback functions | 20+ hooks | 1 hook |
| Code deleted | Proposed `[-]` deletions | 0 deletions |
| Files modified | 7+ files | 2 files |
| Lines of code | 200+ lines | ~50 lines |
| Integration complexity | High - new infrastructure | Low - uses existing enum |
| Default behavior | Would require rewiring | Preserved automatically |

---

## What Policy Interfaces Are Already Exposed?

**Answer to user's question**: "已经有什么 policy 的interface暴露出来了？"

The UVM driver already exposes:

1. ✅ **Fault replay policy enum** - `uvm_perf_fault_replay_policy` (4 existing policies)
2. ✅ **50+ module parameters** - runtime tunable via sysfs
3. ✅ **Complete event callback system** - `uvm_perf_event_callback_t` with FAULT/MIGRATION/REVOCATION events
4. ✅ **Rich fault context** - `uvm_perf_event_data_t` with GPU/CPU fault details
5. ✅ **Thrashing hint API** - `uvm_perf_thrashing_get_hint()` with PIN/THROTTLE hints
6. ✅ **Perf module type system** - extensible enum for new policy modules

**The minimal eBPF integration only adds one callback to the existing replay policy mechanism.**
