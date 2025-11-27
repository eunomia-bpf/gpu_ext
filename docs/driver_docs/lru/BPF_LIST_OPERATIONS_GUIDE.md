# BPF List Operations Implementation Guide

## Overview

This document analyzes how the Linux kernel implements BPF list operations and provides a comprehensive guide for implementing similar functionality in NVIDIA UVM for LRU-based GPU memory management.

**Key Concepts:**
- **Opaque Types**: BPF programs use opaque types (`bpf_list_head`, `bpf_list_node`) that hide kernel implementation details
- **Ownership Tracking**: Prevents use-after-free bugs by tracking which list head owns each node
- **Type Safety**: Compile-time verification ensures only valid operations are performed
- **Zero-Copy**: BPF operates directly on kernel data structures without marshaling

---

## 1. Linux Kernel BPF List Architecture

### 1.1 Type System

The Linux kernel uses a three-layer type system for BPF list operations:

#### Layer 1: BPF Program View (Opaque Types)

**File:** `/linux/include/uapi/linux/bpf.h`

```c
/* Opaque types visible to BPF programs */
struct bpf_list_head {
    __u64 __opaque[2];
} __attribute__((aligned(8)));

struct bpf_list_node {
    __u64 __opaque[3];
} __attribute__((aligned(8)));
```

**Purpose**: Hide kernel implementation details from BPF programs, providing a stable API across kernel versions.

#### Layer 2: Kernel Internal Representation

**File:** `/linux/include/linux/bpf.h:276-279`

```c
/* Internal kernel representation with ownership tracking */
struct bpf_list_node_kern {
    struct list_head list_head;    // Standard Linux list_head
    void *owner;                   // Pointer to bpf_list_head that owns this node
} __attribute__((aligned(8)));
```

**Purpose**: Track ownership to prevent:
- Adding a node to multiple lists simultaneously
- Popping a node from the wrong list
- Use-after-free bugs

#### Layer 3: Container Structures (User-Defined)

**File:** `/linux/tools/testing/selftests/bpf/progs/linked_list.h:9-20`

```c
struct foo {
    struct bpf_list_node node;       // Link for one list
    struct bpf_list_head head __contains(bar, node);  // Nested list head
    struct bpf_spin_lock lock;       // Required for thread safety
    int data;                        // User data
    struct bpf_list_node node2;      // Link for another list (optional)
};
```

**Key Annotation**: `__contains(type, field)` - BTF decl tag that tells the verifier:
- `head` can contain nodes of type `type`
- Nodes are linked via their `field` member

### 1.2 Available BPF List Kfuncs

**File:** `/linux/kernel/bpf/helpers.c:2289-2360`

All list operations are exposed as kfuncs (type-safe kernel functions):

| Kfunc | Purpose | Returns | Locking Required |
|-------|---------|---------|------------------|
| `bpf_list_push_front()` | Add node to list head | 0 on success, -EINVAL if already in list | Yes (spin_lock) |
| `bpf_list_push_back()` | Add node to list tail | 0 on success, -EINVAL if already in list | Yes (spin_lock) |
| `bpf_list_pop_front()` | Remove and return first node | `bpf_list_node*` or NULL | Yes (spin_lock) |
| `bpf_list_pop_back()` | Remove and return last node | `bpf_list_node*` or NULL | Yes (spin_lock) |
| `bpf_list_front()` | Peek at first node (non-destructive) | `bpf_list_node*` or NULL | No |
| `bpf_list_back()` | Peek at last node (non-destructive) | `bpf_list_node*` or NULL | No |

**Memory Management Kfuncs** (Required for Ownership Transfer):

```c
/* Allocate new object */
extern void *bpf_obj_new_impl(__u64 local_type_id, void *meta) __ksym;
#define bpf_obj_new(type) ((type *)bpf_obj_new_impl(bpf_core_type_id_local(type), NULL))

/* Free object and clear all fields */
extern void bpf_obj_drop_impl(void *kptr, void *meta) __ksym;
#define bpf_obj_drop(kptr) bpf_obj_drop_impl(kptr, NULL)
```

### 1.3 Implementation Details

#### Push Operation (`bpf_list_push_front/back`)

**File:** `/linux/kernel/bpf/helpers.c:2270-2307`

```c
static int __bpf_list_add(struct bpf_list_node_kern *node,
                          struct bpf_list_head *head, bool tail,
                          struct btf_record *rec, u64 off)
{
    struct list_head *n = &node->list_head;
    struct list_head *h = (void *)head;

    /* Safety check: Node must not already be owned */
    if (cmpxchg(&node->owner, NULL, head)) {
        /* Node already in a list - free it to prevent leak */
        __bpf_obj_drop_impl((void *)n - off, rec, false);
        return -EINVAL;
    }

    /* Initialize list head if it was zero-initialized by BPF map */
    if (unlikely(!h->next))
        INIT_LIST_HEAD(h);

    /* Add to list using standard Linux kernel list API */
    tail ? list_add_tail(n, h) : list_add(n, h);

    /* Set owner to track this node */
    WRITE_ONCE(node->owner, head);

    return 0;
}
```

**Key Points:**
- `cmpxchg(&node->owner, NULL, head)` - Atomic check-and-set ensures node not in another list
- Uses standard `list_add()` / `list_add_tail()` internally
- Ownership tracking prevents double-add bugs

#### Pop Operation (`bpf_list_pop_front/back`)

**File:** `/linux/kernel/bpf/helpers.c:2309-2340`

```c
static struct bpf_list_node *__bpf_list_del(struct bpf_list_head *head, bool tail)
{
    struct list_head *n, *h = (void *)head;
    struct bpf_list_node_kern *node;

    /* Initialize if needed */
    if (unlikely(!h->next))
        INIT_LIST_HEAD(h);

    /* Empty list check */
    if (list_empty(h))
        return NULL;

    /* Get first or last node */
    n = tail ? h->prev : h->next;
    node = container_of(n, struct bpf_list_node_kern, list_head);

    /* Verify ownership before removing */
    if (WARN_ON_ONCE(READ_ONCE(node->owner) != head))
        return NULL;

    /* Remove from list and clear owner */
    list_del_init(n);
    WRITE_ONCE(node->owner, NULL);

    return (struct bpf_list_node *)n;
}
```

**Key Points:**
- Ownership verification prevents popping from wrong list
- `list_del_init()` safely removes and reinitializes node
- Clearing owner allows node to be re-added elsewhere

---

## 2. Example: Linux Kernel BPF List Usage

### 2.1 Complete Example from Kernel Tests

**File:** `/linux/tools/testing/selftests/bpf/progs/linked_list.c:27-105`

```c
static __always_inline
int list_push_pop(struct bpf_spin_lock *lock, struct bpf_list_head *head, bool leave_in_map)
{
    struct bpf_list_node *n;
    struct foo *f;

    /* 1. Allocate new object */
    f = bpf_obj_new(typeof(*f));
    if (!f)
        return 2;

    /* 2. Check if list is empty (should be) */
    bpf_spin_lock(lock);
    n = bpf_list_pop_front(head);
    bpf_spin_unlock(lock);
    if (n) {
        bpf_obj_drop(container_of(n, struct foo, node2));
        bpf_obj_drop(f);
        return 3;  // Unexpected - list should be empty
    }

    /* 3. Add node to list */
    bpf_spin_lock(lock);
    f->data = 42;
    bpf_list_push_front(head, &f->node2);
    bpf_spin_unlock(lock);

    if (leave_in_map)
        return 0;

    /* 4. Remove node from list */
    bpf_spin_lock(lock);
    n = bpf_list_pop_back(head);
    bpf_spin_unlock(lock);
    if (!n)
        return 5;  // Error - list should have one node

    /* 5. Verify data */
    f = container_of(n, struct foo, node2);
    if (f->data != 42) {
        bpf_obj_drop(f);
        return 6;  // Data corruption
    }

    /* 6. Free the object */
    bpf_obj_drop(f);
    return 0;
}
```

**Critical Pattern**: `container_of(node, struct_type, field_name)`
- Converts `bpf_list_node*` back to full structure pointer
- Must specify correct structure type and field name
- Example: `container_of(n, struct foo, node2)` gets `foo*` from its `node2` field

### 2.2 Locking Requirements

**All destructive operations require locking:**

```c
/* CORRECT */
bpf_spin_lock(&lock);
bpf_list_push_front(&head, &node);
bpf_spin_unlock(&lock);

/* INCORRECT - Verifier will reject */
bpf_list_push_front(&head, &node);  // Error: lock not held
```

**Read-only operations (peek) don't require locks:**

```c
/* Allowed without lock (but may race) */
n = bpf_list_front(&head);
```

---

## 3. Adapting for NVIDIA UVM: Design Proposal

### 3.1 Current NVIDIA UVM List Operations

NVIDIA UVM currently uses standard Linux `list_head` directly:

**File:** `/kernel-open/nvidia-uvm/uvm_pmm_gpu.c:627-651`

```c
static void chunk_update_lists_locked(uvm_pmm_gpu_t *pmm,
                                      uvm_gpu_chunk_t *chunk,
                                      uvm_pmm_gpu_memory_type_t new_type,
                                      struct list_head *new_list)
{
    /* Current direct list manipulation */
    list_move_tail(&chunk->list, new_list);

    /* ... */
}
```

**Problem**: BPF programs cannot directly use `list_head` or call `list_move_tail()`.

### 3.2 Proposed BPF Kfunc Wrappers

To enable BPF control over LRU eviction, we need to expose list operations as kfuncs:

**File:** `kernel-open/nvidia-uvm/uvm_bpf_struct_ops.c` (additions)

```c
/* Opaque types for BPF programs */
struct uvm_bpf_list_head {
    __u64 __opaque[2];
} __attribute__((aligned(8)));

struct uvm_bpf_list_node {
    __u64 __opaque[3];
} __attribute__((aligned(8)));

/* Kfunc: Push chunk to front of list (MRU position) */
__bpf_kfunc int bpf_uvm_list_push_front(struct uvm_bpf_list_head *head,
                                        struct uvm_bpf_list_node *node)
{
    struct list_head *h = (struct list_head *)head;
    struct list_head *n = (struct list_head *)node;

    if (!h || !n)
        return -EINVAL;

    /* Move to front (most recently used) */
    list_move(n, h);
    return 0;
}

/* Kfunc: Push chunk to back of list (LRU position) */
__bpf_kfunc int bpf_uvm_list_push_back(struct uvm_bpf_list_head *head,
                                       struct uvm_bpf_list_node *node)
{
    struct list_head *h = (struct list_head *)head;
    struct list_head *n = (struct list_head *)node;

    if (!h || !n)
        return -EINVAL;

    /* Move to tail (least recently used) */
    list_move_tail(n, h);
    return 0;
}

/* Kfunc: Get first chunk from list (LRU victim) */
__bpf_kfunc struct uvm_bpf_list_node *
bpf_uvm_list_first(struct uvm_bpf_list_head *head)
{
    struct list_head *h = (struct list_head *)head;

    if (!h || list_empty(h))
        return NULL;

    return (struct uvm_bpf_list_node *)h->next;
}

/* Kfunc: Get last chunk from list (MRU) */
__bpf_kfunc struct uvm_bpf_list_node *
bpf_uvm_list_last(struct uvm_bpf_list_head *head)
{
    struct list_head *h = (struct list_head *)head;

    if (!h || list_empty(h))
        return NULL;

    return (struct uvm_bpf_list_node *)h->prev;
}

/* Kfunc: Get next chunk in list */
__bpf_kfunc struct uvm_bpf_list_node *
bpf_uvm_list_next(struct uvm_bpf_list_head *head,
                  struct uvm_bpf_list_node *node)
{
    struct list_head *h = (struct list_head *)head;
    struct list_head *n = (struct list_head *)node;

    if (!h || !n || n->next == h)
        return NULL;  // End of list

    return (struct uvm_bpf_list_node *)n->next;
}

/* Kfunc: Check if list is empty */
__bpf_kfunc bool bpf_uvm_list_empty(struct uvm_bpf_list_head *head)
{
    struct list_head *h = (struct list_head *)head;
    return !h || list_empty(h);
}

/* Register kfuncs */
BTF_KFUNCS_START(uvm_list_kfunc_ids)
BTF_ID_FLAGS(func, bpf_uvm_list_push_front)
BTF_ID_FLAGS(func, bpf_uvm_list_push_back)
BTF_ID_FLAGS(func, bpf_uvm_list_first)
BTF_ID_FLAGS(func, bpf_uvm_list_last)
BTF_ID_FLAGS(func, bpf_uvm_list_next)
BTF_ID_FLAGS(func, bpf_uvm_list_empty)
BTF_KFUNCS_END(uvm_list_kfunc_ids)

static const struct btf_kfunc_id_set uvm_list_kfunc_set = {
    .owner = THIS_MODULE,
    .set   = &uvm_list_kfunc_ids,
};
```

### 3.3 BPF Struct Ops Hook for LRU Selection

Extend `uvm_gpu_ext` structure with LRU eviction hook:

```c
struct uvm_gpu_ext {
    /* ... existing hooks ... */

    /**
     * @uvm_lru_select_victim - Select chunk for eviction from LRU list
     *
     * @pmm: GPU memory manager
     * @va_block_used_head: Head of va_block_used list (LRU list)
     * @va_block_unused_head: Head of va_block_unused list
     * @selected_chunk: Output - pointer to selected chunk's list node
     *
     * Called when kernel needs to evict a GPU chunk.
     * BPF program can iterate the list and select eviction victim.
     *
     * Return:
     *   0 - Use kernel default LRU policy (first chunk in list)
     *   1 - BPF selected a chunk (via selected_chunk output)
     *   2 - No suitable chunk found (try next list)
     */
    int (*uvm_lru_select_victim)(
        uvm_pmm_gpu_t *pmm,
        struct uvm_bpf_list_head *va_block_used_head,
        struct uvm_bpf_list_head *va_block_unused_head,
        struct uvm_bpf_list_node **selected_chunk);
};
```

---

## 4. Example: BPF-Based LRU Eviction Policy

### 4.1 Frequency-Aware LRU Policy

This example demonstrates a BPF program that tracks access frequency and avoids evicting hot pages:

**File:** `gpu_ext_policy/src/lru_frequency_aware.bpf.c`

```c
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_core_read.h>
#include "uvm_types.h"

char _license[] SEC("license") = "GPL";

/* BPF map: Track access frequency per chunk (GPU physical address -> count) */
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 10000);
    __type(key, u64);      // GPU physical address
    __type(value, u32);    // Access count
} chunk_access_freq SEC(".maps");

/* Helper: Get access frequency for a chunk */
static __always_inline u32 get_chunk_frequency(u64 gpu_phys_addr)
{
    u32 *freq = bpf_map_lookup_elem(&chunk_access_freq, &gpu_phys_addr);
    return freq ? *freq : 0;
}

/* Kfunc declarations (defined in kernel module) */
extern int bpf_uvm_list_push_front(struct uvm_bpf_list_head *head,
                                   struct uvm_bpf_list_node *node) __ksym;
extern int bpf_uvm_list_push_back(struct uvm_bpf_list_head *head,
                                  struct uvm_bpf_list_node *node) __ksym;
extern struct uvm_bpf_list_node *
    bpf_uvm_list_first(struct uvm_bpf_list_head *head) __ksym;
extern struct uvm_bpf_list_node *
    bpf_uvm_list_next(struct uvm_bpf_list_head *head,
                     struct uvm_bpf_list_node *node) __ksym;
extern bool bpf_uvm_list_empty(struct uvm_bpf_list_head *head) __ksym;

SEC("struct_ops/uvm_lru_select_victim")
int BPF_PROG(uvm_lru_select_victim,
             uvm_pmm_gpu_t *pmm,
             struct uvm_bpf_list_head *va_block_used_head,
             struct uvm_bpf_list_head *va_block_unused_head,
             struct uvm_bpf_list_node **selected_chunk)
{
    struct uvm_bpf_list_node *node, *coldest_node = NULL;
    u32 min_freq = 0xFFFFFFFF;
    int count = 0;

    /* Check if unused list has chunks (highest priority) */
    if (!bpf_uvm_list_empty(va_block_unused_head)) {
        *selected_chunk = bpf_uvm_list_first(va_block_unused_head);
        bpf_printk("LRU: Selected from unused list\\n");
        return 1;  // BPF selected a chunk
    }

    /* Iterate va_block_used list to find coldest chunk */
    node = bpf_uvm_list_first(va_block_used_head);

    #pragma unroll
    for (int i = 0; i < 100 && node; i++) {
        /* Get chunk pointer from list node
         * Note: This requires exposing chunk structure or providing kfunc */
        uvm_gpu_chunk_t *chunk = container_of(node, uvm_gpu_chunk_t, list);
        u64 gpu_addr = BPF_CORE_READ(chunk, address);

        u32 freq = get_chunk_frequency(gpu_addr);

        bpf_printk("LRU: Chunk %llu has frequency %u\\n", gpu_addr, freq);

        /* Track coldest chunk (lowest access frequency) */
        if (freq < min_freq) {
            min_freq = freq;
            coldest_node = node;
        }

        count++;
        node = bpf_uvm_list_next(va_block_used_head, node);
    }

    if (coldest_node) {
        *selected_chunk = coldest_node;
        bpf_printk("LRU: Selected coldest chunk (freq=%u, scanned=%d)\\n",
                   min_freq, count);
        return 1;  // BPF selected a chunk
    }

    /* Fallback to kernel default (first chunk) */
    bpf_printk("LRU: Fallback to kernel default\\n");
    return 0;
}

/* Hook to track chunk access (called from page fault handler) */
SEC("struct_ops/uvm_on_chunk_access")
int BPF_PROG(uvm_on_chunk_access, u64 gpu_phys_addr)
{
    u32 *freq = bpf_map_lookup_elem(&chunk_access_freq, &gpu_phys_addr);

    if (freq) {
        __sync_fetch_and_add(freq, 1);
    } else {
        u32 initial = 1;
        bpf_map_update_elem(&chunk_access_freq, &gpu_phys_addr,
                           &initial, BPF_NOEXIST);
    }

    return 0;
}

/* Define struct_ops map */
SEC(".struct_ops")
struct uvm_gpu_ext uvm_ops_lru_frequency = {
    .uvm_lru_select_victim = (void *)uvm_lru_select_victim,
    .uvm_on_chunk_access = (void *)uvm_on_chunk_access,
};
```

### 4.2 User-Space Program

**File:** `gpu_ext_policy/src/lru_frequency_aware.c`

```c
#include <stdio.h>
#include <signal.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include "lru_frequency_aware.skel.h"

static volatile bool exiting = false;

void handle_signal(int sig) {
    exiting = true;
}

int main(int argc, char **argv)
{
    struct lru_frequency_aware_bpf *skel;
    struct bpf_link *link;
    int err;

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    /* Open and load BPF program */
    skel = lru_frequency_aware_bpf__open_and_load();
    if (!skel) {
        fprintf(stderr, "Failed to open/load BPF skeleton\\n");
        return 1;
    }

    /* Register struct_ops */
    link = bpf_map__attach_struct_ops(skel->maps.uvm_ops_lru_frequency);
    if (!link) {
        fprintf(stderr, "Failed to attach struct_ops\\n");
        goto cleanup;
    }

    printf("Successfully loaded frequency-aware LRU policy!\\n");
    printf("Press Ctrl-C to exit...\\n");

    while (!exiting) {
        sleep(1);
    }

    printf("\\nDetaching...\\n");
    bpf_link__destroy(link);

cleanup:
    lru_frequency_aware_bpf__destroy(skel);
    return err < 0 ? -err : 0;
}
```

---

## 5. Integration with Kernel Module

### 5.1 Kernel Call Site Modification

**File:** `kernel-open/nvidia-uvm/uvm_pmm_gpu.c:1460-1500`

**Before** (current implementation):

```c
static uvm_gpu_root_chunk_t *pick_root_chunk_to_evict(uvm_pmm_gpu_t *pmm)
{
    uvm_gpu_chunk_t *chunk;

    // Priority 1: Free list
    chunk = list_first_chunk(find_free_list(pmm, ...));

    // Priority 2: Unused list
    if (!chunk)
        chunk = list_first_chunk(&pmm->root_chunks.va_block_unused);

    // Priority 3: LRU (least recently used)
    if (!chunk)
        chunk = list_first_chunk(&pmm->root_chunks.va_block_used);

    return chunk ? root_chunk_from_chunk(pmm, chunk) : NULL;
}
```

**After** (with BPF hook integration):

```c
static uvm_gpu_root_chunk_t *pick_root_chunk_to_evict(uvm_pmm_gpu_t *pmm)
{
    uvm_gpu_chunk_t *chunk = NULL;
    struct uvm_bpf_list_node *selected = NULL;
    int ret;

    // Priority 1: Free list (unchanged)
    chunk = list_first_chunk(find_free_list(pmm, ...));
    if (chunk)
        return root_chunk_from_chunk(pmm, chunk);

    /* Try BPF policy if registered */
    if (uvm_gpu_ext_registered()) {
        ret = uvm_gpu_ext_ops->uvm_lru_select_victim(
            pmm,
            (struct uvm_bpf_list_head *)&pmm->root_chunks.va_block_used,
            (struct uvm_bpf_list_head *)&pmm->root_chunks.va_block_unused,
            &selected
        );

        if (ret == 1 && selected) {
            /* BPF selected a chunk */
            chunk = container_of((struct list_head *)selected,
                               uvm_gpu_chunk_t, list);
            goto done;
        }

        if (ret == 2) {
            /* BPF said no suitable chunk */
            return NULL;
        }

        /* ret == 0: Fall through to default policy */
    }

    /* Default kernel policy */
    // Priority 2: Unused list
    if (!chunk)
        chunk = list_first_chunk(&pmm->root_chunks.va_block_unused);

    // Priority 3: LRU (least recently used)
    if (!chunk)
        chunk = list_first_chunk(&pmm->root_chunks.va_block_used);

done:
    return chunk ? root_chunk_from_chunk(pmm, chunk) : NULL;
}
```

---

## 6. Comparison: Linux BPF vs NVIDIA UVM Adaptation

| Aspect | Linux Kernel BPF Lists | NVIDIA UVM Adaptation |
|--------|------------------------|----------------------|
| **Type System** | Opaque types with ownership tracking | Cast existing `list_head` to opaque types |
| **Memory Management** | `bpf_obj_new()` / `bpf_obj_drop()` | Chunks managed by kernel, BPF only selects |
| **Locking** | `bpf_spin_lock` required for modifications | UVM already holds locks at call site |
| **Ownership Transfer** | BPF owns objects in lists | BPF only observes and reorders |
| **Operations** | Full CRUD (create, push, pop, delete) | Read-only iteration + reorder hints |
| **Verification** | BTF `__contains` annotation required | Simpler - cast to opaque, trust kernel |

---

## 7. Testing and Debugging

### 7.1 Debug Techniques

**1. BPF Tracing (bpf_printk)**

```c
bpf_printk("Selected chunk at addr=%llu, freq=%u\\n", addr, freq);
```

View output:
```bash
sudo cat /sys/kernel/debug/tracing/trace_pipe
```

**2. BPF Map Inspection**

```bash
# View access frequency map
sudo bpftool map dump name chunk_access_freq
```

**3. Kernel Log Integration**

Add kernel logging in the call site:
```c
if (selected) {
    uvm_info_print("BPF selected chunk: %p\\n", chunk);
}
```

### 7.2 Verification Tips

**Common BPF Verifier Errors:**

1. **"unbounded loop detected"**
   - Solution: Use `#pragma unroll` or bounded loops

2. **"invalid mem access 'inv'"**
   - Solution: Add NULL checks before dereferencing

3. **"operation not supported on list_node"**
   - Solution: Ensure list modifications are inside `bpf_spin_lock`

---

## 8. Performance Considerations

### 8.1 Overhead Analysis

| Operation | Cost | Notes |
|-----------|------|-------|
| BPF hook invocation | ~50-100ns | JIT-compiled, minimal overhead |
| List iteration (10 chunks) | ~500ns | Depends on chunk count |
| Map lookup (hash) | ~100-200ns | Per chunk if tracking metadata |
| `bpf_printk` | ~1-2µs | **Only for debugging** |

**Recommendation**: For production, disable `bpf_printk` and use BPF maps for statistics.

### 8.2 Optimization Strategies

1. **Limit iteration count**
   ```c
   #pragma unroll
   for (int i = 0; i < 100 && node; i++)  // Max 100 chunks
   ```

2. **Use BPF_MAP_TYPE_LRU_HASH for frequency tracking**
   - Automatic eviction of cold entries
   - No manual cleanup needed

3. **Batch updates**
   - Update access frequency in batches from user-space
   - Reduces map contention

---

## 9. Future Extensions

### 9.1 Additional Hooks

```c
struct uvm_gpu_ext {
    /* ... existing ... */

    /* Hook when chunk is accessed (for frequency tracking) */
    int (*uvm_on_chunk_access)(u64 gpu_phys_addr, u64 size);

    /* Hook before moving chunk between lists */
    int (*uvm_before_list_move)(uvm_gpu_chunk_t *chunk,
                               struct uvm_bpf_list_head *from,
                               struct uvm_bpf_list_head *to);

    /* Hook for custom eviction priority calculation */
    int (*uvm_calc_eviction_priority)(uvm_gpu_chunk_t *chunk,
                                     u32 *priority_out);
};
```

### 9.2 Machine Learning Integration

Potential for ML-based eviction policies:
- Collect access patterns via BPF
- Train model in user-space
- Deploy learned policy as BPF program

---

## 10. Summary

### Key Takeaways

1. **Linux Kernel Pattern**:
   - Opaque types for ABI stability
   - Ownership tracking for memory safety
   - Full CRUD operations via kfuncs

2. **NVIDIA UVM Adaptation**:
   - Simpler model: BPF observes and reorders
   - Kernel retains ownership of chunks
   - Focus on eviction policy flexibility

3. **Implementation Steps**:
   - Add kfunc wrappers in `uvm_bpf_struct_ops.c`
   - Extend `uvm_gpu_ext` with LRU hook
   - Modify `pick_root_chunk_to_evict()` to call BPF
   - Implement example policies (frequency-aware, working-set, etc.)

4. **Benefits**:
   - Hot-swappable eviction policies (no kernel rebuild)
   - Application-specific optimization
   - Safe: BPF verifier prevents crashes
   - Observable: BPF maps expose internal state

### References

- Linux BPF List Implementation: `linux/kernel/bpf/helpers.c:2280-2360`
- BPF Experimental Header: `linux/tools/testing/selftests/bpf/bpf_experimental.h`
- Kernel Test Examples: `linux/tools/testing/selftests/bpf/progs/linked_list.c`
- NVIDIA UVM LRU Code: `kernel-open/nvidia-uvm/uvm_pmm_gpu.c:1460-1500`
