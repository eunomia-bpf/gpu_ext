# NVIDIA Driver Symbol Tracing Investigation

## Overview

This document chronicles the investigation into why certain NVIDIA GPU driver functions cannot be traced with bpftrace, despite having full source code available. The investigation revealed a subtle but critical build-time filtering mechanism.

## Initial Problem Statement

**Question**: Why can't we trace internal NVIDIA driver functions like `kchannelCtrlCmdGpFifoSchedule_IMPL` with bpftrace, even though:
1. The symbols appear in `/proc/kallsyms`
2. We have the full source code in this repository
3. We can trace UVM module functions successfully

**Reference**: Functions documented in `docs/tsg_runlist_insertion.log`:
- `kchannelCtrlCmdGpFifoSchedule_IMPL` (kernel_channel.c:3059)
- `kchannelBindToRunlist_IMPL` (kernel_channel.c:2824)
- `kfifoRunlistSetId_GM107` (kernel_fifo_gm107.c:399)
- `kfifoRunlistSetIdByEngine_HAL` (kernel_fifo_gm107.c:467)

## Investigation Journey

### Step 1: Symbol Export Hypothesis (INCORRECT)

**Initial Theory**: Symbols are not exported with `EXPORT_SYMBOL()` macros.

**Evidence**:
```bash
$ grep EXPORT_SYMBOL kernel-open/nvidia/nv_uvm_interface.c | wc -l
76  # Many symbols exported

$ grep EXPORT_SYMBOL kernel-open/nvidia-uvm/*.c | wc -l
0   # UVM has no explicit exports
```

**Finding**: Export symbols are only for inter-module dependencies, not tracing. This was a red herring.

### Step 2: Symbol Visibility in kallsyms

**Observation**: All symbols appear in `/proc/kallsyms` but with address `0000000000000000`:

```bash
$ cat /proc/kallsyms | grep "kchannelCtrlCmdGpFifoSchedule_IMPL"
0000000000000000 t kchannelCtrlCmdGpFifoSchedule_IMPL	[nvidia]
```

**Symbol Type**: `t` = local text symbol (not global `T`)

**Key Discovery**: The `0000000000000000` address is due to `kptr_restrict=1` security setting:

```bash
$ cat /proc/sys/kernel/kptr_restrict
1  # Hides addresses from unprivileged users

$ sudo cat /proc/kallsyms | grep "kchannelCtrlCmdGpFifoSchedule_IMPL"
ffffffffc11e07a0 t kchannelCtrlCmdGpFifoSchedule_IMPL	[nvidia]
```

✅ **Real addresses exist**, so this isn't the root cause either.

### Step 3: Understanding ftrace and available_filter_functions

**Critical File**: `/sys/kernel/debug/tracing/available_filter_functions`

This file lists **only functions that can be traced by kprobes/ftrace**.

**Statistics**:
```bash
# Total traceable functions in system
$ sudo cat /sys/kernel/debug/tracing/available_filter_functions | wc -l
91493

# nvidia module
$ sudo cat /proc/kallsyms | grep -c " t .*\[nvidia\]"
17104  # Total symbols

$ sudo cat /sys/kernel/debug/tracing/available_filter_functions | grep -c "\[nvidia\]"
885    # Only 5% are traceable!

# nvidia-uvm module
$ sudo cat /proc/kallsyms | grep -c " t .*\[nvidia_uvm\]"
4726   # Total symbols

$ sudo cat /sys/kernel/debug/tracing/available_filter_functions | grep -c "\[nvidia_uvm\]"
2086   # 44% are traceable
```

**Massive Discrepancy**: 95% of nvidia module functions are NOT traceable!

### Step 4: Testing Traceability

**Attempt to trace the function**:
```bash
$ sudo bpftrace -e 'kprobe:kchannelCtrlCmdGpFifoSchedule_IMPL { printf("Hit\n"); }'
stdin:1:1-42: WARNING: kchannelCtrlCmdGpFifoSchedule_IMPL is not traceable
(either non-existing, inlined, or marked as "notrace")
Attaching 1 probe...
cannot attach kprobe, Invalid argument
ERROR: Error attaching probe: 'kprobe:kchannelCtrlCmdGpFifoSchedule_IMPL'
```

**Verification**:
```bash
$ sudo grep "kchannelCtrlCmdGpFifoSchedule_IMPL" \
  /sys/kernel/debug/tracing/available_filter_functions
# No output - function is NOT in the traceable list
```

**Comparison with working functions**:
```bash
$ sudo cat /sys/kernel/debug/tracing/available_filter_functions | \
  grep "\[nvidia\]" | grep -i channel | head -5
nv_restore_user_channels [nvidia]
os_imex_channel_count [nvidia]
os_imex_channel_get [nvidia]
nv_caps_imex_channel_get [nvidia]
nv_caps_imex_channel_count [nvidia]
```

These functions CAN be traced because they're in `available_filter_functions`.

## Root Cause Discovery

### Repository Structure

The repository has two distinct parts:

```
open-gpu-kernel-modules/
├── kernel-open/          # Linux kernel interface (Kbuild)
│   ├── nvidia/           # Kernel module wrapper
│   ├── nvidia-uvm/       # UVM driver (fully open, fully traceable)
│   └── nvidia-modeset/   # Modeset wrapper
│
└── src/                  # OS-agnostic core implementation
    ├── nvidia/           # Core GPU driver logic
    │   └── Makefile      # ⚠️ Custom build system
    └── nvidia-modeset/   # Core modeset logic
```

### The Build Process

From `README.md` and top-level `Makefile`:

1. **Phase 1**: Build `nv-kernel.o` from source
   ```makefile
   # Makefile (line 35-36)
   $(nv_kernel_o):
       $(MAKE) -C src/nvidia
   ```

2. **Phase 2**: Create symlink as "binary"
   ```makefile
   # Makefile (line 38-39)
   $(nv_kernel_o_binary): $(nv_kernel_o)
       cd $(dir $@) && ln -sf ../../$^ $(notdir $@)
   ```

   This creates: `kernel-open/nvidia/nv-kernel.o_binary` → `src/nvidia/_out/Linux_x86_64/nv-kernel.o`

3. **Phase 3**: Link into final module
   ```makefile
   # kernel-open/nvidia/nvidia.Kbuild (line 40-48)
   NVIDIA_BINARY_OBJECT := $(src)/nvidia/nv-kernel.o_binary
   NVIDIA_BINARY_OBJECT_O := nvidia/nv-kernel.o

   $(obj)/$(NVIDIA_BINARY_OBJECT_O): $(NVIDIA_BINARY_OBJECT) FORCE
       $(call if_changed,symlink)

   nvidia-y += $(NVIDIA_BINARY_OBJECT_O)
   ```

### The Critical Difference: Custom Build vs. Kbuild

**src/nvidia/Makefile** uses a completely custom build system:

```makefile
# src/nvidia/Makefile (excerpts)
CFLAGS += -I kernel/inc
CFLAGS += -I interface
CFLAGS += -Werror-implicit-function-declaration
CFLAGS += -Wwrite-strings
CFLAGS += -ffreestanding
CFLAGS += -fno-stack-protector
CFLAGS += -msoft-float   # (on x86_64)

# Build objects with their own rules
$(NV_KERNEL_O): $(OBJS) $(EXPORTS_LINK_COMMAND) $(LINKER_SCRIPT)
    $(call quiet_cmd,LD) \
      $(NV_KERNEL_O_LDFLAGS) \
      -T $(LINKER_SCRIPT) \
      -r -o $(NV_KERNEL_O) $(OBJS) @$(EXPORTS_LINK_COMMAND)
```

**Key Problem**: This Makefile:
- ❌ Does NOT use kernel's Kbuild infrastructure during compilation
- ❌ Does NOT inherit kernel's CONFIG_DYNAMIC_FTRACE flags
- ❌ Does NOT add `-pg` or `-mfentry` for ftrace instrumentation
- ❌ Uses custom CFLAGS (`-ffreestanding`, `-fno-stack-protector`)

**In contrast, nvidia-uvm**:

```makefile
# kernel-open/nvidia-uvm/nvidia-uvm.Kbuild
obj-m += nvidia-uvm.o
nvidia-uvm-y := $(NVIDIA_UVM_OBJECTS)

# Objects compiled via standard Kbuild
NVIDIA_UVM_OBJECTS += $(patsubst %.c,%.o,$(NVIDIA_UVM_SOURCES))
```

✅ Uses kernel Kbuild → ✅ Gets `-pg`/`-mfentry` automatically → ✅ Functions are traceable

## How ftrace Instrumentation Works

When `CONFIG_DYNAMIC_FTRACE=y` is set in kernel config, the kernel build system automatically adds compiler flags for function tracing:

```c
// Original function:
void my_function(void) {
    // function body
}

// With -pg or -mfentry, compiler generates:
void my_function(void) {
    __fentry__();  // ← Tracing hook inserted at function entry
    // function body
}
```

This `__fentry__()` call:
1. Initially does nothing (NOP instructions)
2. Can be dynamically patched to call tracing code
3. Is what makes the function appear in `available_filter_functions`
4. Enables kprobes, bpftrace, perf, etc. to attach

**Functions built with src/nvidia/Makefile don't get this instrumentation!**

## Evidence Summary

| Aspect | nvidia.ko (src/nvidia/) | nvidia-uvm.ko |
|--------|------------------------|---------------|
| **Build System** | Custom Makefile | Kernel Kbuild |
| **Ftrace Instrumentation** | ❌ No | ✅ Yes |
| **Total Symbols** | 17,104 | 4,726 |
| **Traceable Symbols** | 885 (5%) | 2,086 (44%) |
| **Source Available** | ✅ Yes | ✅ Yes |
| **Can Trace Internal Functions** | ❌ No | ✅ Yes |

## Detailed Symbol Analysis

### Functions We WANT to trace (from tsg_runlist_insertion.log):

```bash
$ sudo cat /proc/kallsyms | grep -E "kchannelCtrlCmdGpFifoSchedule|kchannelBindToRunlist|kfifoRunlistSetId"
ffffffffc11e07a0 t kchannelCtrlCmdGpFifoSchedule_IMPL	[nvidia]
ffffffffc11e0080 t kchannelBindToRunlist_IMPL	[nvidia]
ffffffffc1053790 t kfifoRunlistSetIdByEngine_GM107	[nvidia]
ffffffffc11dfda0 t kfifoRunlistSetId_GM107	[nvidia]
```

✅ Symbols exist with real addresses
❌ NOT in `available_filter_functions`
❌ Cannot be traced

### Functions we CAN trace (from open-source wrappers):

```bash
$ sudo cat /sys/kernel/debug/tracing/available_filter_functions | grep "\[nvidia\]" | head -10
nv_platform_device_probe [nvidia]
nv_platform_device_remove_wrapper [nvidia]
nvidia_soc_isr_kthread_bh [nvidia]
nv_restore_user_channels [nvidia]
nvidia_p2p_init_mapping [nvidia]
nvidia_p2p_destroy_mapping [nvidia]
...
```

These are from files in `kernel-open/nvidia/*.c` that were compiled via Kbuild.

## Source Code Location Mapping

| Function | Source File | Build System |
|----------|-------------|--------------|
| `kchannelCtrlCmdGpFifoSchedule_IMPL` | `src/nvidia/src/kernel/gpu/fifo/kernel_channel.c:3059` | Custom Makefile ❌ |
| `kchannelBindToRunlist_IMPL` | `src/nvidia/src/kernel/gpu/fifo/kernel_channel.c:2824` | Custom Makefile ❌ |
| `kfifoRunlistSetId_GM107` | `src/nvidia/src/kernel/gpu/fifo/arch/maxwell/kernel_fifo_gm107.c:399` | Custom Makefile ❌ |
| `nv_restore_user_channels` | `kernel-open/nvidia/nv-*.c` | Kbuild ✅ |
| `nvidia_p2p_*` | `kernel-open/nvidia/nv-p2p.c` | Kbuild ✅ |
| `uvm_*` functions | `kernel-open/nvidia-uvm/*.c` | Kbuild ✅ |

## Solution: Enable Tracing for nvidia.ko Functions

To make the TSG/channel/fifo functions traceable, you need to modify the custom build system to add ftrace instrumentation.

### Option 1: Add -pg flag to src/nvidia/Makefile

```makefile
# Add to src/nvidia/Makefile after line 77

# Enable ftrace/kprobe support
CFLAGS += -pg
# Or for newer kernels (better performance):
# CFLAGS += -mfentry -mrecord-mcount
```

**Caveats**:
- May conflict with `-ffreestanding` flag
- May require removing `-fno-stack-protector`
- Needs testing to ensure it doesn't break driver functionality

### Option 2: Rebuild with Kbuild integration

Restructure `src/nvidia/` to be built directly through kernel's Kbuild system instead of custom Makefile. This is a massive undertaking but would provide:
- Proper ftrace support
- Better kernel integration
- Standard kernel module conventions

### Option 3: Use alternative tracing methods

Since you have the source code:

1. **Add manual tracepoints**:
   ```c
   // In kernel_channel.c
   #include <trace/events/nvidia_trace.h>

   NV_STATUS kchannelCtrlCmdGpFifoSchedule_IMPL(...) {
       trace_nvidia_channel_schedule(pKernelChannel, pSchedParams);
       // ... rest of function
   }
   ```

2. **Add eBPF-friendly hooks**:
   ```c
   // Expose key state via global variables or BTF
   ```

3. **Use existing traceable functions**:
   Focus tracing on the 885 functions that ARE traceable, often these are wrappers that call the non-traceable internal functions.

## Testing and Validation

### Verify current state:

```bash
# Check if function has ftrace instrumentation
$ sudo cat /sys/kernel/debug/tracing/available_filter_functions | \
  grep kchannelCtrlCmdGpFifoSchedule_IMPL
# (no output = not traceable)

# Try to trace
$ sudo bpftrace -e 'kprobe:kchannelCtrlCmdGpFifoSchedule_IMPL { @[comm] = count(); }'
# Will fail: "cannot attach kprobe, Invalid argument"
```

### After rebuilding with -pg:

```bash
# Rebuild the module
$ make clean
$ # Edit src/nvidia/Makefile to add CFLAGS += -pg
$ make modules -j$(nproc)
$ sudo rmmod nvidia-uvm nvidia-modeset nvidia-drm nvidia
$ sudo make modules_install
$ sudo modprobe nvidia

# Verify function is now traceable
$ sudo cat /sys/kernel/debug/tracing/available_filter_functions | \
  grep kchannelCtrlCmdGpFifoSchedule_IMPL
kchannelCtrlCmdGpFifoSchedule_IMPL [nvidia]

# Should now work
$ sudo bpftrace -e 'kprobe:kchannelCtrlCmdGpFifoSchedule_IMPL {
    printf("Channel schedule called by %s\n", comm);
}'
```

## Key Takeaways

1. **Symbol visibility ≠ Traceability**
   - Just because a symbol is in `/proc/kallsyms` doesn't mean you can trace it
   - Must be in `/sys/kernel/debug/tracing/available_filter_functions`

2. **Build system matters**
   - Custom Makefiles bypass kernel's ftrace infrastructure
   - nvidia.ko: 5% traceable (custom build)
   - nvidia-uvm.ko: 44% traceable (Kbuild)

3. **Source availability ≠ Traceability**
   - Having source code doesn't automatically mean functions are traceable
   - Must be compiled with proper instrumentation (`-pg` or `-mfentry`)

4. **The filtering happens at compile time, not runtime**
   - Not about `EXPORT_SYMBOL()` macros
   - Not about symbol visibility
   - About whether ftrace hooks were compiled into the code

5. **Security features can hide addresses but not prevent tracing**
   - `kptr_restrict=1` hides addresses from unprivileged users
   - But with `sudo`, both addresses and tracing work (if instrumented)

## References

- Source: `src/nvidia/src/kernel/gpu/fifo/kernel_channel.c`
- Build: `src/nvidia/Makefile`, `kernel-open/nvidia/nvidia.Kbuild`
- Documentation: `docs/tsg_runlist_insertion.log`
- Kernel ftrace: Documentation/trace/ftrace.rst
- NVIDIA driver version: 575.57.08

## Timeline of Investigation

1. ❌ Checked for `EXPORT_SYMBOL()` - not relevant for tracing
2. ❌ Found `0000000000000000` addresses - just `kptr_restrict` hiding them
3. ✅ Discovered `available_filter_functions` - the real gatekeeper
4. ✅ Found massive discrepancy in traceable percentages (5% vs 44%)
5. ✅ Identified custom build system in `src/nvidia/Makefile`
6. ✅ **Root cause**: Custom Makefile doesn't add ftrace instrumentation

**Conclusion**: The "symbol export filtering" is actually **ftrace instrumentation filtering** that occurs at build time due to the custom Makefile not integrating with kernel's CONFIG_DYNAMIC_FTRACE system.
