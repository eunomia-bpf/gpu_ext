/* SPDX-License-Identifier: GPL-2.0
 *
 * XSched Level-2 guardian NVBit tool: entry control affecting real command
 * execution. On the first target launch the tool inserts a call to the
 * shared trusted trampoline xg_tramp before the target kernel's entry
 * instruction, with:
 *   arg0: guard predicate value (NVBit convention)
 *   arg1: uint64 launch value = per-command actuator context device address,
 *         armed per launch via nvbit_set_at_launch
 *   arg2: uint32 decision mode constant (1 = device eBPF, 0 = native C)
 *
 * The HAL (XSched libhalcuda in BPF-actuator mode) publishes each command's
 * context block address in-process on the launching thread right before
 * cuLaunchKernel via the exported xg_host_prepare symbol, passing the real
 * (CUcontext, CUfunction); xg_host_prepare stores the value in the
 * thread-local armed marker and still issues the outer
 * nvbit_set_at_launch as a fallback. The authoritative per-launch value is
 * set inside the NVBit pre-launch callback after instrumentation: the
 * value is associated with the instrumented function, so each launch
 * publishes its own slot instead of every later launch retaining the
 * first one. The HAL serializes all actuator launches on launch_mtx_, so
 * the armed value always belongs to the launch that follows it on the
 * same thread. The value registered at launch stays in place after the
 * launch call returns: CTAs may enter later. A target launch that did not
 * go through the XSched queue instead disarms the stale value in its own
 * pre-callback (one-shot), so it can never act on a stale context block;
 * the disarm is applied after instrumentation for the same ordering
 * reason.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "nvbit.h"
#include "nvbit_tool.h"

static std::string target_symbol;
static uint32_t decision_mode = 0; /* 0 = native, 1 = eBPF */
static uint32_t guarded_launches = 0;
static uint32_t instrumented_functions = 0;
static std::unordered_set<CUfunction> instrumented;
static std::mutex instrumented_mtx;
static thread_local uint64_t armed_val = 0; /* set by xg_host_prepare on the
                                             * launching thread, one-shot */
static std::atomic<uint32_t> xg3_set_cnt{0};
static std::atomic<uint32_t> xg3_cb_cnt{0};
static std::atomic<uint32_t> xg3_disarm_cnt{0};

static bool is_launch_cbid(nvbit_api_cuda_t cbid)
{
    return cbid == API_CUDA_cuLaunchKernel ||
           cbid == API_CUDA_cuLaunchKernel_ptsz ||
           cbid == API_CUDA_cuLaunchKernelEx ||
           cbid == API_CUDA_cuLaunchKernelEx_ptsz;
}

static CUfunction launch_func(nvbit_api_cuda_t cbid, const void *params)
{
    if (cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        return reinterpret_cast<const cuLaunchKernelEx_params *>(params)->f;
    }
    if (cbid == API_CUDA_cuLaunchKernel_ptsz) {
        return reinterpret_cast<const cuLaunchKernel_ptsz_params *>(params)->f;
    }
    return reinterpret_cast<const cuLaunchKernel_params *>(params)->f;
}

static CUstream launch_stream(nvbit_api_cuda_t cbid, const void *params)
{
    if (cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        return reinterpret_cast<const cuLaunchKernelEx_params *>(params)
            ->config->hStream;
    }
    if (cbid == API_CUDA_cuLaunchKernel_ptsz) {
        return reinterpret_cast<const cuLaunchKernel_ptsz_params *>(params)
            ->hStream;
    }
    return reinterpret_cast<const cuLaunchKernel_params *>(params)->hStream;
}

static void insert_entry_call(CUcontext ctx, CUfunction f)
{
    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
    if (instrs.empty()) {
        fprintf(stderr, "XG error: no instructions in target\n");
        exit(1);
    }
    Instr *entry = instrs[0];
    uint32_t entry_offset = entry->getOffset();
    for (Instr *in : instrs) {
        if (in->getOffset() < entry_offset) {
            entry = in;
            entry_offset = in->getOffset();
        }
    }
    const uint32_t entry_idx = entry->getIdx();
    nvbit_insert_call(entry, "xg_tramp", IPOINT_BEFORE);
    nvbit_add_call_arg_guard_pred_val(entry);
    nvbit_add_call_arg_launch_val64(entry, 0);
    nvbit_add_call_arg_const_val32(entry, decision_mode);
    nvbit_enable_instrumented(ctx, f, true, false);
    instrumented_functions++;
    fprintf(stderr,
            "XG instrumented_entry function_idx=%u entry_offset=%u decision_mode=%u\n",
            entry_idx, entry_offset, decision_mode);
}

extern "C" __attribute__((visibility("default")))
void xg_host_prepare(CUcontext ctx, CUfunction f, uint64_t ctx_dev)
{
    if (ctx_dev == 0) {
        /* unpublish: clear the one-shot marker only; the armed value must
         * stay in place until the next launch overwrites it */
        armed_val = 0;
        return;
    }
    armed_val = ctx_dev;
    const uint32_t n = xg3_set_cnt.fetch_add(1);
    if (n < 512) {
        fprintf(stderr, "XG3 set tid=%lu val=%p f=%p ctx=%p\n",
                (unsigned long)pthread_self(), (void *)ctx_dev,
                (void *)f, (void *)ctx);
    }
    /* fallback: also published outside the callback, so launches whose
     * pre-callback is not delivered still get the host value. The
     * authoritative set happens inside the callback after
     * instrumentation, which runs later on this same thread and wins. */
    nvbit_set_at_launch(ctx, f, ctx_dev);
}

void nvbit_at_init()
{
    const char *mode = getenv("XG_DECISION");
    if (mode != nullptr && strcmp(mode, "bpf") == 0) decision_mode = 1u;
    const char *sym = getenv("XG_TARGET_SYMBOL");
    if (sym != nullptr && sym[0] != '\0') target_symbol = sym;
    fprintf(stderr, "XG tool loaded decision=%s target=%s\n",
            decision_mode == 1u ? "bpf" : "native", target_symbol.c_str());
}

void nvbit_tool_init(CUcontext ctx)
{
    (void)ctx;
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *event_name, void *params,
                         CUresult *pStatus)
{
    (void)event_name;
    (void)pStatus;
    if (!is_launch_cbid(cbid)) return;
    CUfunction f = launch_func(cbid, params);
    if (f == nullptr || !nvbit_is_func_kernel(ctx, f)) return;
    if (is_exit) return;

    bool insert_entry = false;
    {
        std::lock_guard<std::mutex> lock(instrumented_mtx);
        if (instrumented.find(f) == instrumented.end()) {
            const char *mangled = nvbit_get_func_name(ctx, f, true);
            const char *demangled = nvbit_get_func_name(ctx, f, false);
            if (target_symbol != mangled && target_symbol != demangled) return;
            insert_entry = true;
        }
    }

    /* one-shot on this thread: xg_host_prepare armed the value right before
     * this launch's cuLaunchKernel. A callback without a pending armed
     * value means this target launch bypassed the XSched queue: the
     * trailing nvbit_set_at_launch below disarms the stale value so its
     * CTAs cannot act on a stale context block */
    uint64_t armed = armed_val;
    if (armed == 0) {
        const uint32_t n = xg3_disarm_cnt.fetch_add(1);
        if (n < 512) {
            fprintf(stderr, "XG3 disarm tid=%lu f=%p\n",
                    (unsigned long)pthread_self(), (void *)f);
        }
    } else {
        armed_val = 0; /* consume the one-shot TLS marker */
    }

    {
        const uint32_t n = xg3_cb_cnt.fetch_add(1);
        if (n < 512) {
            fprintf(stderr, "XG3 cb tid=%lu f=%p\n",
                    (unsigned long)pthread_self(), (void *)f);
        }
    }

    {
        std::lock_guard<std::mutex> lock(instrumented_mtx);
        guarded_launches++;
        if (insert_entry && instrumented.insert(f).second)
            insert_entry_call(ctx, f);
    }

    /* register the per-launch value after instrumentation, matching the
     * official NVBit example (mem_trace enter_kernel_launch): the value is
     * associated with the instrumented function, so each launch publishes
     * the slot its host prepared instead of an early one. armed == 0
     * applies the one-shot disarm to this launch. */
    nvbit_set_at_launch(ctx, f, armed);
    /* Match the example's per-launch ordering, not just first insertion. */
    nvbit_enable_instrumented(ctx, f, true, false);
}

void nvbit_at_term()
{
    fprintf(stderr, "XG done functions=%u launches=%u sets=%u cbs=%u disarms=%u\n",
            instrumented_functions, guarded_launches,
            xg3_set_cnt.load(), xg3_cb_cnt.load(), xg3_disarm_cnt.load());
}
