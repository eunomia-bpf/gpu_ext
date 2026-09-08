/* SPDX-License-Identifier: GPL-2.0
 *
 * XSched Level-2 guardian NVBit tool: entry control affecting real command
 * execution. On the first target launch the tool inserts a call to the
 * shared trusted trampoline xg_tramp before the target kernel's entry
 * instruction, with:
 *   arg0: guard predicate value (NVBit convention)
 *   arg1: uint64 launch value = per-command actuator context device address,
 *         captured per launch via nvbit_set_at_launch
 *   arg2: uint32 decision mode constant (1 = device eBPF, 0 = native C)
 *
 * The HAL (XSched libhalcuda in BPF-actuator mode) publishes each command's
 * context block address in-process on the launching thread right before
 * cuLaunchKernel via the exported xg_host_publish symbol; the callback
 * below reads it on the same thread, so there is one published slot per
 * in-flight launch record.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "nvbit.h"
#include "nvbit_tool.h"

static std::string target_symbol;
static uint32_t decision_mode = 0; /* 0 = native, 1 = eBPF */
static thread_local uint64_t published_ctx = 0; /* set by xg_host_publish */
static uint32_t guarded_launches = 0;
static uint32_t instrumented_functions = 0;
static std::unordered_set<CUfunction> instrumented;
static std::mutex instrument_mu;

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
void xg_host_publish(uint64_t ctx_dev)
{
    published_ctx = ctx_dev;
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

    std::lock_guard<std::mutex> lock(instrument_mu);
    if (instrumented.find(f) == instrumented.end()) {
        const char *mangled = nvbit_get_func_name(ctx, f, true);
        const char *demangled = nvbit_get_func_name(ctx, f, false);
        if (target_symbol != mangled && target_symbol != demangled) return;
    }

    nvbit_set_at_launch(ctx, f, published_ctx, launch_stream(cbid, params));
    const uint64_t consumed = published_ctx;
    /* one-shot consumption: a later launch that bypasses the XSched queue
     * must never act on a stale context block */
    published_ctx = 0;
    if (consumed != 0) guarded_launches++;
    if (instrumented.insert(f).second) insert_entry_call(ctx, f);
}

void nvbit_at_term()
{
    fprintf(stderr, "XG done functions=%u launches=%u\n",
            instrumented_functions, guarded_launches);
}
