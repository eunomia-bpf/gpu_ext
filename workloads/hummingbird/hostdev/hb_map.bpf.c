// hb_map.bpf.c: eBPF device engine for the Hummingbird coordinate mapping.
//
// Compiled with clang -target bpf into section "cuda__/hb_device_map". The bounded
// hostdev exporter loads that section, runs it through the GPU eBPF verifier
// against the 48-byte HbMapContext, and emits a standalone device-callable
// PTX function `hb_device_bpf_map(context_ptr, context_length)` that the PTX
// patcher drops into the split-cubin translation unit in place of the native
// `hb_device_map` call.
//
// The body is the shared hb_map_policy with HB_ENGINE_BPF, so it produces
// byte-identical out_* to the native engine for identical in_*.

#include "hb_map_policy.h"

__attribute__((section("cuda__/hb_device_map"), used))
int cuda__hb_device_map(struct HbMapContext *ctx, hb_u64 len)
{
    hb_map_policy(ctx, len, HB_ENGINE_BPF);
    return 0;
}

char LICENSE[] __attribute__((section("license"), used)) = "Dual BSD/GPL";
