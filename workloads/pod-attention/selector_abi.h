/* POD task-selector ABI: ordinary CUDA-owned global memory, no host callbacks. */
#ifndef POD_SELECTOR_ABI_H
#define POD_SELECTOR_ABI_H
typedef unsigned int pod_u32;
typedef unsigned long long pod_u64;

#define POD_ABI_VERSION 1u
#define POD_ENGINE_CUDA 1u
#define POD_ENGINE_BPF 2u
#define POD_WORK 1u
#define POD_EXHAUSTED 2u
#define POD_BAD_INPUT 3u
#define POD_UNSET 0xffffffffu

struct PodSelectorContext {
    pod_u64 counters; /* device address of nsmid + 2 atomic u32 counters */
    pod_u32 abi_version;
    pod_u32 nsmid;
    pod_u32 smid;
    pod_u32 prefill_slots;
    pod_u32 decode_slots;
    pod_u32 proportional;
    pod_u32 grid_ctas;
    pod_u32 out_op;
    pod_u32 out_cta;
    pod_u32 status;
    pod_u32 engine;
    pod_u32 ticket;
    pod_u32 first_op;
    pod_u32 first_claim;
    pod_u32 fallback_claim;
    pod_u32 reserved;
};

#if defined(__cplusplus)
static_assert(sizeof(PodSelectorContext) == 72, "CUDA/BPF context size mismatch");
#else
_Static_assert(sizeof(struct PodSelectorContext) == 72, "BPF context size mismatch");
#endif
#endif
