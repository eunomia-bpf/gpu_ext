/* Integer-only, single-(device, MoE-layer) Expert Buffering snapshot. */
#ifndef EXPERT_BUFFERING_SECTION_VI_H
#define EXPERT_BUFFERING_SECTION_VI_H

typedef unsigned int eb_u32;
typedef unsigned long long eb_u64;
#define EB_ABI_VERSION 1u
#define EB_MAX_EXPERTS 60u
#define EB_NO_VICTIM 0xffffffffu
#define EB_RESIDENT 1u
#define EB_ELIGIBLE 2u

enum eb_status { EB_HIT, EB_ADMIT, EB_EVICT, EB_INVALID, EB_BLOCKED };

struct eb_entry {
    eb_u32 token_count; /* Actual current-batch routing, not predicted heat. */
    eb_u32 flags;
    eb_u64 admission; /* Successful insertion order; hits do not refresh it. */
};

struct eb_input {
    eb_u32 abi_version;
    eb_u32 count;
    eb_u32 capacity;
    eb_u32 incoming;
    eb_u32 layer_id;
    eb_u32 device_id;
    eb_u64 batch_epoch;
    struct eb_entry experts[EB_MAX_EXPERTS]; /* Index is layer-local expert ID. */
};

struct eb_output {
    eb_u64 batch_epoch;
    eb_u32 status;
    eb_u32 victim;
};

struct eb_context {
    struct eb_input input;
    struct eb_output output;
};

#ifdef __cplusplus
static_assert(sizeof(eb_entry) == 16 && sizeof(eb_context) == 1008, "EB ABI");
extern "C" {
#else
_Static_assert(sizeof(struct eb_entry) == 16 && sizeof(struct eb_context) == 1008,
               "EB ABI");
#endif
eb_u64 eb_select(struct eb_context *ctx);
#ifdef __cplusplus
}
#endif
#endif
