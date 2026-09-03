/* The original POD proportional-SM-ticket rule, compiled for CUDA and eBPF.
 * Source: microsoft/vattention, pod_attn/fused_fwd_kernel.h:1444-1498,
 * revision 71a0e91aa46ff8fa985bcca3327efe0ab9929a39. No score/ratio retuning.
 */
#ifndef POD_SELECTOR_POLICY_H
#define POD_SELECTOR_POLICY_H
#include "selector_abi.h"
#ifndef POD_POLICY_INLINE
#define POD_POLICY_INLINE static __inline__ __attribute__((always_inline))
#endif
#ifndef POD_FETCH_ADD
#define POD_FETCH_ADD(ptr) __sync_fetch_and_add((ptr), 1u)
#endif

POD_POLICY_INLINE void pod_select_policy(struct PodSelectorContext *c, pod_u64 len,
                                         pod_u32 engine) {
    /* The executor owns and validates the pointer itself; this is not a claim
     * that a kernel eBPF verifier permits arbitrary nested pointers. */
    if (len != sizeof(*c)) return;
    c->status = POD_BAD_INPUT;
    c->engine = engine;
    c->out_op = c->out_cta = POD_UNSET;
    c->ticket = c->first_op = c->first_claim = c->fallback_claim = POD_UNSET;
    if (c->abi_version != POD_ABI_VERSION || !c->counters ||
        !c->nsmid || c->smid >= c->nsmid || !c->prefill_slots ||
        !c->decode_slots || c->proportional > 1u || !c->grid_ctas ||
        c->prefill_slots > 0x3fffffffu || c->decode_slots > 0x3fffffffu ||
        c->grid_ctas > 0x3fffffffu) return;
    pod_u32 *counters = (pod_u32 *)c->counters;
    pod_u32 ticket = POD_FETCH_ADD(&counters[c->smid]);
    pod_u32 op;
    if (c->proportional) {
        if (c->prefill_slots <= c->decode_slots) {
            pod_u32 tags = c->decode_slots / c->prefill_slots + 1u;
            op = ticket % tags > 0u ? 1u : 0u;
        } else {
            pod_u32 tags = c->prefill_slots / c->decode_slots;
            op = ticket % (tags + 1u) < tags ? 0u : 1u;
        }
    } else {
        op = ticket % 2u;
    }
    c->ticket = ticket;
    c->first_op = op;
    pod_u32 slot = POD_FETCH_ADD(&counters[c->nsmid + op]);
    c->first_claim = slot;
    if ((op == 0u && slot >= c->prefill_slots) ||
        (op == 1u && slot >= c->decode_slots)) {
        op = 1u - op;
        slot = POD_FETCH_ADD(&counters[c->nsmid + op]);
        c->fallback_claim = slot;
    }
    c->out_op = op;
    c->out_cta = slot;
    c->status = slot < (op == 0u ? c->prefill_slots : c->decode_slots)
                    ? POD_WORK : POD_EXHAUSTED;
}
#endif
