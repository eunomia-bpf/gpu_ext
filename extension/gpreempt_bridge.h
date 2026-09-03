/* SPDX-License-Identifier: GPL-2.0 */
#ifndef GPUBPF_GPREEMPT_BRIDGE_H
#define GPUBPF_GPREEMPT_BRIDGE_H

/* Fixed-width ABI shared by kernel BPF, host JIT, and the original clients. */
typedef unsigned int gp_u32;
typedef unsigned long long gp_u64;

enum gp_role { GP_LC = 0, GP_BE = 1 };
enum gp_event { GP_PREPROCESS = 1, GP_DUE = 2, GP_INFER = 3 };
enum gp_action { GP_RESET = 1, GP_HINT = 2, GP_BLOCK = 4, GP_RELEASE = 8 };

struct gp_hint_input {
    gp_u64 now_ns;       /* Both timestamps are original system_clock values. */
    gp_u64 deadline_ns;
    gp_u32 event;
    gp_u32 role;
    gp_u32 initialized;
    gp_u32 reserve;
};

struct gp_scope {
    gp_u32 role;
    gp_u32 gr_inits;
    gp_u32 registered;
    gp_u32 errors;
};

struct gp_handle_key { gp_u32 hclient, htsg; };
struct gp_record {
    gp_u64 pid_tgid;
    gp_u64 tsg_id;
    gp_u64 timeslice_us;
    gp_u64 cuda_context;
    gp_u32 role;
    gp_u32 engine;
    gp_u32 registered;
    gp_u32 reserved;
};

enum gp_stat {
    GP_SCOPE_ENTER, GP_SCOPE_LEAVE, GP_GR_INIT, GP_OTHER_ENGINE,
    GP_UNKNOWN_ENGINE, GP_TIMESLICE_OK, GP_SETTER_ERROR, GP_ALLOC_CAPTURED,
    GP_ALLOC_ERROR, GP_REGISTERED, GP_REGISTER_ERROR, GP_BIND_MATCH,
    GP_BIND_MISMATCH, GP_DESTROY, GP_MAP_ERROR, GP_SCOPE_ERROR,
    GP_STAT_COUNT
};

#ifndef GP_BPF_ONLY
#ifdef __cplusplus
extern "C" {
#endif
/* Default: original C policy. GPREEMPT_POLICY=bpf selects both actual BPF paths.
 * Every negative return is fatal to the caller; never silently fall back.
 * A begin/register/end scope encloses ONLY a newly-created CUDA context.
 * The BPF arm MUST NOT call the original C set_priority implementation. */
int gpreempt_bpf_enabled(void);
int gpreempt_ctx_begin(gp_u32 role);
int gpreempt_ctx_register(gp_u64 cuda_context, gp_u32 hclient,
                         gp_u32 htsg, gp_u32 role);
int gpreempt_ctx_end(void);
int gpreempt_hint_decide(gp_u32 event, gp_u32 role, gp_u64 now_ns,
                        gp_u64 deadline_ns, gp_u32 initialized, gp_u32 reserve);
#ifdef __cplusplus
}
#endif
#endif
#endif
