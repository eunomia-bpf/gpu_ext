#define FM_BPF 1
#include "finemoe_policy.h"
fm_u64 finemoe_select_bpf_program(struct fm_context *ctx)
{
    return fm_select(ctx);
}
