#include "finemoe_policy.h"
_Static_assert(sizeof(struct fm_input) == 256, "input ABI");
_Static_assert(sizeof(struct fm_output) == 24, "output ABI");
_Static_assert(sizeof(struct fm_context) == 280, "context ABI");
int finemoe_select_native(struct fm_context *ctx)
{
    return ctx ? (int)fm_select(ctx) : -1;
}
