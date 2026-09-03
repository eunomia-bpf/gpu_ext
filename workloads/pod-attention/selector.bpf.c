#include "selector_policy.h"

/* bpftime's GPU entry returns void. The actual result is explicitly stored in
 * device-global context fields and consumed by the attention executor. */
__attribute__((section("pod_selector"), used))
int cuda__podsel(struct PodSelectorContext *ctx, pod_u64 len) {
    pod_select_policy(ctx, len, POD_ENGINE_BPF);
    return 0;
}

char LICENSE[] __attribute__((section("license"), used)) = "Dual BSD/GPL";
