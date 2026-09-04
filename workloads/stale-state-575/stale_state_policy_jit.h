/* SPDX-License-Identifier: MIT */
#ifndef STALE_STATE_575_POLICY_JIT_H
#define STALE_STATE_575_POLICY_JIT_H

#include "stale_state_policy_model.h"

#include <stddef.h>

/*
 * One bounded context is the entire host-uBPF ABI. The snapshot and decision
 * timestamp are immutable inputs; the BPF program writes only decision and
 * status. No driver, CUDA, or host pointer crosses this boundary.
 */
struct stale_state_575_jit_context {
    struct stale_state_575_snapshot snapshot;
    uint64_t decision_mono_ns;
    struct stale_state_575_decision decision;
    uint32_t status;
    uint32_t reserved;
};

#ifdef __cplusplus
extern "C" {
#endif

void *stale_state_575_jit_open(const char *path, char *error, size_t capacity);
int stale_state_575_jit_choose(void *handle,
                               struct stale_state_575_jit_context *context,
                               size_t context_bytes);
uint64_t stale_state_575_jit_calls(void *handle);
uint64_t stale_state_575_jit_contract_errors(void *handle);
void stale_state_575_jit_close(void *handle);

#ifdef __cplusplus
}
#endif

#endif
