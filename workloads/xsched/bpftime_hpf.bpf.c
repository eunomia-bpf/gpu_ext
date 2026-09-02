/* SPDX-License-Identifier: GPL-2.0 */
#include "bpftime_hpf.h"

/* The exact HPF decision for a bounded snapshot: suspend a queue iff a
 * ready queue on the same device has strictly higher priority. Ties run.
 * No CUDA, XSched, or driver pointers are exposed to the BPF program. */
unsigned long long hpf(const struct bpftime_hpf_snapshot *snapshot,
                       unsigned long length)
{
    unsigned long long suspend = 0;
    if (length < sizeof(*snapshot) || snapshot->count > BPFTIME_HPF_MAX_QUEUES)
        return ~0ULL;
    for (unsigned int i = 0; i < snapshot->count; ++i) {
        for (unsigned int j = 0; j < snapshot->count; ++j) {
            if (snapshot->queues[j].ready
                && snapshot->queues[j].device == snapshot->queues[i].device
                && snapshot->queues[j].priority > snapshot->queues[i].priority) {
                suspend |= 1ULL << i;
                break;
            }
        }
    }
    return suspend;
}
