/* SPDX-License-Identifier: GPL-2.0 */
#ifndef BPFTIME_HPF_H
#define BPFTIME_HPF_H

#define BPFTIME_HPF_MAX_QUEUES 64
struct bpftime_hpf_queue {
    unsigned long long device;
    int priority;
    unsigned int ready;
};
struct bpftime_hpf_snapshot {
    unsigned int count;
    unsigned int reserved;
    struct bpftime_hpf_queue queues[BPFTIME_HPF_MAX_QUEUES];
};

#endif
