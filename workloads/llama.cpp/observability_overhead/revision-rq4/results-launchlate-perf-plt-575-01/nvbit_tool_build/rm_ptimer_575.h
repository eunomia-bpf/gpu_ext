#pragma once

#include <stdint.h>

#define RM_PTIMER_QUANTIZATION_NS 32ULL
#define RM_PTIMER_MAX_OUTER_NS 10000000ULL

struct rm_ptimer_575_client {
    int control_fd;
    int gpu_fd;
    uint32_t root;
    uint32_t device;
    uint32_t subdevice;
};

struct rm_ptimer_575_sample {
    uint64_t outer_before_raw_ns;
    uint64_t outer_after_raw_ns;
    uint64_t cpu_before_raw_ns;
    uint64_t gpu_ptimer_ns;
    uint64_t cpu_after_raw_ns;
    uint64_t outer_width_ns;
    uint64_t selected_gap_ns;
    int64_t offset_low_ns;
    int64_t offset_high_ns;
    uint64_t bracket_width_ns;
    uint32_t rm_status;
};

#ifdef __cplusplus
extern "C" {
#endif
void rm_ptimer_575_client_init(struct rm_ptimer_575_client* client);
int rm_ptimer_575_open(struct rm_ptimer_575_client* client);
int rm_ptimer_575_sample(struct rm_ptimer_575_client* client,
                         struct rm_ptimer_575_sample* sample);
int rm_ptimer_575_close(struct rm_ptimer_575_client* client);
int rm_ptimer_575_self_test(void);
#ifdef __cplusplus
}
#endif
