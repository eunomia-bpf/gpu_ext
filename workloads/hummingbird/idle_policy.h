/* SPDX-License-Identifier: GPL-2.0 */
#ifndef GPUBPF_HUMMINGBIRD_IDLE_POLICY_H
#define GPUBPF_HUMMINGBIRD_IDLE_POLICY_H

typedef unsigned long long hb_u64;
typedef unsigned int hb_u32;

enum hb_action { HB_ERROR = 0, HB_STOP_LP = 1, HB_WAIT = 2, HB_SPLIT = 3, HB_WHOLE = 4 };
enum hb_bubble { HB_NO_BUBBLE = 0, HB_SMALL_BUBBLE = 1, HB_LARGE_BUBBLE = 2 };
enum hb_wait { HB_NOT_WAITING = 0, HB_WAIT_EMPTY = 1, HB_WAIT_HP = 2,
               HB_WAIT_BUBBLE = 3, HB_WAIT_TICK = 4, HB_WAIT_LP_EVENT = 5 };

/* Adapter observations only: no CUDA pointers or kernel-control handles.
 * All timestamps use one monotonic host clock. last_hp_activity_ns advances
 * on real HP enqueue/completion observations, never on a guessed request rate.
 * hp_gpu_done/lp_gpu_done require successful CUDA event completion queries.
 */
struct hb_input {
    hb_u64 now_ns;
    hb_u64 last_hp_activity_ns;
    hb_u64 large_after_ns;
    hb_u64 tick_due_ns;
    hb_u64 launch_overhead_ns;
    hb_u64 split_ns;
    hb_u64 whole_ns;
    hb_u32 hp_pending;
    hb_u32 hp_gpu_done;
    hb_u32 small_active;
    hb_u32 small_start_done;
    hb_u32 lp_pending;
    hb_u32 lp_gpu_done;
    hb_u32 kernel_unstarted;
    hb_u32 consolidate;
};

struct hb_output {
    hb_u64 next_tick_ns;
    hb_u32 action;
    hb_u32 bubble;
    hb_u32 wait_reason;
    hb_u32 reserved;
};

struct hb_call { struct hb_input input; struct hb_output output; };

#ifdef __cplusplus
extern "C" {
#endif
hb_u64 hb_decide(struct hb_call *call, unsigned long length);
#ifdef __cplusplus
}
#endif
#endif
