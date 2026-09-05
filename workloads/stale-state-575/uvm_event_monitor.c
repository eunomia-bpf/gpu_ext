/* SPDX-License-Identifier: MIT */
/*
 * Process-scoped UVM Tools event monitor for the stale-state experiment.
 * Counts actual driver events; it does not estimate faults from policy calls.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define UVM_TOOLS_INIT_EVENT_TRACKER_V2 76
#define UVM_TOOLS_SET_NOTIFICATION_THRESHOLD 57
#define UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS 58

#define UVM_EVENT_TYPE_MIGRATION 2U
#define UVM_EVENT_TYPE_GPU_FAULT 3U
#define UVM_EVENT_TYPE_FAULT_BUFFER_OVERFLOW 5U
#define UVM_EVENT_TYPE_THRASHING_DETECTED 10U
#define UVM_EVENT_TYPE_EVICTION 14U
#define UVM_EVENT_ENABLE(type) (UINT64_C(1) << (type))
#define UVM_MIGRATION_CAUSE_PREFETCH 3U

#define QUEUE_ENTRIES 65536U
#define EVENT_ENTRY_V2_BYTES 72U
#define EVENT_TYPE_COUNT_ALL 64U
#define NV_OK 0U
#define NV_ERR_ILLEGAL_ACTION 0x00000016U
#define UVM_CANDIDATES 2U

struct init_event_tracker_v2 {
    uint64_t queue_buffer;
    uint64_t queue_buffer_size;
    uint64_t control_buffer;
    uint8_t processor_uuid[16];
    uint32_t all_processors;
    uint32_t uvm_fd;
    uint32_t rm_status;
};

struct set_notification_threshold {
    uint32_t notification_threshold;
    uint32_t rm_status;
};

struct event_queue_enable_events {
    uint64_t event_type_flags;
    uint32_t rm_status;
    uint32_t padding;
};

struct event_control {
    volatile uint32_t get_ahead;
    volatile uint32_t get_behind;
    volatile uint32_t put_ahead;
    volatile uint32_t put_behind;
    uint64_t dropped[EVENT_TYPE_COUNT_ALL];
};

/* NVIDIA 575 UvmEventMigrationInfo_V2, kept local to avoid private headers. */
struct migration_v2 {
    uint8_t event_type;
    uint8_t migration_cause;
    uint16_t padding16;
    uint16_t src_index;
    uint16_t dst_index;
    int32_t src_nid;
    int32_t dst_nid;
    uint64_t address;
    uint64_t migrated_bytes;
    uint64_t begin_timestamp;
    uint64_t end_timestamp;
    uint64_t range_group_id;
    uint64_t begin_timestamp_gpu;
    uint64_t end_timestamp_gpu;
};

struct counters {
    uint64_t gpu_faults;
    uint64_t migrations;
    uint64_t migrated_bytes;
    uint64_t prefetch_migrations;
    uint64_t prefetch_bytes;
    uint64_t thrashing_events;
    uint64_t eviction_events;
    uint64_t fault_buffer_overflows;
};

_Static_assert(sizeof(struct init_event_tracker_v2) == 56,
               "UVM init-tracker ABI drift");
_Static_assert(sizeof(struct event_queue_enable_events) == 16,
               "UVM event-enable ABI drift");
_Static_assert(sizeof(struct event_control) == 528,
               "UVM event-control ABI drift");
_Static_assert(sizeof(struct migration_v2) == EVENT_ENTRY_V2_BYTES,
               "UVM migration-event ABI drift");

static volatile sig_atomic_t exiting;

struct uvm_candidate {
    long source_fd;
    long inherited_fd;
    uint32_t probe_status;
};

static void handle_signal(int signo)
{
    (void)signo;
    exiting = 1;
}

static int checked_ioctl(int fd, unsigned long command, void *params,
                         uint32_t *rm_status, const char *name)
{
    if (ioctl(fd, command, params) < 0) {
        fprintf(stderr, "%s ioctl failed: %s\n", name, strerror(errno));
        return -errno;
    }
    if (*rm_status != NV_OK) {
        fprintf(stderr, "%s returned NV_STATUS %u\n", name, *rm_status);
        return -EIO;
    }
    return 0;
}

static int parse_candidate(const char *text, struct uvm_candidate *candidate)
{
    char *separator = NULL;
    char *end = NULL;

    errno = 0;
    candidate->source_fd = strtol(text, &separator, 10);
    if (errno || separator == text || *separator != ':' ||
        candidate->source_fd < 0 || candidate->source_fd > INT32_MAX)
        return -EINVAL;
    errno = 0;
    candidate->inherited_fd = strtol(separator + 1, &end, 10);
    if (errno || end == separator + 1 || *end != '\0' ||
        candidate->inherited_fd < 0 || candidate->inherited_fd > INT32_MAX ||
        fcntl((int)candidate->inherited_fd, F_GETFD) < 0)
        return -EINVAL;
    return 0;
}

static int probe_candidate(struct uvm_candidate *candidate,
                           uint8_t *queue,
                           struct event_control *control)
{
    struct init_event_tracker_v2 init = {
        .queue_buffer = (uintptr_t)queue,
        .queue_buffer_size = QUEUE_ENTRIES,
        .control_buffer = (uintptr_t)control,
        .all_processors = 1,
        .uvm_fd = (uint32_t)candidate->inherited_fd,
    };
    int tools_fd = open("/dev/nvidia-uvm-tools", O_RDWR | O_CLOEXEC);
    int result;

    if (tools_fd < 0) {
        fprintf(stderr, "failed to open /dev/nvidia-uvm-tools: %s\n",
                strerror(errno));
        return -errno;
    }
    result = ioctl(tools_fd, UVM_TOOLS_INIT_EVENT_TRACKER_V2, &init);
    if (result < 0) {
        result = -errno;
        fprintf(stderr, "candidate %ld tracker probe failed: %s\n",
                candidate->source_fd, strerror(errno));
        close(tools_fd);
        return result;
    }
    candidate->probe_status = init.rm_status;
    close(tools_fd);
    return 0;
}

static void drain_events(uint8_t *queue, struct event_control *control,
                         struct counters *result)
{
    const uint32_t mask = QUEUE_ENTRIES - 1U;
    uint32_t get = control->get_ahead & mask;

    atomic_thread_fence(memory_order_acquire);
    while (get != (control->put_behind & mask)) {
        uint8_t *entry = queue + (size_t)get * EVENT_ENTRY_V2_BYTES;
        const uint8_t event_type = entry[0];

        if (event_type == UVM_EVENT_TYPE_GPU_FAULT) {
            result->gpu_faults++;
        }
        else if (event_type == UVM_EVENT_TYPE_MIGRATION) {
            struct migration_v2 event = {0};
            memcpy(&event, entry, sizeof(event));
            result->migrations++;
            result->migrated_bytes += event.migrated_bytes;
            if (event.migration_cause == UVM_MIGRATION_CAUSE_PREFETCH) {
                result->prefetch_migrations++;
                result->prefetch_bytes += event.migrated_bytes;
            }
        }
        else if (event_type == UVM_EVENT_TYPE_THRASHING_DETECTED) {
            result->thrashing_events++;
        }
        else if (event_type == UVM_EVENT_TYPE_EVICTION) {
            result->eviction_events++;
        }
        else if (event_type == UVM_EVENT_TYPE_FAULT_BUFFER_OVERFLOW) {
            result->fault_buffer_overflows++;
        }

        get = (get + 1U) & mask;
        control->get_ahead = get;
        atomic_thread_fence(memory_order_release);
        control->get_behind = get;
        atomic_thread_fence(memory_order_acquire);
    }
}

static void emit(const char *event, const struct counters *result,
                 const struct event_control *control)
{
    printf("{\"event\":\"%s\",\"pid\":%ld,"
           "\"gpu_faults\":%llu,\"migrations\":%llu,"
           "\"migrated_bytes\":%llu,\"prefetch_migrations\":%llu,"
           "\"prefetch_bytes\":%llu,\"thrashing_events\":%llu,"
           "\"eviction_events\":%llu,\"fault_buffer_overflows\":%llu,"
           "\"dropped_gpu_faults\":%llu,\"dropped_migrations\":%llu,"
           "\"dropped_thrashing\":%llu,\"dropped_evictions\":%llu}\n",
           event, (long)getpid(),
           (unsigned long long)result->gpu_faults,
           (unsigned long long)result->migrations,
           (unsigned long long)result->migrated_bytes,
           (unsigned long long)result->prefetch_migrations,
           (unsigned long long)result->prefetch_bytes,
           (unsigned long long)result->thrashing_events,
           (unsigned long long)result->eviction_events,
           (unsigned long long)result->fault_buffer_overflows,
           (unsigned long long)control->dropped[UVM_EVENT_TYPE_GPU_FAULT],
           (unsigned long long)control->dropped[UVM_EVENT_TYPE_MIGRATION],
           (unsigned long long)control->dropped[UVM_EVENT_TYPE_THRASHING_DETECTED],
           (unsigned long long)control->dropped[UVM_EVENT_TYPE_EVICTION]);
    fflush(stdout);
}

int main(int argc, char **argv)
{
    struct init_event_tracker_v2 init = {0};
    struct set_notification_threshold threshold = {
        .notification_threshold = 1,
    };
    struct event_queue_enable_events enable = {
        .event_type_flags = UVM_EVENT_ENABLE(UVM_EVENT_TYPE_MIGRATION) |
                            UVM_EVENT_ENABLE(UVM_EVENT_TYPE_GPU_FAULT) |
                            UVM_EVENT_ENABLE(UVM_EVENT_TYPE_FAULT_BUFFER_OVERFLOW) |
                            UVM_EVENT_ENABLE(UVM_EVENT_TYPE_THRASHING_DETECTED) |
                            UVM_EVENT_ENABLE(UVM_EVENT_TYPE_EVICTION),
    };
    struct event_control *control = NULL;
    struct counters result = {0};
    struct uvm_candidate candidates[UVM_CANDIDATES] = {0};
    uint8_t *queue = NULL;
    char *pid_end = NULL;
    long target_pid;
    unsigned int selected = UVM_CANDIDATES;
    unsigned int rejected = UVM_CANDIDATES;
    unsigned int index;
    int tools_fd = -1;
    int err = 1;

    if (argc != 7 || strcmp(argv[1], "--uvm-candidate") != 0 ||
        strcmp(argv[3], "--uvm-candidate") != 0 ||
        strcmp(argv[5], "--target-pid") != 0) {
        fprintf(stderr, "usage: %s --uvm-candidate SOURCE:INHERITED "
                "--uvm-candidate SOURCE:INHERITED --target-pid PID\n", argv[0]);
        return 2;
    }
    if (parse_candidate(argv[2], &candidates[0]) != 0 ||
        parse_candidate(argv[4], &candidates[1]) != 0 ||
        candidates[0].source_fd == candidates[1].source_fd ||
        candidates[0].inherited_fd == candidates[1].inherited_fd) {
        fprintf(stderr, "invalid or duplicate inherited UVM candidates\n");
        return 2;
    }
    errno = 0;
    target_pid = strtol(argv[6], &pid_end, 10);
    if (errno || pid_end == NULL || *pid_end != '\0' || target_pid <= 0) {
        fprintf(stderr, "invalid target pid\n");
        return 2;
    }

    if (posix_memalign((void **)&queue, 4096,
                       (size_t)QUEUE_ENTRIES * EVENT_ENTRY_V2_BYTES) != 0 ||
        posix_memalign((void **)&control, 4096, 4096) != 0) {
        fprintf(stderr, "failed to allocate aligned event buffers\n");
        goto out;
    }
    memset(queue, 0, (size_t)QUEUE_ENTRIES * EVENT_ENTRY_V2_BYTES);
    memset(control, 0, 4096);

    for (index = 0; index < UVM_CANDIDATES; ++index) {
        memset(queue, 0, (size_t)QUEUE_ENTRIES * EVENT_ENTRY_V2_BYTES);
        memset(control, 0, 4096);
        err = probe_candidate(&candidates[index], queue, control);
        if (err != 0)
            goto out;
        if (candidates[index].probe_status == NV_OK) {
            if (selected != UVM_CANDIDATES) {
                fprintf(stderr, "multiple UVM candidates are VA-space FDs\n");
                err = -EPROTO;
                goto out;
            }
            selected = index;
        }
        else if (candidates[index].probe_status == NV_ERR_ILLEGAL_ACTION) {
            if (rejected != UVM_CANDIDATES) {
                fprintf(stderr, "multiple UVM candidates are non-VA-space FDs\n");
                err = -EPROTO;
                goto out;
            }
            rejected = index;
        }
        else {
            fprintf(stderr, "candidate %ld returned unexpected NV_STATUS %u\n",
                    candidates[index].source_fd, candidates[index].probe_status);
            err = -EPROTO;
            goto out;
        }
    }
    if (selected == UVM_CANDIDATES || rejected == UVM_CANDIDATES) {
        fprintf(stderr, "UVM candidates did not resolve to one VA-space and one MM FD\n");
        err = -EPROTO;
        goto out;
    }

    memset(queue, 0, (size_t)QUEUE_ENTRIES * EVENT_ENTRY_V2_BYTES);
    memset(control, 0, 4096);
    tools_fd = open("/dev/nvidia-uvm-tools", O_RDWR | O_CLOEXEC);
    if (tools_fd < 0) {
        fprintf(stderr, "failed to open /dev/nvidia-uvm-tools: %s\n",
                strerror(errno));
        goto out;
    }

    init.queue_buffer = (uintptr_t)queue;
    init.queue_buffer_size = QUEUE_ENTRIES;
    init.control_buffer = (uintptr_t)control;
    init.all_processors = 1;
    init.uvm_fd = (uint32_t)candidates[selected].inherited_fd;
    err = checked_ioctl(tools_fd, UVM_TOOLS_INIT_EVENT_TRACKER_V2, &init,
                        &init.rm_status, "init event tracker v2");
    if (err != 0)
        goto out;
    err = checked_ioctl(tools_fd, UVM_TOOLS_SET_NOTIFICATION_THRESHOLD,
                        &threshold, &threshold.rm_status,
                        "set notification threshold");
    if (err != 0)
        goto out;
    err = checked_ioctl(tools_fd, UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS, &enable,
                        &enable.rm_status, "enable UVM events");
    if (err != 0)
        goto out;

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("{\"event\":\"ready\",\"pid\":%ld,\"target_pid\":%ld,"
           "\"uvm_fd\":%ld,\"candidate_source_fds\":[%ld,%ld],"
           "\"candidate_targets\":[\"/dev/nvidia-uvm\",\"/dev/nvidia-uvm\"],"
           "\"selected_source_fd\":%ld,\"rejected_source_fd\":%ld,"
           "\"rejected_status\":%u,\"queue_entries\":%u,\"entry_bytes\":%u}\n",
           (long)getpid(), target_pid, candidates[selected].inherited_fd,
           candidates[0].source_fd, candidates[1].source_fd,
           candidates[selected].source_fd, candidates[rejected].source_fd,
           candidates[rejected].probe_status, QUEUE_ENTRIES, EVENT_ENTRY_V2_BYTES);

    while (!exiting) {
        struct pollfd pfd = {.fd = tools_fd, .events = POLLIN};
        int poll_result = poll(&pfd, 1, 1000);
        if (poll_result < 0 && errno != EINTR) {
            fprintf(stderr, "event poll failed: %s\n", strerror(errno));
            err = -errno;
            goto out;
        }
        drain_events(queue, control, &result);
        emit("uvm_stats", &result, control);
    }
    drain_events(queue, control, &result);
    emit("final_uvm_stats", &result, control);
    err = 0;

out:
    if (tools_fd >= 0)
        close(tools_fd);
    for (index = 0; index < UVM_CANDIDATES; ++index)
        close((int)candidates[index].inherited_fd);
    free(control);
    free(queue);
    return err < 0 ? -err : err;
}
