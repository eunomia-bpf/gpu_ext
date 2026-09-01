/* SPDX-License-Identifier: MIT */
/* Count UVM migration events, including speculative prefetch, for one process. */

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
#include <sys/syscall.h>
#include <unistd.h>

#define UVM_TOOLS_INIT_EVENT_TRACKER_V2 76
#define UVM_TOOLS_SET_NOTIFICATION_THRESHOLD 57
#define UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS 58
#define UVM_EVENT_TYPE_MIGRATION 2
#define UVM_EVENT_ENABLE_MIGRATION (UINT64_C(1) << UVM_EVENT_TYPE_MIGRATION)
#define UVM_MIGRATION_CAUSE_PREFETCH 3
#define QUEUE_ENTRIES 65536U
#define EVENT_ENTRY_V2_BYTES 72U
#define EVENT_TYPE_COUNT_ALL 64U
#define NV_OK 0U

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
    uint64_t migrations;
    uint64_t migrated_bytes;
    uint64_t prefetch_migrations;
    uint64_t prefetch_bytes;
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

static int duplicate_target_fd(long pid, long target_fd)
{
    int pidfd = syscall(SYS_pidfd_open, pid, 0);
    int result;

    if (pidfd < 0) {
        fprintf(stderr, "pidfd_open failed: %s\n", strerror(errno));
        return -1;
    }
    result = syscall(SYS_pidfd_getfd, pidfd, target_fd, 0);
    if (result < 0)
        fprintf(stderr, "pidfd_getfd failed: %s\n", strerror(errno));
    close(pidfd);
    return result;
}

static void drain_events(uint8_t *queue, struct event_control *control,
                         struct counters *result)
{
    const uint32_t mask = QUEUE_ENTRIES - 1;
    uint32_t get = control->get_ahead & mask;

    atomic_thread_fence(memory_order_acquire);
    while (get != (control->put_behind & mask)) {
        uint8_t *entry = queue + (size_t)get * EVENT_ENTRY_V2_BYTES;
        if (entry[0] == UVM_EVENT_TYPE_MIGRATION) {
            struct migration_v2 event = {};
            memcpy(&event, entry, sizeof(event));
            result->migrations++;
            result->migrated_bytes += event.migrated_bytes;
            if (event.migration_cause == UVM_MIGRATION_CAUSE_PREFETCH) {
                result->prefetch_migrations++;
                result->prefetch_bytes += event.migrated_bytes;
            }
        }
        get = (get + 1) & mask;
        control->get_ahead = get;
        atomic_thread_fence(memory_order_release);
        control->get_behind = get;
        atomic_thread_fence(memory_order_acquire);
    }
}

static void emit(const char *event, const struct counters *result,
                 const struct event_control *control)
{
    printf("{\"event\":\"%s\",\"pid\":%ld,\"migrations\":%llu,"
           "\"migrated_bytes\":%llu,\"prefetch_migrations\":%llu,"
           "\"prefetch_bytes\":%llu,\"dropped_migrations\":%llu}\n",
           event, (long)getpid(),
           (unsigned long long)result->migrations,
           (unsigned long long)result->migrated_bytes,
           (unsigned long long)result->prefetch_migrations,
           (unsigned long long)result->prefetch_bytes,
           (unsigned long long)control->dropped[UVM_EVENT_TYPE_MIGRATION]);
    fflush(stdout);
}

int main(int argc, char **argv)
{
    struct init_event_tracker_v2 init = {};
    struct set_notification_threshold threshold = {.notification_threshold = 1};
    struct event_queue_enable_events enable = {
        .event_type_flags = UVM_EVENT_ENABLE_MIGRATION,
    };
    struct event_control *control = NULL;
    struct counters result = {};
    uint8_t *queue = NULL;
    char *pid_end = NULL;
    char *fd_end = NULL;
    long target_pid;
    long target_fd;
    int duplicated_fd = -1;
    int tools_fd = -1;
    int err = 1;

    if (argc != 5 || strcmp(argv[1], "--pid") != 0 ||
        strcmp(argv[3], "--target-fd") != 0) {
        fprintf(stderr, "usage: %s --pid PID --target-fd FD\n", argv[0]);
        return 2;
    }
    errno = 0;
    target_pid = strtol(argv[2], &pid_end, 10);
    target_fd = strtol(argv[4], &fd_end, 10);
    if (errno || !pid_end || *pid_end || !fd_end || *fd_end ||
        target_pid <= 0 || target_fd < 0 || target_fd > INT32_MAX) {
        fprintf(stderr, "invalid pid or target fd\n");
        return 2;
    }

    duplicated_fd = duplicate_target_fd(target_pid, target_fd);
    if (duplicated_fd < 0)
        goto out;
    if (posix_memalign((void **)&queue, 4096,
                       (size_t)QUEUE_ENTRIES * EVENT_ENTRY_V2_BYTES) ||
        posix_memalign((void **)&control, 4096, 4096)) {
        fprintf(stderr, "failed to allocate aligned event buffers\n");
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
    init.uvm_fd = (uint32_t)duplicated_fd;
    err = checked_ioctl(tools_fd, UVM_TOOLS_INIT_EVENT_TRACKER_V2, &init,
                        &init.rm_status, "init event tracker v2");
    if (err)
        goto out;
    err = checked_ioctl(tools_fd, UVM_TOOLS_SET_NOTIFICATION_THRESHOLD,
                        &threshold, &threshold.rm_status,
                        "set notification threshold");
    if (err)
        goto out;
    err = checked_ioctl(tools_fd, UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS, &enable,
                        &enable.rm_status, "enable migration events");
    if (err)
        goto out;

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("{\"event\":\"ready\",\"pid\":%ld,\"target_pid\":%ld,"
           "\"target_fd\":%ld,\"queue_entries\":%u}\n",
           (long)getpid(), target_pid, target_fd, QUEUE_ENTRIES);

    while (!exiting) {
        struct pollfd pfd = {.fd = tools_fd, .events = POLLIN};
        int poll_result = poll(&pfd, 1, 1000);
        if (poll_result < 0 && errno != EINTR) {
            fprintf(stderr, "event poll failed: %s\n", strerror(errno));
            err = -errno;
            goto out;
        }
        drain_events(queue, control, &result);
        emit("migration_stats", &result, control);
    }
    drain_events(queue, control, &result);
    emit("final_migration_stats", &result, control);
    err = 0;

out:
    if (tools_fd >= 0)
        close(tools_fd);
    if (duplicated_fd >= 0)
        close(duplicated_fd);
    free(control);
    free(queue);
    return err < 0 ? -err : err;
}
