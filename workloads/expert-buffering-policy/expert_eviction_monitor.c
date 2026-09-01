/* SPDX-License-Identifier: MIT */
/* Count completed UVM evictions by an admitted expert-layout class table. */

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
#define UVM_EVENT_TYPE_EVICTION 14
#define UVM_EVENT_ENABLE_EVICTION (UINT64_C(1) << UVM_EVENT_TYPE_EVICTION)
#define QUEUE_ENTRIES 4096U
#define EVENT_ENTRY_V2_BYTES 72U
#define EVENT_TYPE_COUNT_ALL 64U
#define EXPERT_BLOCK_BYTES (2ULL * 1024ULL * 1024ULL)
#define EXPERT_MAX_LAYOUT_BLOCKS 65536U
#define NV_OK 0U

enum expert_block_class {
    EXPERT_BLOCK_DEFAULT = 0,
    EXPERT_BLOCK_COLD = 1,
    EXPERT_BLOCK_HOT = 2,
    EXPERT_BLOCK_SHARED = 3,
    EXPERT_BLOCK_CLASS_COUNT = 4,
};

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

struct eviction_v2 {
    uint8_t event_type;
    uint8_t padding8;
    uint16_t padding16;
    uint16_t src_index;
    uint16_t dst_index;
    uint64_t address_out;
    uint64_t address_in;
    uint64_t size;
    uint64_t timestamp;
};

struct layout {
    uint64_t base;
    uint32_t blocks;
    uint8_t *classes;
};

struct counters {
    uint64_t evictions;
    uint64_t evicted_bytes;
    uint64_t class_bytes[EXPERT_BLOCK_CLASS_COUNT];
};

_Static_assert(sizeof(struct init_event_tracker_v2) == 56,
               "UVM init-tracker ABI drift");
_Static_assert(sizeof(struct event_queue_enable_events) == 16,
               "UVM event-enable ABI drift");
_Static_assert(sizeof(struct event_control) == 528,
               "UVM event-control ABI drift");
_Static_assert(sizeof(struct eviction_v2) == 40,
               "UVM eviction-event ABI drift");

static volatile sig_atomic_t exiting;

static void handle_signal(int signo)
{
    (void)signo;
    exiting = 1;
}

static int read_class_table(const char *path, struct layout *layout)
{
    FILE *input;
    char line[256];
    unsigned long long base;
    unsigned long long hot_bytes;
    unsigned int blocks;
    unsigned int registrations;
    unsigned int line_number = 0;
    bool saw_header = false;

    input = fopen(path, "r");
    if (!input)
        return -errno;
    while (fgets(line, sizeof(line), input)) {
        unsigned int index;
        unsigned int class_value;
        char extra;

        line_number++;
        if (line[0] == '#' || line[0] == '\n')
            continue;
        if (!saw_header) {
            if (sscanf(line,
                       "base %llu blocks %u hot_bytes %llu registrations %u %c",
                       &base, &blocks, &hot_bytes, &registrations, &extra) != 4 ||
                blocks == 0 || blocks > EXPERT_MAX_LAYOUT_BLOCKS) {
                fprintf(stderr, "invalid class-table header at line %u\n",
                        line_number);
                fclose(input);
                return -EINVAL;
            }
            (void)hot_bytes;
            (void)registrations;
            layout->base = base;
            layout->blocks = blocks;
            layout->classes = calloc(blocks, sizeof(*layout->classes));
            if (!layout->classes) {
                fclose(input);
                return -ENOMEM;
            }
            saw_header = true;
            continue;
        }
        if (sscanf(line, "%u %u %c", &index, &class_value, &extra) != 2 ||
            index >= layout->blocks || class_value < EXPERT_BLOCK_COLD ||
            class_value > EXPERT_BLOCK_SHARED || layout->classes[index] != 0) {
            fprintf(stderr, "invalid class-table entry at line %u\n",
                    line_number);
            fclose(input);
            return -EINVAL;
        }
        layout->classes[index] = (uint8_t)class_value;
    }
    if (ferror(input)) {
        fclose(input);
        return -EIO;
    }
    fclose(input);
    return saw_header ? 0 : -EINVAL;
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

static uint8_t address_class(const struct layout *layout, uint64_t address)
{
    uint64_t offset;
    uint64_t index;

    if (address < layout->base)
        return EXPERT_BLOCK_DEFAULT;
    offset = address - layout->base;
    index = offset / EXPERT_BLOCK_BYTES;
    if (index >= layout->blocks)
        return EXPERT_BLOCK_DEFAULT;
    return layout->classes[index];
}

static uint64_t next_boundary(const struct layout *layout, uint64_t address)
{
    uint64_t end = layout->base + (uint64_t)layout->blocks * EXPERT_BLOCK_BYTES;

    if (address < layout->base)
        return layout->base;
    if (address >= end)
        return UINT64_MAX;
    return layout->base +
           ((address - layout->base) / EXPERT_BLOCK_BYTES + 1) *
               EXPERT_BLOCK_BYTES;
}

static void classify_bytes(const struct layout *layout, uint64_t address,
                           uint64_t size, struct counters *result)
{
    uint64_t cursor = address;
    uint64_t remaining = size;

    while (remaining) {
        uint64_t boundary = next_boundary(layout, cursor);
        uint64_t segment = remaining;
        uint8_t class_value = address_class(layout, cursor);

        if (boundary != UINT64_MAX && boundary > cursor &&
            boundary - cursor < segment)
            segment = boundary - cursor;
        result->class_bytes[class_value] += segment;
        if (UINT64_MAX - cursor < segment)
            break;
        cursor += segment;
        remaining -= segment;
    }
}

static void drain_events(uint8_t *queue, struct event_control *control,
                         const struct layout *layout, struct counters *result)
{
    const uint32_t mask = QUEUE_ENTRIES - 1;
    uint32_t get = control->get_ahead & mask;

    atomic_thread_fence(memory_order_acquire);
    while (get != (control->put_behind & mask)) {
        uint8_t *entry = queue + (size_t)get * EVENT_ENTRY_V2_BYTES;
        if (entry[0] == UVM_EVENT_TYPE_EVICTION) {
            struct eviction_v2 event = {};

            memcpy(&event, entry, sizeof(event));
            result->evictions++;
            result->evicted_bytes += event.size;
            classify_bytes(layout, event.address_out, event.size, result);
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
    printf("{\"event\":\"%s\",\"pid\":%ld,\"evictions\":%llu,"
           "\"evicted_bytes\":%llu,\"default_bytes\":%llu,"
           "\"cold_bytes\":%llu,\"hot_bytes\":%llu,"
           "\"shared_bytes\":%llu,\"dropped_evictions\":%llu}\n",
           event, (long)getpid(),
           (unsigned long long)result->evictions,
           (unsigned long long)result->evicted_bytes,
           (unsigned long long)result->class_bytes[EXPERT_BLOCK_DEFAULT],
           (unsigned long long)result->class_bytes[EXPERT_BLOCK_COLD],
           (unsigned long long)result->class_bytes[EXPERT_BLOCK_HOT],
           (unsigned long long)result->class_bytes[EXPERT_BLOCK_SHARED],
           (unsigned long long)control->dropped[UVM_EVENT_TYPE_EVICTION]);
    fflush(stdout);
}

int main(int argc, char **argv)
{
    struct init_event_tracker_v2 init = {};
    struct set_notification_threshold threshold = {.notification_threshold = 1};
    struct event_queue_enable_events enable = {
        .event_type_flags = UVM_EVENT_ENABLE_EVICTION,
    };
    struct event_control *control = NULL;
    struct counters result = {};
    struct layout layout = {};
    uint8_t *queue = NULL;
    char *end = NULL;
    long inherited_fd;
    int tools_fd = -1;
    int err;

    if (argc != 4 || strcmp(argv[1], "--uvm-fd") != 0) {
        fprintf(stderr,
                "usage: %s --uvm-fd INHERITED_FD CLASS_TABLE\n", argv[0]);
        return 2;
    }
    errno = 0;
    inherited_fd = strtol(argv[2], &end, 10);
    if (errno || !end || *end != '\0' || inherited_fd < 0 ||
        inherited_fd > INT32_MAX) {
        fprintf(stderr, "invalid inherited UVM fd: %s\n", argv[2]);
        return 2;
    }
    err = read_class_table(argv[3], &layout);
    if (err) {
        fprintf(stderr, "failed to read class table: %s\n", strerror(-err));
        goto out;
    }
    if (posix_memalign((void **)&queue, 4096,
                       (size_t)QUEUE_ENTRIES * EVENT_ENTRY_V2_BYTES) ||
        posix_memalign((void **)&control, 4096, 4096)) {
        err = -ENOMEM;
        goto out;
    }
    memset(queue, 0, (size_t)QUEUE_ENTRIES * EVENT_ENTRY_V2_BYTES);
    memset(control, 0, 4096);

    tools_fd = open("/dev/nvidia-uvm-tools", O_RDWR | O_CLOEXEC);
    if (tools_fd < 0) {
        err = -errno;
        goto out;
    }
    init.queue_buffer = (uintptr_t)queue;
    init.queue_buffer_size = QUEUE_ENTRIES;
    init.control_buffer = (uintptr_t)control;
    init.all_processors = 1;
    init.uvm_fd = (uint32_t)inherited_fd;
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
                        &enable.rm_status, "enable eviction events");
    if (err)
        goto out;

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    setvbuf(stdout, NULL, _IOLBF, 0);
    printf("{\"event\":\"ready\",\"pid\":%ld,\"uvm_fd\":%ld,"
           "\"layout_base\":%llu,\"layout_blocks\":%u}\n",
           (long)getpid(), inherited_fd,
           (unsigned long long)layout.base, layout.blocks);

    while (!exiting) {
        struct pollfd pfd = {.fd = tools_fd, .events = POLLIN};
        int poll_result = poll(&pfd, 1, 1000);

        if (poll_result < 0 && errno != EINTR) {
            err = -errno;
            goto out;
        }
        drain_events(queue, control, &layout, &result);
        emit("eviction_stats", &result, control);
    }
    drain_events(queue, control, &layout, &result);
    emit("final_eviction_stats", &result, control);
    err = 0;

out:
    if (err)
        fprintf(stderr, "expert eviction monitor failed: %s\n", strerror(-err));
    if (tools_fd >= 0)
        close(tools_fd);
    free(layout.classes);
    free(control);
    free(queue);
    return err < 0 ? -err : err;
}
