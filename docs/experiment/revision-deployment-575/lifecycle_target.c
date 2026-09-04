#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <link.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t keep_running = 1;

static void stop_target(int signal_number)
{
    (void)signal_number;
    keep_running = 0;
}

static uint64_t realtime_ns(void)
{
    struct timespec value;

    if (clock_gettime(CLOCK_REALTIME, &value) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
           (uint64_t)value.tv_nsec;
}

struct loaded_state {
    bool agent;
    bool syscall_server;
};

static int inspect_loaded_object(struct dl_phdr_info *info, size_t size,
                                 void *opaque)
{
    struct loaded_state *state = opaque;

    (void)size;
    if (info->dlpi_name == NULL)
        return 0;
    if (strstr(info->dlpi_name, "libbpftime-agent.so") != NULL)
        state->agent = true;
    if (strstr(info->dlpi_name, "libbpftime-syscall-server.so") != NULL)
        state->syscall_server = true;
    return 0;
}

static struct loaded_state loaded_state(void)
{
    struct loaded_state state = { false, false };

    dl_iterate_phdr(inspect_loaded_object, &state);
    return state;
}

static void sleep_one_millisecond(void)
{
    const struct timespec duration = { .tv_sec = 0, .tv_nsec = 1000000 };

    while (nanosleep(&duration, NULL) != 0 && errno == EINTR) {
    }
}

static int run_server(void)
{
    int fd = open("/dev/null", O_RDONLY);
    struct loaded_state state;

    if (fd < 0) {
        perror("open");
        return 2;
    }
    close(fd);
    state = loaded_state();
    printf("SERVER_READY realtime_ns=%llu syscall_server_loaded=%d\n",
           (unsigned long long)realtime_ns(), state.syscall_server ? 1 : 0);
    fflush(stdout);
    while (keep_running)
        sleep_one_millisecond();
    return 0;
}

static int run_target(void)
{
    bool reported_agent_ready = false;
    struct loaded_state initial = loaded_state();

    if (prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0) != 0) {
        perror("prctl(PR_SET_PTRACER)");
        return 2;
    }
    printf("TARGET_READY pid=%ld realtime_ns=%llu agent_loaded=%d "
           "bpftime_used=%d\n",
           (long)getpid(), (unsigned long long)realtime_ns(),
           initial.agent ? 1 : 0, getenv("BPFTIME_USED") != NULL ? 1 : 0);
    fflush(stdout);

    while (keep_running) {
        const char *used = getenv("BPFTIME_USED");
        if (!reported_agent_ready && used != NULL) {
            struct loaded_state state = loaded_state();
            printf("AGENT_READY pid=%ld realtime_ns=%llu agent_named=%d "
                   "bpftime_used=1\n",
                   (long)getpid(), (unsigned long long)realtime_ns(),
                   state.agent ? 1 : 0);
            fflush(stdout);
            reported_agent_ready = true;
        }
        sleep_one_millisecond();
    }
    return reported_agent_ready ? 0 : 3;
}

int main(int argc, char **argv)
{
    struct sigaction action = { 0 };

    action.sa_handler = stop_target;
    sigemptyset(&action.sa_mask);
    sigaction(SIGTERM, &action, NULL);
    sigaction(SIGINT, &action, NULL);

    if (argc == 2 && strcmp(argv[1], "--server") == 0)
        return run_server();
    if (argc == 1)
        return run_target();

    fprintf(stderr, "usage: %s [--server]\n", argv[0]);
    return 2;
}
