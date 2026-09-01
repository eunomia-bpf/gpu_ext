/* SPDX-License-Identifier: GPL-2.0 */

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <bpf/libbpf.h>

struct fixture {
    const char *name;
    const char *object_name;
    bool expect_load;
    unsigned int denied_write_offset;
};

static const struct fixture fixtures[] = {
    { "sched-immutable-read", "revision_sched_immutable_read.bpf.o", true, 0 },
    { "sched-timeslice-setter", "revision_sched_timeslice_setter.bpf.o", true, 0 },
    { "sched-interleave-low-setter", "revision_sched_interleave_low_setter.bpf.o", true, 0 },
    { "pmm-reorder-setter", "revision_pmm_reorder_setter.bpf.o", true, 0 },
    { "sched-input-write", "revision_sched_input_write.bpf.o", false, 16 },
    { "sched-hidden-write", "revision_sched_hidden_write.bpf.o", false, 32 },
    { "pmm-hidden-write", "revision_pmm_hidden_write.bpf.o", false, 56 },
};

enum {
    POSITIVE_FIXTURE_COUNT = 4,
    VERIFIER_LOG_SIZE = 1024 * 1024,
};

static bool verbose;

static int libbpf_log(enum libbpf_print_level level,
                      const char *format,
                      va_list args)
{
    if (!verbose && level == LIBBPF_DEBUG)
        return 0;

    return vfprintf(stderr, format, args);
}

static void usage(const char *program)
{
    fprintf(stderr, "Usage: %s [-v] [-d OBJECT_DIR] [-l LOG_DIR]\n", program);
}

static int save_verifier_log(const char *log_dir,
                             const struct fixture *fixture,
                             int load_error,
                             const char *verifier_log)
{
    char path[512];
    FILE *output;

    if (snprintf(path, sizeof(path), "%s/%s.log", log_dir,
                 fixture->name) >= (int)sizeof(path)) {
        fprintf(stderr, "verifier log path too long for %s\n", fixture->name);
        return -ENAMETOOLONG;
    }

    output = fopen(path, "w");
    if (!output) {
        int error = -errno;

        fprintf(stderr, "cannot save verifier log %s: %s\n",
                path, strerror(errno));
        return error;
    }

    fprintf(output, "fixture=%s\nload_error=%d\nexpected_write_offset=%u\n",
            fixture->name, load_error, fixture->denied_write_offset);
    if (verifier_log[0] != '\0') {
        fputs(verifier_log, output);
        if (verifier_log[strlen(verifier_log) - 1] != '\n')
            fputc('\n', output);
    }

    if (fclose(output) != 0) {
        fprintf(stderr, "cannot finish verifier log %s: %s\n",
                path, strerror(errno));
        return -EIO;
    }

    return 0;
}

static bool is_expected_write_denial(const struct fixture *fixture,
                                     int load_error,
                                     bool positive_controls_admitted)
{
    (void)fixture;

    /*
     * These stores target the PTR_TO_BTF_ID callback argument. The driver's
     * btf_struct_access callback returns -EACCES without adding a stable log
     * string. First admitting all four controls rules out missing struct_ops,
     * kfunc, BTF, and general load support; only then is -EACCES accepted for
     * the three minimal direct-write fixtures. The raw verifier log is retained
     * independently and the built instructions establish the attempted offset.
     */
    return positive_controls_admitted && load_error == -EACCES;
}

int main(int argc, char **argv)
{
    const char *object_dir = ".output";
    const char *log_dir = NULL;
    char default_log_dir[512];
    size_t attempted = 0;
    size_t admitted = 0;
    size_t rejected = 0;
    size_t passed = 0;
    bool positive_controls_admitted = false;
    size_t i;
    int option;

    while ((option = getopt(argc, argv, "d:hl:v")) != -1) {
        switch (option) {
        case 'd':
            object_dir = optarg;
            break;
        case 'l':
            log_dir = optarg;
            break;
        case 'v':
            verbose = true;
            break;
        case 'h':
            usage(argv[0]);
            return 0;
        default:
            usage(argv[0]);
            return 2;
        }
    }

    if (geteuid() != 0) {
        fprintf(stderr, "precondition failed: root privilege is required\n");
        return 2;
    }

    if (access("/sys/kernel/btf/nvidia", R_OK) != 0) {
        fprintf(stderr,
                "precondition failed: running nvidia module has no exported BTF\n");
        return 2;
    }

    if (access("/sys/kernel/btf/nvidia_uvm", R_OK) != 0) {
        fprintf(stderr,
                "precondition failed: running nvidia_uvm module has no exported BTF\n");
        return 2;
    }

    if (!log_dir) {
        if (snprintf(default_log_dir, sizeof(default_log_dir),
                     "%s/revision-sched-verifier-logs", object_dir) >=
            (int)sizeof(default_log_dir)) {
            fprintf(stderr, "default verifier log path is too long\n");
            return 2;
        }
        log_dir = default_log_dir;
    }

    if (mkdir(log_dir, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "cannot create verifier log directory %s: %s\n",
                log_dir, strerror(errno));
        return 2;
    }

    libbpf_set_print(libbpf_log);

    for (i = 0; i < sizeof(fixtures) / sizeof(fixtures[0]); ++i) {
        struct bpf_object *object;
        struct bpf_program *program;
        char *verifier_log;
        char path[512];
        long open_error;
        int load_error;
        int setup_error = 0;
        size_t program_count = 0;

        if (snprintf(path, sizeof(path), "%s/%s", object_dir,
                     fixtures[i].object_name) >= (int)sizeof(path)) {
            fprintf(stderr, "object path too long for %s\n", fixtures[i].name);
            return 2;
        }

        object = bpf_object__open_file(path, NULL);
        open_error = libbpf_get_error(object);
        if (open_error) {
            fprintf(stderr, "ERROR %s open failed: %ld\n",
                    fixtures[i].name, open_error);
            break;
        }

        verifier_log = calloc(1, VERIFIER_LOG_SIZE);
        if (!verifier_log) {
            fprintf(stderr, "ERROR %s verifier log allocation failed\n",
                    fixtures[i].name);
            bpf_object__close(object);
            break;
        }

        bpf_object__for_each_program(program, object) {
            ++program_count;
            setup_error = bpf_program__set_log_level(program, 1);
            if (!setup_error)
                setup_error = bpf_program__set_log_buf(program, verifier_log,
                                                       VERIFIER_LOG_SIZE);
            if (setup_error)
                break;
        }

        if (setup_error || program_count != 1) {
            fprintf(stderr,
                    "ERROR %s verifier log setup failed: error=%d programs=%zu\n",
                    fixtures[i].name, setup_error, program_count);
            free(verifier_log);
            bpf_object__close(object);
            break;
        }

        ++attempted;
        load_error = bpf_object__load(object);
        if (load_error == 0)
            ++admitted;
        else
            ++rejected;

        if (save_verifier_log(log_dir, &fixtures[i], load_error,
                              verifier_log) != 0) {
            free(verifier_log);
            bpf_object__close(object);
            break;
        }

        if ((fixtures[i].expect_load && load_error == 0) ||
            (!fixtures[i].expect_load &&
             is_expected_write_denial(&fixtures[i], load_error,
                                      positive_controls_admitted))) {
            ++passed;
            printf("PASS %s expected=%s observed=%s\n",
                   fixtures[i].name,
                   fixtures[i].expect_load ? "admit" : "reject",
                   load_error == 0 ? "admit" : "reject");
        }
        else {
            printf("FAIL %s expected=%s observed=%s error=%d\n",
                   fixtures[i].name,
                   fixtures[i].expect_load ? "admit" : "reject",
                   load_error == 0 ? "admit" : "reject",
                   load_error);
        }

        if (i < POSITIVE_FIXTURE_COUNT && load_error != 0) {
            fprintf(stderr,
                    "ABORT positive control failed; negative fixtures were not run\n");
            free(verifier_log);
            bpf_object__close(object);
            break;
        }

        if (i + 1 == POSITIVE_FIXTURE_COUNT) {
            positive_controls_admitted = true;
            printf("GUARD positive_controls_admitted=4; running negatives\n");
        }

        free(verifier_log);
        bpf_object__close(object);
    }

    printf("SUMMARY attempted=%zu expected=7 admitted=%zu expected_admitted=4 "
           "rejected=%zu expected_rejected=3 passed=%zu\n",
           attempted, admitted, rejected, passed);

    return (attempted == 7 && admitted == 4 && rejected == 3 && passed == 7)
               ? 0
               : 1;
}
