/* SPDX-License-Identifier: GPL-2.0 */

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

enum {
    UVM_INITIALIZE = 0x30000001,
    UVM_TEST_PMM_BPF_TRANSITION = 277,
};

struct uvm_initialize_params {
    uint64_t flags;
    uint32_t rm_status;
};

struct uvm_test_pmm_bpf_transition_params {
    uint32_t rm_status;
};

int main(void)
{
    struct uvm_initialize_params initialize = {0};
    struct uvm_test_pmm_bpf_transition_params transition = {0};
    int fd;

    fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr, "open /dev/nvidia-uvm failed: %s\n", strerror(errno));
        return 2;
    }

    if (ioctl(fd, UVM_INITIALIZE, &initialize) != 0 || initialize.rm_status != 0) {
        fprintf(stderr,
                "UVM_INITIALIZE failed: errno=%d rm_status=0x%x\n",
                errno,
                initialize.rm_status);
        close(fd);
        return 2;
    }

    if (ioctl(fd, UVM_TEST_PMM_BPF_TRANSITION, &transition) != 0 ||
        transition.rm_status != 0) {
        fprintf(stderr,
                "UVM_TEST_PMM_BPF_TRANSITION failed: errno=%d rm_status=0x%x\n",
                errno,
                transition.rm_status);
        close(fd);
        return 1;
    }

    close(fd);
    printf("PASS UVM_TEST_PMM_BPF_TRANSITION rm_status=0\n");
    return 0;
}
