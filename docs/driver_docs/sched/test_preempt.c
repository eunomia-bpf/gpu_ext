/*
 * test_preempt.c - Test GPU TSG preempt via NVIDIA ioctl
 *
 * This program tests the NVA06C_CTRL_CMD_PREEMPT ioctl to trigger
 * GPU hardware preemption via NV_RUNLIST_PREEMPT register.
 *
 * Build: gcc -o test_preempt test_preempt.c -lcuda -lnvidia-ml
 * Run:   sudo ./test_preempt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>

/* NVIDIA escape codes from nv_escape.h */
#define NV_ESC_RM_CONTROL   0x2A
#define NV_ESC_RM_ALLOC     0x2B

/* NVIDIA control command from ctrla06c.h */
#define NVA06C_CTRL_CMD_PREEMPT     0xa06c0105
#define NVA06C_CTRL_CMD_GET_INFO    0xa06c0106

/* Max timeout for preempt in microseconds */
#define NVA06C_CTRL_CMD_PREEMPT_MAX_MANUAL_TIMEOUT_US (1000000)

/* Basic NV types */
typedef uint32_t NvHandle;
typedef uint32_t NvV32;
typedef uint32_t NvU32;
typedef uint64_t NvU64;
typedef uint8_t  NvBool;
typedef void*    NvP64;

#define NV_TRUE  1
#define NV_FALSE 0

/* NVOS54_PARAMETERS - RM Control structure */
typedef struct {
    NvHandle hClient;
    NvHandle hObject;
    NvV32    cmd;
    NvU32    flags;
    NvP64    params;
    NvU32    paramsSize;
    NvV32    status;
} __attribute__((packed, aligned(8))) NVOS54_PARAMETERS;

/* NVA06C_CTRL_PREEMPT_PARAMS */
typedef struct {
    NvBool bWait;           /* Wait for preempt completion */
    NvBool bManualTimeout;  /* Use custom timeout */
    NvU32  timeoutUs;       /* Timeout in microseconds */
} NVA06C_CTRL_PREEMPT_PARAMS;

/* NVA06C_CTRL_GET_INFO_PARAMS */
typedef struct {
    NvU32 tsgID;
} NVA06C_CTRL_GET_INFO_PARAMS;

/* Get current timestamp in microseconds */
static uint64_t get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

/* Try to find NVIDIA control device */
static int open_nvidia_ctl(void) {
    int fd;

    /* Try nvidia-ctl first */
    fd = open("/dev/nvidiactl", O_RDWR);
    if (fd >= 0) return fd;

    /* Try nvidia0 */
    fd = open("/dev/nvidia0", O_RDWR);
    if (fd >= 0) return fd;

    return -1;
}

/*
 * Note: This is a simplified test. In real usage, you need:
 * 1. Get a valid hClient from RM
 * 2. Get a valid hObject (TSG handle) from allocated channel group
 *
 * Without CUDA context, we can only demonstrate the ioctl structure.
 */
int test_preempt_ioctl(int fd, NvHandle hClient, NvHandle hTsg) {
    NVOS54_PARAMETERS ctrl_params;
    NVA06C_CTRL_PREEMPT_PARAMS preempt_params;
    int ret;
    uint64_t start_us, end_us;

    printf("Testing NVA06C_CTRL_CMD_PREEMPT (0x%08x)\n", NVA06C_CTRL_CMD_PREEMPT);
    printf("  hClient: 0x%08x\n", hClient);
    printf("  hTsg:    0x%08x\n", hTsg);

    /* Setup preempt parameters */
    memset(&preempt_params, 0, sizeof(preempt_params));
    preempt_params.bWait = NV_TRUE;           /* Wait for completion */
    preempt_params.bManualTimeout = NV_TRUE;  /* Use custom timeout */
    preempt_params.timeoutUs = 100000;        /* 100ms timeout */

    /* Setup control parameters */
    memset(&ctrl_params, 0, sizeof(ctrl_params));
    ctrl_params.hClient = hClient;
    ctrl_params.hObject = hTsg;
    ctrl_params.cmd = NVA06C_CTRL_CMD_PREEMPT;
    ctrl_params.flags = 0;
    ctrl_params.params = &preempt_params;
    ctrl_params.paramsSize = sizeof(preempt_params);
    ctrl_params.status = 0;

    printf("\nSending ioctl NV_ESC_RM_CONTROL...\n");

    start_us = get_time_us();
    ret = ioctl(fd, NV_ESC_RM_CONTROL, &ctrl_params);
    end_us = get_time_us();

    printf("\nResult:\n");
    printf("  ioctl return: %d (errno: %d - %s)\n", ret, errno, strerror(errno));
    printf("  RM status:    0x%08x\n", ctrl_params.status);
    printf("  Duration:     %lu us\n", (unsigned long)(end_us - start_us));

    return ret;
}

/* Scan /proc to find NVIDIA GPU processes and their handles */
void scan_nvidia_processes(void) {
    FILE *fp;
    char line[1024];

    printf("\n=== Scanning for NVIDIA GPU processes ===\n");

    /* Use nvidia-smi to list GPU processes */
    fp = popen("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null", "r");
    if (fp) {
        printf("GPU Processes:\n");
        while (fgets(line, sizeof(line), fp)) {
            printf("  %s", line);
        }
        pclose(fp);
    } else {
        printf("  (nvidia-smi not available)\n");
    }
}

/* Show program usage */
void usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\nOptions:\n");
    printf("  -c <client>  Client handle (hex)\n");
    printf("  -t <tsg>     TSG handle (hex)\n");
    printf("  -s           Scan for NVIDIA processes\n");
    printf("  -h           Show this help\n");
    printf("\nExample:\n");
    printf("  %s -c 0x12345678 -t 0x87654321\n", prog);
    printf("\nNote: You need valid client and TSG handles from an active CUDA context.\n");
    printf("      Without valid handles, the ioctl will return an error.\n");
}

int main(int argc, char *argv[]) {
    int fd;
    int opt;
    NvHandle hClient = 0;
    NvHandle hTsg = 0;
    int do_scan = 0;

    while ((opt = getopt(argc, argv, "c:t:sh")) != -1) {
        switch (opt) {
        case 'c':
            hClient = (NvHandle)strtoul(optarg, NULL, 0);
            break;
        case 't':
            hTsg = (NvHandle)strtoul(optarg, NULL, 0);
            break;
        case 's':
            do_scan = 1;
            break;
        case 'h':
        default:
            usage(argv[0]);
            return 0;
        }
    }

    printf("=== NVIDIA GPU Preempt Test ===\n\n");

    /* Open NVIDIA device */
    fd = open_nvidia_ctl();
    if (fd < 0) {
        perror("Failed to open NVIDIA device");
        printf("Make sure NVIDIA driver is loaded and you have permissions.\n");
        return 1;
    }
    printf("Opened NVIDIA device (fd=%d)\n", fd);

    if (do_scan) {
        scan_nvidia_processes();
    }

    if (hClient != 0 && hTsg != 0) {
        /* Test preempt with provided handles */
        test_preempt_ioctl(fd, hClient, hTsg);
    } else {
        printf("\nNo client/TSG handles provided.\n");
        printf("To test preempt, you need:\n");
        printf("  1. A running CUDA application\n");
        printf("  2. Valid client handle and TSG handle\n");
        printf("\nYou can get these from:\n");
        printf("  - CUDA driver API (cuCtxGetCurrent, cuStreamGetCtx)\n");
        printf("  - nvidia-smi or nvtop to find GPU processes\n");
        printf("  - Debugging CUDA applications\n");

        /* Demo: show what the ioctl structure looks like */
        printf("\n=== Demo: IOCTL structure ===\n");
        printf("NV_ESC_RM_CONTROL = 0x%02x\n", NV_ESC_RM_CONTROL);
        printf("NVA06C_CTRL_CMD_PREEMPT = 0x%08x\n", NVA06C_CTRL_CMD_PREEMPT);
        printf("sizeof(NVOS54_PARAMETERS) = %zu\n", sizeof(NVOS54_PARAMETERS));
        printf("sizeof(NVA06C_CTRL_PREEMPT_PARAMS) = %zu\n", sizeof(NVA06C_CTRL_PREEMPT_PARAMS));
    }

    close(fd);
    return 0;
}
