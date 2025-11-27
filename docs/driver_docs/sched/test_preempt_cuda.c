/*
 * test_preempt_cuda.c - Test GPU TSG preempt with CUDA context
 *
 * This program creates a CUDA context, launches a long-running kernel,
 * and then tests preemption via NVA06C_CTRL_CMD_PREEMPT ioctl.
 *
 * Build:
 *   nvcc -o test_preempt_cuda test_preempt_cuda.c -lcuda
 * or:
 *   gcc -o test_preempt_cuda test_preempt_cuda.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda
 *
 * Run:
 *   sudo ./test_preempt_cuda
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
#include <pthread.h>
#include <signal.h>

#include <cuda.h>

/* NVIDIA escape codes */
#define NV_ESC_RM_CONTROL   0x2A

/* NVIDIA control commands */
#define NVA06C_CTRL_CMD_PREEMPT             0xa06c0105
#define NVA06C_CTRL_CMD_GET_INFO            0xa06c0106
#define NVA06C_CTRL_CMD_SET_TIMESLICE       0xa06c0103
#define NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL 0xa06c0107

/* Basic NV types */
typedef uint32_t NvHandle;
typedef uint32_t NvV32;
typedef uint32_t NvU32;
typedef uint64_t NvU64;
typedef uint8_t  NvBool;
typedef void*    NvP64;

#define NV_TRUE  1
#define NV_FALSE 0

/* NVOS54_PARAMETERS */
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
    NvBool bWait;
    NvBool bManualTimeout;
    NvU32  timeoutUs;
} NVA06C_CTRL_PREEMPT_PARAMS;

/* NVA06C_CTRL_TIMESLICE_PARAMS */
typedef struct {
    NvU64 timesliceUs;
} NVA06C_CTRL_TIMESLICE_PARAMS;

/* Interleave levels */
#define NVA06C_CTRL_INTERLEAVE_LEVEL_LOW    0
#define NVA06C_CTRL_INTERLEAVE_LEVEL_MEDIUM 1
#define NVA06C_CTRL_INTERLEAVE_LEVEL_HIGH   2

/* NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS */
typedef struct {
    NvU32 tsgInterleaveLevel;
} NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS;

/* Global state */
static volatile int g_running = 1;
static int g_nvidia_fd = -1;

/* Get timestamp in microseconds */
static uint64_t get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

/* Signal handler */
static void signal_handler(int sig) {
    g_running = 0;
    printf("\nReceived signal %d, stopping...\n", sig);
}

/* Check CUDA error */
#define CHECK_CUDA(call) do { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char *errStr; \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, errStr); \
        exit(1); \
    } \
} while(0)

/* Simple long-running PTX kernel */
static const char *ptx_source =
"    .version 7.0\n"
"    .target sm_50\n"
"    .address_size 64\n"
"\n"
"    .visible .entry busy_loop(\n"
"        .param .u64 param_iterations\n"
"    )\n"
"    {\n"
"        .reg .u64 %rd<3>;\n"
"        .reg .pred %p1;\n"
"\n"
"        ld.param.u64 %rd1, [param_iterations];\n"
"        mov.u64 %rd2, 0;\n"
"\n"
"    loop:\n"
"        add.u64 %rd2, %rd2, 1;\n"
"        setp.lt.u64 %p1, %rd2, %rd1;\n"
"        @%p1 bra loop;\n"
"\n"
"        ret;\n"
"    }\n";

/* Open NVIDIA device */
static int open_nvidia_device(void) {
    int fd = open("/dev/nvidiactl", O_RDWR);
    if (fd < 0) {
        fd = open("/dev/nvidia0", O_RDWR);
    }
    return fd;
}

/* Send RM control command */
static int rm_control(int fd, NvHandle hClient, NvHandle hObject,
                      NvU32 cmd, void *params, NvU32 paramsSize) {
    NVOS54_PARAMETERS ctrl;
    int ret;

    memset(&ctrl, 0, sizeof(ctrl));
    ctrl.hClient = hClient;
    ctrl.hObject = hObject;
    ctrl.cmd = cmd;
    ctrl.flags = 0;
    ctrl.params = params;
    ctrl.paramsSize = paramsSize;
    ctrl.status = 0;

    ret = ioctl(fd, NV_ESC_RM_CONTROL, &ctrl);

    if (ret < 0) {
        return -errno;
    }
    return ctrl.status;
}

/* Test preempt */
static int test_preempt(int fd, NvHandle hClient, NvHandle hTsg, int wait_ms) {
    NVA06C_CTRL_PREEMPT_PARAMS params;
    uint64_t start, end;
    int ret;

    memset(&params, 0, sizeof(params));
    params.bWait = NV_TRUE;
    params.bManualTimeout = NV_TRUE;
    params.timeoutUs = wait_ms * 1000;

    printf("  Sending PREEMPT (timeout=%dms)...\n", wait_ms);
    start = get_time_us();
    ret = rm_control(fd, hClient, hTsg, NVA06C_CTRL_CMD_PREEMPT,
                     &params, sizeof(params));
    end = get_time_us();

    printf("  Result: %d, Duration: %lu us\n", ret, (unsigned long)(end - start));
    return ret;
}

/* Test set timeslice */
static int test_set_timeslice(int fd, NvHandle hClient, NvHandle hTsg, uint64_t timeslice_us) {
    NVA06C_CTRL_TIMESLICE_PARAMS params;
    int ret;

    memset(&params, 0, sizeof(params));
    params.timesliceUs = timeslice_us;

    printf("  Setting timeslice to %lu us...\n", (unsigned long)timeslice_us);
    ret = rm_control(fd, hClient, hTsg, NVA06C_CTRL_CMD_SET_TIMESLICE,
                     &params, sizeof(params));
    printf("  Result: %d\n", ret);
    return ret;
}

/* Test set interleave level */
static int test_set_interleave(int fd, NvHandle hClient, NvHandle hTsg, NvU32 level) {
    NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS params;
    const char *level_names[] = {"LOW", "MEDIUM", "HIGH"};
    int ret;

    memset(&params, 0, sizeof(params));
    params.tsgInterleaveLevel = level;

    printf("  Setting interleave level to %s (%u)...\n",
           level < 3 ? level_names[level] : "UNKNOWN", level);
    ret = rm_control(fd, hClient, hTsg, NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL,
                     &params, sizeof(params));
    printf("  Result: %d\n", ret);
    return ret;
}

/* Worker thread that runs GPU kernel */
static void *gpu_worker(void *arg) {
    CUcontext ctx = (CUcontext)arg;
    CUmodule module;
    CUfunction kernel;
    CUresult res;
    uint64_t iterations = 10000000000ULL;  /* Very long loop */

    /* Set context */
    res = cuCtxSetCurrent(ctx);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "Worker: Failed to set context\n");
        return NULL;
    }

    /* Load PTX module */
    res = cuModuleLoadData(&module, ptx_source);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "Worker: Failed to load PTX module\n");
        return NULL;
    }

    /* Get kernel function */
    res = cuModuleGetFunction(&kernel, module, "busy_loop");
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "Worker: Failed to get kernel function\n");
        cuModuleUnload(module);
        return NULL;
    }

    printf("GPU Worker: Launching long-running kernel...\n");

    while (g_running) {
        void *args[] = { &iterations };

        /* Launch kernel */
        res = cuLaunchKernel(kernel,
                            1, 1, 1,      /* grid dim */
                            1, 1, 1,      /* block dim */
                            0,            /* shared mem */
                            0,            /* stream */
                            args, NULL);

        if (res != CUDA_SUCCESS) {
            const char *errStr;
            cuGetErrorString(res, &errStr);
            fprintf(stderr, "Worker: Kernel launch failed: %s\n", errStr);
            break;
        }

        /* Synchronize */
        res = cuCtxSynchronize();
        if (res != CUDA_SUCCESS) {
            const char *errStr;
            cuGetErrorString(res, &errStr);
            printf("Worker: Sync result: %s (may be due to preempt)\n", errStr);
        }

        if (g_running) {
            printf("GPU Worker: Kernel completed, relaunching...\n");
        }
    }

    cuModuleUnload(module);
    printf("GPU Worker: Exiting\n");
    return NULL;
}

int main(int argc, char *argv[]) {
    CUdevice device;
    CUcontext ctx;
    pthread_t worker_thread;
    int device_count;

    printf("=== NVIDIA GPU Preempt Test with CUDA ===\n\n");

    /* Setup signal handlers */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    /* Initialize CUDA */
    printf("Initializing CUDA...\n");
    CHECK_CUDA(cuInit(0));

    CHECK_CUDA(cuDeviceGetCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }
    printf("Found %d CUDA device(s)\n", device_count);

    CHECK_CUDA(cuDeviceGet(&device, 0));

    char name[256];
    CHECK_CUDA(cuDeviceGetName(name, sizeof(name), device));
    printf("Using device: %s\n", name);

    /* Create context */
    printf("Creating CUDA context...\n");
    CHECK_CUDA(cuCtxCreate(&ctx, 0, device));

    /* Open NVIDIA device for ioctl */
    g_nvidia_fd = open_nvidia_device();
    if (g_nvidia_fd < 0) {
        fprintf(stderr, "Failed to open NVIDIA device: %s\n", strerror(errno));
        fprintf(stderr, "Note: The preempt ioctl test requires special handles.\n");
        fprintf(stderr, "The CUDA kernel test will still run.\n");
    } else {
        printf("Opened NVIDIA device (fd=%d)\n", g_nvidia_fd);
    }

    /* Start GPU worker thread */
    printf("\nStarting GPU worker thread...\n");
    if (pthread_create(&worker_thread, NULL, gpu_worker, ctx) != 0) {
        perror("Failed to create worker thread");
        goto cleanup;
    }

    /* Give the kernel time to start */
    sleep(2);

    printf("\n=== Testing scheduling controls ===\n");
    printf("Note: These tests require valid client/TSG handles.\n");
    printf("Without internal RM handles, they will return errors.\n\n");

    /*
     * Note: To actually test preempt via ioctl, you need:
     * 1. hClient - the RM client handle
     * 2. hTsg - the TSG (channel group) handle
     *
     * These are internal to the NVIDIA driver and not directly
     * accessible from user space without special APIs.
     *
     * For demonstration, we show the structure but the calls will fail
     * without valid handles.
     */

    if (g_nvidia_fd >= 0) {
        /* Demo calls - will fail without valid handles */
        NvHandle hClient = 0x12345678;  /* Dummy - would need real handle */
        NvHandle hTsg = 0x87654321;     /* Dummy - would need real handle */

        printf("Demo preempt call (will fail with dummy handles):\n");
        test_preempt(g_nvidia_fd, hClient, hTsg, 100);

        printf("\nDemo timeslice call:\n");
        test_set_timeslice(g_nvidia_fd, hClient, hTsg, 1000);

        printf("\nDemo interleave level call:\n");
        test_set_interleave(g_nvidia_fd, hClient, hTsg, NVA06C_CTRL_INTERLEAVE_LEVEL_HIGH);
    }

    printf("\n=== GPU kernel is running ===\n");
    printf("The kernel will run until interrupted (Ctrl+C).\n");
    printf("You can observe GPU activity with: nvidia-smi dmon -s u\n");
    printf("Or watch for preempt events with: dmesg -w | grep -i preempt\n\n");

    /* Wait for signal */
    while (g_running) {
        sleep(1);
        printf(".");
        fflush(stdout);
    }

    printf("\nStopping...\n");

    /* Wait for worker to finish */
    pthread_join(worker_thread, NULL);

cleanup:
    if (g_nvidia_fd >= 0) {
        close(g_nvidia_fd);
    }

    cuCtxDestroy(ctx);
    printf("Done.\n");
    return 0;
}
