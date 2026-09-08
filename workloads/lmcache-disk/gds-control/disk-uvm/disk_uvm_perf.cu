/*
 * disk_uvm_perf.cu
 *
 * Small real-performance client for the disk-backed managed-UVM mechanism in
 * the nvidia-uvm driver built from gpu_ext-kernel-575-gds (branch
 * revision/gpu-storage-decision-575, source checkpoint db156f27).
 *
 * Flow:
 *   1. Open a real local backing file with O_DIRECT (plus O_DSYNC with
 *      --durability). The driver's offload worker does kernel_read /
 *      kernel_write on this exact file descriptor, so O_DIRECT is what makes
 *      the disk arm measure real direct I/O instead of filesystem page cache.
 *      If O_DIRECT is unsupported on the path, the client reports it as a
 *      real error and never falls back to buffered I/O. The actual kernel
 *      open flags are recorded verbatim from /proc/self/fdinfo.
 *   2. Allocate real CUDA managed memory (a multiple of the 2 MiB UVM VA
 *      block size) and initialize it with a sealed deterministic KV-like
 *      per-word pattern through one host->managed H2D copy; the host pattern
 *      buffer is discarded afterwards.
 *   3. Locate the /dev/nvidia-uvm fd whose UVM va space owns the managed
 *      range (libcuda may open several /dev/nvidia-uvm fds in this process;
 *      only one carries the process UVM va space, and that owner is selected
 *      by probing the managed range with a read-only QUERY, not by counting
 *      fds) and issue the disk-backing ioctls on it:
 *          84 UVM_DISK_BACKING_REGISTER  (attach + seal the whole range)
 *          85 UVM_DISK_BACKING_OFFLOAD   (async disk offload)
 *          86 UVM_DISK_BACKING_QUERY     (completion / state counters)
 *   4. Time reads at the SAME managed VA. Every timed read traverses the
 *      full buffer, and each CTA/page additionally stores one plain sampled
 *      element (an individually verifiable word, no reduction):
 *          baseline_gpu : data resident on the GPU (separate from the disk
 *                         arm; no backing file involved)
 *          cpu_restore  : first-touch HOST read after offload: the UVM CPU
 *                         fault hydrates the chunk from the backing file
 *                         (direct-I/O file read, no copy)
 *          gpu_restore  : first-touch GPU read after a second offload: the
 *                         UVM GPU fault restores via CPU-first hydration from
 *                         the file, then CPU->GPU copy
 *          steady_gpu   : GPU read with the data resident again
 *
 * A small fixed set of sampled words is printed alongside the word the
 * deterministic pattern defines at that index, giving a simple observable
 * read result. No aggregate content fingerprint is generated or compared.
 * Driver-reported I/O error pages exit nonzero; all raw numbers are printed
 * and saved.
 *
 * No cuFile, no GDS transport, and no copied host array serves any disk-UVM
 * read arm: the only read paths are the managed VA itself, from host or GPU.
 * The backing file and the raw output are preserved on disk after the run.
 */

#include <cuda_runtime.h>

#include <cstddef>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* ------------------------------------------------------------------ ABI --
 * Exact userspace mirror of UVM_DISK_BACKING_{REGISTER,OFFLOAD,QUERY}_PARAMS
 * in kernel-open/nvidia-uvm/uvm_ioctl.h (ABI v1). On Linux
 * UVM_IOCTL_BASE(i) == i, so the raw ioctl command numbers are 84, 85, 86.
 */
#define UVM_DISK_BACKING_ABI_VERSION 1u
#define UVM_DISK_BACKING_REGISTER    84u
#define UVM_DISK_BACKING_OFFLOAD     85u
#define UVM_DISK_BACKING_QUERY       86u

#define UVM_VA_BLOCK_SIZE ((size_t)1u << 21) /* 2 MiB */
#define SYS_PAGE_SIZE     4096u
#define WORDS_PER_PAGE    (SYS_PAGE_SIZE / 4u)

typedef struct
{
    uint32_t abiVersion;
    uint32_t pad0;
    uint64_t rangeStart;
    uint64_t rangeEnd;
    uint64_t fileOffset;
    int32_t  fileFd;
    uint32_t pad1;
    int32_t  rmStatus;
} uvm_disk_backing_register_params_t;

typedef struct
{
    uint32_t abiVersion;
    uint32_t pad0;
    uint64_t rangeStart;
    uint64_t rangeEnd;
    int32_t  rmStatus;
} uvm_disk_backing_offload_params_t;

typedef struct
{
    uint32_t abiVersion;
    uint32_t pad0;
    uint64_t rangeStart;
    uint64_t rangeEnd;
    uint32_t totalNumPages;
    uint32_t onDiskPages;
    uint32_t pendingPages;
    uint32_t errorPages;
    int32_t  rmStatus;
} uvm_disk_backing_query_params_t;

static_assert(sizeof(uvm_disk_backing_register_params_t) == 48,
              "UVM_DISK_BACKING_REGISTER_PARAMS layout");
static_assert(offsetof(uvm_disk_backing_register_params_t, fileFd) == 32,
              "fileFd offset");
static_assert(sizeof(uvm_disk_backing_offload_params_t) == 32,
              "UVM_DISK_BACKING_OFFLOAD_PARAMS layout");
static_assert(sizeof(uvm_disk_backing_query_params_t) == 48,
              "UVM_DISK_BACKING_QUERY_PARAMS layout");
static_assert(offsetof(uvm_disk_backing_query_params_t, totalNumPages) == 24,
              "totalNumPages offset");

/* -------------------------------------------------------------- helpers -- */
#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        cudaError_t cuda_err__ = (expr);                                     \
        if (cuda_err__ != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, \
                    __LINE__, cudaGetErrorString(cuda_err__));               \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

static uint64_t now_ns(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Per-word sealed KV-like content: one deterministic 32-bit value per word.
 * Used to fill the data and to state the expected value of a single
 * individually sampled word. No aggregate over the buffer is ever computed.
 */
static uint32_t pattern_word(uint64_t i)
{
    uint64_t x = i + 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return (uint32_t)(x ^ (x >> 31));
}

static const char *nv_status_str(int32_t s)
{
    switch (s) {
    case 0x00000000: return "NV_OK";
    case 0x00000003: return "NV_ERR_BUSY_RETRY";
    case 0x00000016: return "NV_ERR_ILLEGAL_ACTION";
    case 0x0000001E: return "NV_ERR_INVALID_ADDRESS";
    case 0x0000001F: return "NV_ERR_INVALID_ARGUMENT";
    case 0x00000038: return "NV_ERR_INVALID_OPERATION";
    case 0x00000040: return "NV_ERR_INVALID_STATE";
    case 0x00000051: return "NV_ERR_NO_MEMORY";
    case 0x00000057: return "NV_ERR_OBJECT_NOT_FOUND";
    case 0x00000063: return "NV_ERR_STATE_IN_USE";
    case 0x00010005: return "NV_WARN_MORE_PROCESSING_REQUIRED";
    default:         return "?";
    }
}

/* Record the kernel's actual open flags for an fd, verbatim from
 * /proc/self/fdinfo/<fd>. */
static void read_fdinfo_flags(int fd, char *out, size_t out_len)
{
    char path[64];
    FILE *f;
    char line[256];

    snprintf(path, sizeof(path), "/proc/self/fdinfo/%d", fd);
    f = fopen(path, "re");
    if (!f) {
        snprintf(out, out_len, "unreadable");
        return;
    }
    out[0] = '\0';
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "flags:", 6) == 0) {
            size_t p = 6;

            while (line[p] == '\t' || line[p] == ' ')
                ++p;
            strncpy(out, line + p, out_len - 1);
            out[out_len - 1] = '\0';
            out[strcspn(out, "\r\n")] = '\0';
            break;
        }
    }
    fclose(f);
}

/* NV status values for the read-only fd probe, with values from
 * src/common/sdk/nvidia/inc/nvstatuscodes.h. */
#define NV_OK                   0x00000000
#define NV_ERR_ILLEGAL_ACTION   0x00000016
#define NV_ERR_INVALID_ARGUMENT 0x0000001F
#define NV_ERR_INVALID_STATE    0x00000040

static int uvm_query(int uvm_fd, uint64_t start, uint64_t end,
                     uint32_t *total, uint32_t *on_disk, uint32_t *pending,
                     uint32_t *err_pages);

/* Find the /dev/nvidia-uvm fd whose UVM va space owns the managed range. The
 * CUDA driver opens more than one /dev/nvidia-uvm fd in this process; each
 * open is a distinct UVM file, and only the fd CUDA actually initialized
 * carries the process UVM va space with the managed range in it, so the fd
 * count alone does not identify the owner (multiple fds are normal, not an
 * error). The client itself never opens /dev/nvidia-uvm.
 * The owner is selected by a read-only probe: for every candidate fd, issue
 * UVM_DISK_BACKING_QUERY on the managed range. Per the driver
 * (uvm_api_disk_backing_query in uvm.c), the answers are:
 *   NV_OK / NV_ERR_INVALID_STATE : this fd's va space contains the range;
 *     NV_ERR_INVALID_STATE is the pre-REGISTER state, where the range is
 *     owned but has no disk backing attached yet
 *   NV_ERR_INVALID_ARGUMENT      : the range is absent from this fd's va
 *     space, so this fd does not own the range
 *   NV_ERR_ILLEGAL_ACTION        : this fd was never initialized to a va
 *     space (the routing init check in uvm_api.h rejects it before the
 *     handler runs), so it does not own the range
 * Returns: the owning fd, -1 if no /dev/nvidia-uvm fd exists, -2 if no
 * candidate owns the range, -3 if more than one candidate does. */
static int find_cuda_uvm_fd(uint64_t mstart, uint64_t mend)
{
    DIR *directory;
    struct dirent *d;
    char link[64];
    char target[512];
    int candidates[4096];
    int probe[4096];
    int n_candidates = 0;
    int n_owners = 0;
    int owner = -1;
    int i;

    directory = opendir("/proc/self/fd");
    if (!directory) {
        perror("opendir /proc/self/fd");
        return -1;
    }

    while ((d = readdir(directory)) != NULL) {
        ssize_t n;

        if (d->d_name[0] == '.')
            continue;
        if (n_candidates >= 4096)
            break;
        snprintf(link, sizeof(link), "/proc/self/fd/%s", d->d_name);
        n = readlink(link, target, sizeof(target) - 1);
        if (n <= 0)
            continue;
        target[n] = '\0';
        if (strcmp(target, "/dev/nvidia-uvm") == 0)
            candidates[n_candidates++] = atoi(d->d_name);
    }
    closedir(directory);

    if (n_candidates == 0)
        return -1;

    for (i = 0; i < n_candidates; ++i) {
        uint32_t total, od, pend, er;

        probe[i] = uvm_query(candidates[i], mstart, mend, &total, &od, &pend,
                             &er);
        if (probe[i] == NV_OK || probe[i] == NV_ERR_INVALID_STATE) {
            owner = candidates[i];
            ++n_owners;
        }
    }

    for (i = 0; i < n_candidates; ++i) {
        if (probe[i] < 0)
            fprintf(stderr, "uvm_fd candidate %d: QUERY probe ioctl errno=%d "
                    "(%s)\n",
                    candidates[i], -probe[i], strerror(-probe[i]));
        else
            fprintf(stderr, "uvm_fd candidate %d: QUERY probe "
                    "rmStatus=0x%" PRIX32 " (%s)\n",
                    candidates[i], (uint32_t)probe[i],
                    nv_status_str(probe[i]));
    }

    if (n_owners == 0)
        return -2;
    if (n_owners > 1)
        return -3;
    return owner;
}

/* ------------------------------------------------- disk-backing ioctls -- */
static int uvm_register(int uvm_fd, uint64_t start, uint64_t end, int backing_fd)
{
    uvm_disk_backing_register_params_t p;

    memset(&p, 0, sizeof(p));
    p.abiVersion = UVM_DISK_BACKING_ABI_VERSION;
    p.rangeStart = start;
    p.rangeEnd = end;
    p.fileOffset = 0;
    p.fileFd = backing_fd;
    if (ioctl(uvm_fd, UVM_DISK_BACKING_REGISTER, &p) != 0)
        return -errno;
    return p.rmStatus;
}

static int uvm_offload(int uvm_fd, uint64_t start, uint64_t end)
{
    uvm_disk_backing_offload_params_t p;

    memset(&p, 0, sizeof(p));
    p.abiVersion = UVM_DISK_BACKING_ABI_VERSION;
    p.rangeStart = start;
    p.rangeEnd = end;
    if (ioctl(uvm_fd, UVM_DISK_BACKING_OFFLOAD, &p) != 0)
        return -errno;
    return p.rmStatus;
}

static int uvm_query(int uvm_fd, uint64_t start, uint64_t end,
                     uint32_t *total, uint32_t *on_disk, uint32_t *pending,
                     uint32_t *err_pages)
{
    uvm_disk_backing_query_params_t p;

    memset(&p, 0, sizeof(p));
    p.abiVersion = UVM_DISK_BACKING_ABI_VERSION;
    p.rangeStart = start;
    p.rangeEnd = end;
    if (ioctl(uvm_fd, UVM_DISK_BACKING_QUERY, &p) != 0)
        return -errno;
    *total = p.totalNumPages;
    *on_disk = p.onDiskPages;
    *pending = p.pendingPages;
    *err_pages = p.errorPages;
    return p.rmStatus;
}

/* Poll QUERY until no page of the span is pending; fills the final counters
 * on success. *t_done_ns receives the absolute wall-clock time (ns) of the
 * completing poll; the caller derives the duration from its own t0. */
static int wait_offload_done(int uvm_fd, uint64_t start, uint64_t end,
                             unsigned poll_us, uint64_t *t_done_ns,
                             uint32_t *on_disk, uint32_t *err_pages)
{
    for (;;) {
        uint32_t total, od, pend, er;
        int st = uvm_query(uvm_fd, start, end, &total, &od, &pend, &er);

        if (st != 0)
            return st;
        if (pend == 0) {
            *t_done_ns = now_ns();
            *on_disk = od;
            *err_pages = er;
            return 0;
        }
        usleep(poll_us);
    }
}

/* Report an offload ioctl result; negative values are -errno. */
static void report_offload_failure(const char *tag, int st)
{
    if (st < 0)
        fprintf(stderr, "%s ioctl: errno=%d (%s)\n", tag, -st,
                strerror(-st));
    else
        fprintf(stderr, "%s: rmStatus=0x%x (%s)\n", tag, st,
                nv_status_str(st));
}

/* --------------------------------------------------------------- kernels -- */
static cudaStream_t g_stream = NULL;
#define GPU_CTAS 1024
#define GPU_THREADS 256

/* Full-buffer traversal: every CTA reads all words of its strided region
 * (volatile source loads cannot be eliminated) and additionally stores one plain
 * sampled element - the first word of the CTA's region - with no reduction.
 */
__global__ void kv_read_traverse(const volatile uint32_t *p, size_t n_words,
                                 uint32_t *sample_out)
{
    volatile uint32_t sink = 0u;
    size_t stride = (size_t)gridDim.x * (size_t)blockDim.x;
    size_t base = (size_t)blockIdx.x * (size_t)blockDim.x;
    size_t i;

    for (i = base + threadIdx.x; i < n_words; i += stride)
        sink = p[i];

    if (threadIdx.x == 0)
        sample_out[blockIdx.x] = p[base];
}

/* Time a full GPU read of the managed range. Returns wall time (launch +
 * sync) in ns; the sampled CTA words come back in samples[3] at the CTA
 * indices 0, GPU_CTAS/2, GPU_CTAS-1. */
static uint64_t gpu_read_full(const void *managed, size_t size,
                              uint32_t *samples, int *sample_ctas)
{
    unsigned long long *unused = NULL;
    uint32_t *dev_samples;
    uint32_t host_samples[GPU_CTAS];
    size_t n_words = size / sizeof(uint32_t);
    uint64_t t0, t1;
    int cta;

    (void)unused;
    CUDA_CHECK(cudaMalloc((void **)&dev_samples,
                          sizeof(uint32_t) * GPU_CTAS));

    t0 = now_ns();
    kv_read_traverse<<<GPU_CTAS, GPU_THREADS, 0, g_stream>>>(
        (const uint32_t *)managed, n_words, dev_samples);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
    t1 = now_ns();

    CUDA_CHECK(cudaMemcpy(host_samples, dev_samples,
                          sizeof(uint32_t) * GPU_CTAS,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dev_samples));

    sample_ctas[0] = 0;
    sample_ctas[1] = GPU_CTAS / 2;
    sample_ctas[2] = GPU_CTAS - 1;
    for (cta = 0; cta < 3; ++cta)
        samples[cta] = host_samples[sample_ctas[cta]];
    return t1 - t0;
}

/* Time a full HOST read of the managed range; one sampled word per 4K page
 * is stored into page_words (n_pages entries). */
static uint64_t cpu_read_full(const void *managed, size_t size,
                              uint32_t *page_words, size_t n_pages)
{
    volatile const uint32_t *p = (volatile const uint32_t *)managed;
    size_t n_words = size / sizeof(uint32_t);
    size_t i;
    uint64_t t0, t1;

    t0 = now_ns();
    for (i = 0; i < n_words; ++i) {
        uint32_t v = p[i];

        if ((i & (WORDS_PER_PAGE - 1)) == 0)
            page_words[i / WORDS_PER_PAGE] = v;
    }
    t1 = now_ns();
    return t1 - t0;
}

/* Print one sampled word next to the word the pattern defines at that
 * index. A single-word observation only; nothing is aggregated. */
static int describe_sampled_word(uint32_t value, uint64_t word_index)
{
    uint32_t expected = pattern_word(word_index);

    printf("    word[%" PRIu64 "] value=0x%08" PRIX32 " expected=0x%08" PRIX32
           " match=%d\n",
           word_index, value, expected, (int)(value == expected));
    return (int)(value == expected);
}

/* -------------------------------------------------------------- raw sink -- */
static char g_raw[32768];
static size_t g_raw_len = 0;

static void raw_add(const char *fmt, ...)
{
    va_list ap;
    int n;

    va_start(ap, fmt);
    n = vsnprintf(g_raw + g_raw_len, sizeof(g_raw) - g_raw_len, fmt, ap);
    va_end(ap);
    if (n > 0)
        g_raw_len += (size_t)n;
}

static void save_raw(void)
{
    char path[512];
    FILE *f;

    snprintf(path, sizeof(path), "raw/disk_uvm_%" PRIu32 "_%" PRIu64 ".raw",
             (uint32_t)getpid(), now_ns() / 1000000000ull);
    if (mkdir("raw", 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "warn: mkdir raw/: %s\n", strerror(errno));
        return;
    }
    f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "warn: cannot write %s: %s\n", path, strerror(errno));
        return;
    }
    fwrite(g_raw, 1, g_raw_len, f);
    fclose(f);
    fprintf(stderr, "raw output saved: %s\n", path);
}

/* ------------------------------------------------------------------- main -- */
static int parse_size(const char *s, size_t *out)
{
    char *end;
    unsigned long long v = strtoull(s, &end, 0);
    size_t mult = 1;

    if (end == s || v == 0)
        return -1;
    if (*end == 'K' || *end == 'k') {
        mult = 1024;
        ++end;
    } else if (strncmp(end, "MiB", 3) == 0) {
        mult = 1024ull * 1024;
        end += 3;
    } else if (strncmp(end, "GiB", 3) == 0) {
        mult = 1024ull * 1024 * 1024;
        end += 3;
    } else if (*end == 'M') {
        mult = 1024ull * 1024;
        ++end;
    } else if (*end == 'G') {
        mult = 1024ull * 1024 * 1024;
        ++end;
    }
    if (*end != '\0')
        return -1;
    *out = (size_t)v * mult;
    return 0;
}

int main(int argc, char **argv)
{
    size_t size = 256ull << 20;
    const char *backing_path = "disk_uvm_backing.bin";
    int device = 0;
    int durability = 0;
    unsigned poll_us = 200;
    int backing_flags;
    int backing_fd;
    int uvm_fd;
    char fdinfo_flags[64];
    void *managed = NULL;
    uint32_t *init_buf;
    uint64_t mstart, mend;
    size_t n_words, n_pages;
    uint32_t *page_words;
    int i;
    int st;
    uint64_t t;
    int exit_code = 0;
    cudaDeviceProp prop;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            if (parse_size(argv[++i], &size) != 0) {
                fprintf(stderr, "bad --size '%s'\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--backing-file") == 0 && i + 1 < argc) {
            backing_path = argv[++i];
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--durability") == 0) {
            durability = 1;
        } else if (strcmp(argv[i], "--poll-us") == 0 && i + 1 < argc) {
            poll_us = (unsigned)atoi(argv[++i]);
            if (poll_us == 0)
                poll_us = 200;
        } else {
            fprintf(stderr,
                    "usage: %s [--size N[K|MiB|GiB]] [--backing-file PATH] "
                    "[--device N] [--durability] [--poll-us US]\n",
                    argv[0]);
            return 1;
        }
    }

    if (size < UVM_VA_BLOCK_SIZE || size % UVM_VA_BLOCK_SIZE != 0) {
        fprintf(stderr,
                "size must be a multiple of the UVM VA block size (%zu); "
                "got %zu\n",
                UVM_VA_BLOCK_SIZE, size);
        return 1;
    }

    /* 1. Real local backing file, direct I/O only. */
    backing_flags = O_RDWR | O_CREAT | O_DIRECT;
    if (durability)
        backing_flags |= O_DSYNC;

    backing_fd = open(backing_path, backing_flags, 0644);
    if (backing_fd < 0) {
        fprintf(stderr,
                "BLOCKER: open(\"%s\", O_RDWR|O_CREAT|O_DIRECT%s) failed: %s. "
                "The disk arm requires direct I/O; buffered I/O fallback is "
                "refused by design. Use a local filesystem/device that "
                "supports O_DIRECT.\n",
                backing_path, durability ? "|O_DSYNC" : "", strerror(errno));
        return 1;
    }
    if (ftruncate(backing_fd, (off_t)size) != 0) {
        fprintf(stderr, "ftruncate(%s, %zu): %s\n", backing_path, size,
                strerror(errno));
        return 1;
    }
    read_fdinfo_flags(backing_fd, fdinfo_flags, sizeof(fdinfo_flags));

    /* 2. Real CUDA managed memory + sealed KV-like content. */
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    CUDA_CHECK(cudaStreamCreate(&g_stream));

    st = cudaMallocManaged(&managed, size);
    if (st != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged(%zu): %s (managed memory / UVA "
                        "unsupported on this device?)\n",
                size, cudaGetErrorString((cudaError_t)st));
        return 1;
    }
    mstart = (uint64_t)managed;
    mend = mstart + size - 1;

    if (mstart % UVM_VA_BLOCK_SIZE != 0) {
        fprintf(stderr,
                "BLOCKER: cudaMallocManaged returned 0x%" PRIx64 ", not "
                "aligned to the UVM VA block size (%zu). The REGISTER ABI "
                "attaches a backing to the whole managed range, so the "
                "allocation must be block-aligned; no sub-range "
                "registration exists in this driver revision.\n",
                mstart, UVM_VA_BLOCK_SIZE);
        return 3;
    }

    n_words = size / sizeof(uint32_t);
    n_pages = size / SYS_PAGE_SIZE;
    init_buf = (uint32_t *)malloc(size);
    page_words = (uint32_t *)malloc(n_pages * sizeof(uint32_t));
    if (!init_buf || !page_words) {
        fprintf(stderr, "malloc: out of memory\n");
        return 1;
    }
    for (i = 0; i < (int)n_words; ++i)
        init_buf[i] = pattern_word((uint64_t)i);
    CUDA_CHECK(
        cudaMemcpyAsync(managed, init_buf, size, cudaMemcpyHostToDevice,
                        g_stream));
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
    free(init_buf); /* the pattern buffer is never a read path afterwards */

    /* 3. The /dev/nvidia-uvm fd whose va space owns the managed range,
     *     selected by the read-only range probe. */
    uvm_fd = find_cuda_uvm_fd(mstart, mend);
    if (uvm_fd == -1) {
        fprintf(stderr,
                "BLOCKER: no /dev/nvidia-uvm fd in this process. UVA/UVM may "
                "be disabled, or the CUDA runtime was not initialized before "
                "this point.\n");
        return 3;
    }
    if (uvm_fd == -2) {
        fprintf(stderr,
                "BLOCKER: /dev/nvidia-uvm fds exist, but no candidate's va "
                "space owns the managed range 0x%" PRIx64 "-0x%" PRIx64
                " (per-fd QUERY probes above). UVA/UVM may be disabled, or "
                "the CUDA runtime was not initialized before this point.\n",
                mstart, mend);
        return 3;
    }
    if (uvm_fd == -3) {
        fprintf(stderr,
                "BLOCKER: more than one /dev/nvidia-uvm fd's va space owns "
                "the managed range (per-fd QUERY probes above); a managed "
                "range lives in exactly one va space, so this should not "
                "happen.\n");
        return 3;
    }

    raw_add(
        "disk_uvm_perf raw\n"
        "pid=%d\n"
        "device=%d gpu=%s\n"
        "size_bytes=%zu\n"
        "uvm_va_block_bytes=%zu\n"
        "managed_range=0x%" PRIx64 "-0x%" PRIx64 "\n"
        "uvm_fd=%d\n"
        "backing_file=%s\n"
        "backing_requested_flags=%s\n"
        "backing_kernel_flags=%s\n",
        (int)getpid(), device, prop.name, size, UVM_VA_BLOCK_SIZE, mstart,
        mend, uvm_fd, backing_path,
        durability ? "O_RDWR|O_CREAT|O_DIRECT|O_DSYNC"
                   : "O_RDWR|O_CREAT|O_DIRECT",
        fdinfo_flags);

    /* 4a. Baseline: data is GPU-resident now; this arm is independent of
     *     the backing file. */
    {
        uint32_t samples[3];
        int sample_ctas[3];
        int cta;

        t = gpu_read_full(managed, size, samples, sample_ctas);
        printf("baseline_gpu: t_ns=%" PRIu64 "\n", t);
        for (cta = 0; cta < 3; ++cta) {
            uint64_t word_idx =
                (uint64_t)sample_ctas[cta] * GPU_THREADS;
            int ok = describe_sampled_word(samples[cta], word_idx);
            raw_add("    baseline_gpu word[%" PRIu64 "] value=0x%08" PRIX32
                    " match=%d\n",
                    word_idx, samples[cta], ok);
            if (!ok)
                exit_code = 2;
        }
        raw_add("arm=baseline_gpu t_ns=%" PRIu64 "\n", t);
    }

    /* 4b. REGISTER: attach + seal the whole managed range read-only. */
    {
        uint64_t t0 = now_ns();
        uint32_t total, od, pend, er;

        st = uvm_register(uvm_fd, mstart, mend, backing_fd);
        t = now_ns() - t0;
        if (st < 0) {
            fprintf(stderr,
                    "BLOCKER: REGISTER ioctl failed with errno=%d (%s). "
                    "ENOTTY means the loaded nvidia-uvm module predates the "
                    "disk-backing ioctls 84/85/86; rebuild/reload the module "
                    "from this tree.\n",
                    -st, strerror(-st));
            return 3;
        }
        if (st != 0) {
            fprintf(stderr, "REGISTER: rmStatus=0x%x (%s)\n", st,
                    nv_status_str(st));
            return 1;
        }
        st = uvm_query(uvm_fd, mstart, mend, &total, &od, &pend, &er);
        if (st != 0) {
            fprintf(stderr, "QUERY after register: rmStatus=0x%x (%s)\n", st,
                    nv_status_str(st));
            return 1;
        }
        printf("register: t_ns=%" PRIu64 " total=%u on_disk=%u pending=%u "
               "error=%u\n",
               t, total, od, pend, er);
        raw_add("arm=register t_ns=%" PRIu64 " total=%u on_disk=%u "
                "pending=%u error=%u\n",
                t, total, od, pend, er);
        if (total != (uint32_t)n_pages || od != 0 || pend != 0 || er != 0)
            exit_code = 1;
    }

    /* 4c. OFFLOAD #1: write the whole range to the backing file (direct
     *     I/O) and release the in-memory copies. */
    {
        uint64_t t0 = now_ns();
        uint64_t t_submit;
        uint32_t od, er;

        st = uvm_offload(uvm_fd, mstart, mend);
        t_submit = now_ns() - t0;
        if (st != 0) {
            report_offload_failure("OFFLOAD", st);
            return 1;
        }
        st = wait_offload_done(uvm_fd, mstart, mend, poll_us, &t, &od, &er);
        if (st != 0) {
            fprintf(stderr, "QUERY while waiting offload: rmStatus=0x%x (%s)\n",
                    st, nv_status_str(st));
            return 1;
        }
        printf("offload1: t_submit_ns=%" PRIu64 " t_complete_ns=%" PRIu64
               " bytes_expected=%zu on_disk=%u pending=%u error=%u\n",
               t_submit, t - t0, size, od, 0u, er);
        raw_add("arm=offload1 t_submit_ns=%" PRIu64 " t_complete_ns=%" PRIu64
                 " bytes_expected=%zu on_disk=%u pending=%u error=%u\n",
                 t_submit, t - t0, size, od, 0u, er);
        if (er != 0 || od != (uint32_t)n_pages)
            exit_code = 2;
    }

    /* 4d. First-touch HOST read at the same managed VA: UVM CPU faults
     *     hydrate chunks from the backing file. */
    {
        size_t mid = n_pages / 2;
        int ok;

        t = cpu_read_full(managed, size, page_words, n_pages);
        printf("cpu_restore: t_ns=%" PRIu64 "\n", t);
        printf("    page[0] word[0] value=0x%08" PRIX32 " expected=0x%08" PRIX32 "\n",
               page_words[0], pattern_word(0));
        ok = (int)(page_words[0] == pattern_word(0));
        printf("    page[%" PRIu64 "] word[%" PRIu64 "] value=0x%08" PRIX32
               " expected=0x%08" PRIX32 " match=%d\n",
               mid, mid * WORDS_PER_PAGE,
               page_words[mid], pattern_word(mid * WORDS_PER_PAGE),
               (int)(page_words[mid] ==
                     pattern_word(mid * WORDS_PER_PAGE)));
        printf("    page[%" PRIu64 "] word[%" PRIu64 "] value=0x%08" PRIX32
               " expected=0x%08" PRIX32 " match=%d\n",
               n_pages - 1, (n_pages - 1) * WORDS_PER_PAGE,
               page_words[n_pages - 1],
               pattern_word((n_pages - 1) * WORDS_PER_PAGE),
               (int)(page_words[n_pages - 1] ==
                     pattern_word((n_pages - 1) * WORDS_PER_PAGE)));
        raw_add("arm=cpu_restore t_ns=%" PRIu64 "\n"
                "    cpu_restore word[0] value=0x%08" PRIX32 " match=%d\n"
                "    cpu_restore word[%" PRIu64 "] value=0x%08" PRIX32
                " match=%d\n"
                "    cpu_restore word[%" PRIu64 "] value=0x%08" PRIX32
                " match=%d\n",
                t,
                page_words[0], ok,
                mid * WORDS_PER_PAGE, page_words[mid],
                (int)(page_words[mid] == pattern_word(mid * WORDS_PER_PAGE)),
                (n_pages - 1) * WORDS_PER_PAGE, page_words[n_pages - 1],
                (int)(page_words[n_pages - 1] ==
                      pattern_word((n_pages - 1) * WORDS_PER_PAGE)));
        if (page_words[0] != pattern_word(0) ||
            page_words[mid] != pattern_word(mid * WORDS_PER_PAGE) ||
            page_words[n_pages - 1] !=
                pattern_word((n_pages - 1) * WORDS_PER_PAGE))
            exit_code = 2;
    }

    /* 4e. OFFLOAD #2: spans are already on disk, so this pass performs no
     *     file writes; it unmaps and frees the CPU copies restored above. */
    {
        uint64_t t0 = now_ns();
        uint64_t t_submit;
        uint32_t od, er;

        st = uvm_offload(uvm_fd, mstart, mend);
        t_submit = now_ns() - t0;
        if (st != 0) {
            report_offload_failure("OFFLOAD#2", st);
            return 1;
        }
        st = wait_offload_done(uvm_fd, mstart, mend, poll_us, &t, &od, &er);
        if (st != 0) {
            fprintf(stderr, "QUERY while waiting offload#2: rmStatus=0x%x (%s)\n",
                    st, nv_status_str(st));
            return 1;
        }
        printf("offload2_release: t_submit_ns=%" PRIu64 " t_complete_ns=%" PRIu64
               " bytes_expected=0 (spans already on disk) on_disk=%u "
               "pending=%u error=%u\n",
               t_submit, t - t0, od, 0u, er);
        raw_add("arm=offload2_release t_submit_ns=%" PRIu64
                 " t_complete_ns=%" PRIu64 " bytes_expected=0 on_disk=%u "
                 "pending=%u error=%u\n",
                 t_submit, t - t0, od, 0u, er);
        if (er != 0)
            exit_code = 2;
    }

    /* 4f. First-touch GPU read at the same managed VA: UVM GPU fault ->
     *     CPU-first hydration from the backing file -> CPU->GPU copy. */
    {
        uint32_t samples[3];
        int sample_ctas[3];
        int cta;

        t = gpu_read_full(managed, size, samples, sample_ctas);
        printf("gpu_restore: t_ns=%" PRIu64 "\n", t);
        for (cta = 0; cta < 3; ++cta) {
            uint64_t word_idx = (uint64_t)sample_ctas[cta] * GPU_THREADS;
            int ok = describe_sampled_word(samples[cta], word_idx);
            raw_add("    gpu_restore word[%" PRIu64 "] value=0x%08" PRIX32
                    " match=%d\n",
                    word_idx, samples[cta], ok);
            if (!ok)
                exit_code = 2;
        }
        raw_add("arm=gpu_restore t_ns=%" PRIu64 "\n", t);
    }

    /* 4g. Steady state: data is GPU-resident again. */
    {
        uint32_t samples[3];
        int sample_ctas[3];
        int cta;

        t = gpu_read_full(managed, size, samples, sample_ctas);
        printf("steady_gpu: t_ns=%" PRIu64 "\n", t);
        for (cta = 0; cta < 3; ++cta) {
            uint64_t word_idx = (uint64_t)sample_ctas[cta] * GPU_THREADS;
            int ok = describe_sampled_word(samples[cta], word_idx);
            raw_add("    steady_gpu word[%" PRIu64 "] value=0x%08" PRIX32
                    " match=%d\n",
                    word_idx, samples[cta], ok);
            if (!ok)
                exit_code = 2;
        }
        raw_add("arm=steady_gpu t_ns=%" PRIu64 "\n", t);
    }

    free(page_words);

    printf("%s", g_raw);
    save_raw();

    CUDA_CHECK(cudaStreamDestroy(g_stream));
    CUDA_CHECK(cudaFree(managed)); /* tears down the range; the driver
                                       unregisters the backing view */
    close(backing_fd); /* the driver keeps its own file reference; the file
                          stays on disk with its offloaded content */

    if (exit_code == 0)
        fprintf(stderr, "result: all arms completed, sampled words matched\n");
    else
        fprintf(stderr, "result: ADVERSE exit=%d (see raw output; numbers "
                        "preserved in raw/)\n", exit_code);
    return exit_code;
}
