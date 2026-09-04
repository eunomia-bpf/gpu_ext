#ifndef GPUBPF_TRAMPOLINE_FIXED_WORK_MATRIX_H
#define GPUBPF_TRAMPOLINE_FIXED_WORK_MATRIX_H

#define SCALE_MAX_THREADS 1048576U
#define SCALE_MAX_THREADS_PER_BLOCK 1024U
#define SCALE_COUNTER_KEYS 5U
#define SCALE_CELL_COUNT 5U

/* id, blocks, threads_per_block, active_threads, counter_key */
#define SCALE_CELL_LIST(X) \
    X(0, 128, 1024, 131072, 0) \
    X(1, 256, 512, 131072, 1) \
    X(2, 1024, 128, 131072, 2) \
    X(3, 2048, 64, 131072, 3) \
    X(4, 4096, 32, 131072, 4)

/* blocks, threads_per_block, counter_key */
#define SCALE_COUNTER_GEOMETRY_LIST(X) \
    X(128, 1024, 0) \
    X(256, 512, 1) \
    X(1024, 128, 2) \
    X(2048, 64, 3) \
    X(4096, 32, 4)

#endif
