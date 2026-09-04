#ifndef GPUBPF_TRAMPOLINE_SCALING_MATRIX_H
#define GPUBPF_TRAMPOLINE_SCALING_MATRIX_H

#define SCALE_MAX_THREADS 1048576U
#define SCALE_MAX_THREADS_PER_BLOCK 256U
#define SCALE_COUNTER_KEYS 5U
#define SCALE_CELL_COUNT 9U

/* id, blocks, threads_per_block, active_threads, counter_key */
#define SCALE_CELL_LIST(X) \
    X(0, 256, 256, 65536, 0) \
    X(1, 512, 256, 65536, 1) \
    X(2, 1024, 256, 65536, 2) \
    X(3, 2048, 256, 65536, 3) \
    X(4, 4096, 256, 65536, 4) \
    X(5, 4096, 256, 131072, 4) \
    X(6, 4096, 256, 262144, 4) \
    X(7, 4096, 256, 524288, 4) \
    X(8, 4096, 256, 1048576, 4)

/* blocks, threads_per_block, counter_key */
#define SCALE_COUNTER_GEOMETRY_LIST(X) \
    X(256, 256, 0) \
    X(512, 256, 1) \
    X(1024, 256, 2) \
    X(2048, 256, 3) \
    X(4096, 256, 4)

#endif
