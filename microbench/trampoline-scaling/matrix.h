#ifndef GPUBPF_TRAMPOLINE_SCALING_MATRIX_H
#define GPUBPF_TRAMPOLINE_SCALING_MATRIX_H

#define SCALE_THREADS_PER_BLOCK 256U
#define SCALE_MAX_THREADS 1048576U
#define SCALE_COUNTER_KEYS 5U
#define SCALE_CELL_COUNT 9U

/* id, blocks, active_threads, counter_key */
#define SCALE_CELL_LIST(X) \
    X(0, 256, 65536, 0) \
    X(1, 512, 65536, 1) \
    X(2, 1024, 65536, 2) \
    X(3, 2048, 65536, 3) \
    X(4, 4096, 65536, 4) \
    X(5, 4096, 131072, 4) \
    X(6, 4096, 262144, 4) \
    X(7, 4096, 524288, 4) \
    X(8, 4096, 1048576, 4)

#endif

