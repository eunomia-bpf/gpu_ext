#pragma once

#include <stdint.h>

extern "C" __device__ void bpf_exit(uint64_t ctx, uint64_t ctx_len);
