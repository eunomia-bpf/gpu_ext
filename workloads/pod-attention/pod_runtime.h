#pragma once
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "launch_abi.h"

struct PodWorkspace {
    at::Tensor counters;
    at::Tensor contexts;
    at::Tensor errors;
    at::Tensor metadata;
    PodLaunchView view;
};

void pod_configure(const std::string &mode, bool trace);
PodWorkspace pod_workspace(unsigned grid, unsigned prefill_blocks,
                           unsigned decode_blocks, unsigned factor_p,
                           unsigned factor_d, unsigned smem_bytes,
                           unsigned threads, unsigned fused_op,
                           cudaStream_t stream);
std::vector<at::Tensor> pod_last_launch();
