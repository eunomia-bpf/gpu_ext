#pragma once
#include <stdint.h>

// Process-local measurement only; no CUDA work is performed by this accessor.
struct PodBridgeStats {
    uint64_t launches;
    uint64_t prepared_functions;
    uint64_t runtime_redirects;
    uint64_t requested_dynamic_bytes;
    uint64_t verified_dynamic_bytes;
    uint64_t static_shared_bytes;
    uint64_t device_optin_bytes;
};

extern "C" int pod_bridge_get_stats(PodBridgeStats *out, uint64_t size);
