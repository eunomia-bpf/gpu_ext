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
    uint64_t first_launches;
};

enum { POD_BRIDGE_KERNEL_NAME_BYTES = 512 };
struct PodBridgeFirstLaunch {
    uint64_t monotonic_ns;
    char kernel[POD_BRIDGE_KERNEL_NAME_BYTES];
};

extern "C" int pod_bridge_get_stats(PodBridgeStats *out, uint64_t size);
// Returns 0 for a copied record, 1 when index is past the complete set, and -1
// for an ABI mismatch. Records are process-local and ordered by kernel name.
extern "C" int pod_bridge_get_first_launch(PodBridgeFirstLaunch *out,
                                             uint64_t size, uint64_t index);
