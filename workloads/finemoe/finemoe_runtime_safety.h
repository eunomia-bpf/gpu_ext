// SPDX-License-Identifier: Apache-2.0
// Shared real executor constraints, not policy decisions.
#pragma once
#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <utility>

namespace finemoe_revision {
inline std::int64_t SparseBudget(std::int64_t capacity, std::int64_t dense,
                                 std::int64_t configured, std::int64_t incoming_dense = 0) {
    const auto available = capacity - dense - incoming_dense;
    return configured > 0 ? std::min(available, configured) : available;
}

template <class State, class Transfer>
void CompleteDemand(std::mutex &mutex, std::condition_variable &cv, State &state, Transfer &&transfer) {
    // The acquire thread releases this same mutex atomically when entering its
    // condition-variable wait. Atomic state alone cannot prevent lost wakeups.
    std::lock_guard<std::mutex> lock(mutex);
    std::forward<Transfer>(transfer)();
    state = 0;
    cv.notify_all();
}
}  // namespace finemoe_revision
