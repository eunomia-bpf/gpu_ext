// SPDX-License-Identifier: GPL-2.0
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace hummingbird {
// Bookkeeping for actual completion events, NOT a hardware device-queue gauge.
// The executor records a slot before issued(), and retires only successful
// CUDA queries. A fixed capacity bounds underprediction without skipping policy.
class PipelineEvents {
public:
    explicit PipelineEvents(uint32_t bound) : bound_(bound) {
        if (bound != 1 && bound != 2) throw std::runtime_error("LP bound must be 1 or 2");
    }
    uint32_t bound() const { return bound_; }
    uint32_t outstanding() const { return count_; }
    uint32_t next_slot() const {
        if (count_ == bound_) throw std::runtime_error("policy exceeded LP event bound");
        return (head_ + count_) % bound_;
    }
    void issued() {
        next_slot();
        if (count_) ++overlap_launches;
        ++count_; ++issued_count; peak = std::max(peak, count_);
    }
    template<class Query> bool retire_completed(Query query) {
        while (count_) {
            if (!query(head_)) return false;
            head_ = (head_ + 1) % bound_; --count_; ++retired_count;
        }
        return true;
    }
    uint64_t issued_count = 0, retired_count = 0, overlap_launches = 0;
    uint32_t peak = 0;
private:
    uint32_t bound_, head_ = 0, count_ = 0;
};
} // namespace hummingbird
