#pragma once
#include "load_study_measurement.h"
#include <vector>

namespace hummingbird {
// Extends the existing FIFO clock/accounting; trace rows are never skipped.
class Schedule {
public:
    Schedule(int64_t begin, int64_t end, int64_t interval,
             const std::vector<int64_t>& offsets)
        : fallback_(begin, end, interval), begin_(begin), end_(end),
          interval_(interval), offsets_(offsets) {
        if (interval && !offsets.empty()) throw std::invalid_argument("two arrival modes");
        int64_t previous = -1;
        for (auto offset : offsets) {
            if (offset < 0 || offset < previous || offset >= end - begin)
                throw std::invalid_argument("trace offsets must be sorted inside window");
            previous = offset;
        }
    }
    bool finite() const { return interval_ || !offsets_.empty(); }
    int64_t next_ns() const {
        if (offsets_.empty()) return fallback_.next_ns();
        return started_ < offsets_.size() ? begin_ + offsets_[started_] : end_;
    }
    bool admit(int64_t now, gpreempt_load_study::Request& request) {
        if (offsets_.empty()) return fallback_.admit(now, request);
        if (started_ >= offsets_.size() || now < next_ns() || now >= end_) return false;
        request = {started_, next_ns(), now, 0};
        ++started_;
        return true;
    }
    uint64_t started() const { return offsets_.empty() ? fallback_.started() : started_; }
    uint64_t offered() const { return offsets_.empty() ? fallback_.offered() : offsets_.size(); }
private:
    gpreempt_load_study::Schedule fallback_;
    int64_t begin_, end_, interval_;
    const std::vector<int64_t>& offsets_;
    uint64_t started_ = 0;
};
}
