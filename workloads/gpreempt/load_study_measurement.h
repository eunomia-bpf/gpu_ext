#pragma once

// Independent load-study instrumentation; no policy, CUDA, or JSON dependency.
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>

namespace gpreempt_load_study {
using Clock = std::chrono::steady_clock;
inline int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count();
}
inline Clock::time_point at_ns(int64_t ns) {
    return Clock::time_point(std::chrono::nanoseconds(ns));
}

struct Window {
    int64_t begin_ns = 0, end_ns = 0;
    std::chrono::system_clock::time_point wall_begin, wall_end;
};

// Publish one epoch only after both CUDA workers have finished initialization.
class CommonStart {
public:
    CommonStart(unsigned participants, int64_t duration_ns, int64_t delay_ns)
        : remaining_(participants), duration_(duration_ns), delay_(delay_ns) {
        if (!participants || duration_ns <= 0 || delay_ns < 0)
            throw std::invalid_argument("invalid common window");
    }
    Window wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (--remaining_ == 0) {
            window_.begin_ns = now_ns() + delay_;
            window_.end_ns = window_.begin_ns + duration_;
            window_.wall_begin = std::chrono::system_clock::now()
                + std::chrono::nanoseconds(delay_);
            window_.wall_end = window_.wall_begin + std::chrono::nanoseconds(duration_);
            ready_ = true;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [this] { return ready_; });
        }
        return window_;
    }
    Window window() const { return window_; } // Read only after worker joins.
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    unsigned remaining_;
    int64_t duration_, delay_;
    bool ready_ = false;
    Window window_;
};

struct Request {
    uint64_t id;
    int64_t scheduled_ns, started_ns, verified_ready_ns;
};

// A lazy FIFO of periodic arrivals, not a newest-only or finite-capacity queue.
// interval=0 means a closed-loop continuous source, with no offered denominator.
class Schedule {
public:
    Schedule(int64_t begin, int64_t end, int64_t interval)
        : begin_(begin), end_(end), interval_(interval) {
        if (end <= begin || interval < 0)
            throw std::invalid_argument("invalid arrival schedule");
    }
    int64_t next_ns() const { return interval_ ? begin_ + started_ * interval_ : begin_; }
    bool admit(int64_t now, Request &request) {
        if (now < next_ns() || now >= end_ || next_ns() >= end_) return false;
        request = {started_, interval_ ? next_ns() : now, now, 0};
        ++started_;
        return true;
    }
    uint64_t started() const { return started_; }
    uint64_t offered() const {
        if (!interval_) throw std::logic_error("continuous has no offered slots");
        return (end_ - begin_ + interval_ - 1) / interval_;
    }
private:
    int64_t begin_, end_, interval_;
    uint64_t started_ = 0;
};
} // namespace gpreempt_load_study

// Error checking only: policy actions are identical; legacy builds retain their path.
#ifdef GPREEMPT_LOAD_STUDY
#define GPREEMPT_STUDY_CUDA(call) ASSERT_CUDA_ERROR(call)
#define GPREEMPT_STUDY_STATUS(call) CHECK_STATUS(call)
#else
#define GPREEMPT_STUDY_CUDA(call) (call)
#define GPREEMPT_STUDY_STATUS(call) (call)
#endif
