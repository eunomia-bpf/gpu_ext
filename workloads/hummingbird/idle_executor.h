// SPDX-License-Identifier: GPL-2.0
#pragma once
#include "executor.h"
#include "idle_policy.h"
#include "split_grid.h"
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <json/json.h>

namespace hummingbird {
uint64_t now_ns();
void cuda_check(CUresult result, const char *operation);
void status_check(Status result, const char *operation);
Json::Value read_json(const std::string &path);
void print_record(const char *prefix, const Json::Value &value);

struct KernelProfile {
    std::string name;
    std::array<uint32_t, 3> grid{}, block{};
    uint32_t argument_count = 0, cap = 0;
    uint64_t split_ns = 0, whole_ns = 0;
};
struct Profile {
    uint64_t launch_overhead_ns = 0, large_after_ns = 0;
    bool small_input_enabled = false, small_output_enabled = false;
    std::vector<KernelProfile> kernels;
    explicit Profile(const std::string &path);
    explicit Profile(const Json::Value &value);
};

// No guessed CPU gap makes hp_gpu_done true. The HP client records and queries
// actual compute/copy events. The short mutex serializes admissions, not work.
struct SharedState {
    std::mutex admission;
    uint32_t hp_pending = 0;
    bool hp_gpu_done = true, small_active = false;
    bool small_is_input = false;
    uint64_t last_hp_activity_ns = 0;
    CUcontext hp_context = nullptr;
    CUevent small_start = nullptr, small_end = nullptr;
    uint64_t hp_enqueues = 0, hp_completions = 0;
    uint64_t input_bubbles = 0, output_bubbles = 0, small_event_checks = 0;
    uint64_t input_bubble_ns = 0, output_bubble_ns = 0;
    uint64_t small_event_query_ns = 0;
};

class Policy {
public:
    Policy(bool bpf, const std::string &path);
    ~Policy();
    hb_output decide(hb_input input);
    uint64_t decisions = 0, jit_decisions = 0, decision_ns = 0;
private:
    void *vm_ = nullptr;
    hb_u64 (*jit_)(void *, size_t) = nullptr;
};

// Frozen Executor::clear is void and discards Model::clear's returned status.
// This local DNN-only override makes cleanup failure visible without editing it.
class CheckedExecutor : public foo::BaseExecutor {
public:
    void clear() override;
};

// Same original Model storage/parameters, transformed module only. Used by both
// profiler and scheduled LP executor. Nop DtoD copies retain their semantics.
class SplitModel : public CheckedExecutor {
public:
    explicit SplitModel(std::string cubin) : cubin_(std::move(cubin)) {}
    Status load_model(std::string path) override;
    foo::KernelInfo &kernel(size_t index) { return model->get_kernel_info(index); }
    void launch_tile(size_t index, CUstream stream, const Tile &tile);
    void launch_whole(size_t index, CUstream stream);
    Status launch_kernel(size_t index, CUstream stream = nullptr) override;
private:
    std::string cubin_;
};

class IdleExecutor : public SplitModel {
public:
    IdleExecutor(std::string cubin, const Profile &profile, SharedState &state,
                 bool bpf, const std::string &bpf_path);
    ~IdleExecutor() override;
    void start(CUcontext context, CUstream stream);
    Status execute(CUstream stream) override; // accepts at most one whole request, returns asynchronously
    void synchronize(); // waits software FIFO AND final GPU event; propagates worker errors
    void shutdown();
    Json::Value report() const;
private:
    void worker();
    void run_request();
    void verify_profile();
    const Profile &profile_;
    SharedState &state_;
    Policy policy_;
    CUcontext context_ = nullptr;
    CUstream stream_ = nullptr;
    CUevent event_ = nullptr;
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool request_ = false, closing_ = false;
    std::exception_ptr failure_;
    uint64_t requested_ = 0, completed_ = 0, split_launches_ = 0, whole_launches_ = 0;
    uint64_t small_launches_ = 0, large_launches_ = 0, ctas_ = 0, nops_ = 0;
    uint64_t input_small_launches_ = 0, output_small_launches_ = 0;
    uint64_t max_lp_inflight_ = 0, event_query_ns_ = 0;
    uint64_t event_checks_ = 0, event_waits_ = 0, event_wait_ns_ = 0, stops_ = 0;
    uint64_t tick_waits_ = 0, admission_lock_ns_ = 0;
};
} // namespace hummingbird
