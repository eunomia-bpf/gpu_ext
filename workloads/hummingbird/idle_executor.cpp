// SPDX-License-Identifier: GPL-2.0
#include "idle_executor.h"
#include "ebpf-vm.h"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>

namespace hummingbird {
uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
void cuda_check(CUresult result, const char *operation) {
    if (result == CUDA_SUCCESS) return;
    const char *name = nullptr, *detail = nullptr;
    cuGetErrorName(result, &name); cuGetErrorString(result, &detail);
    throw std::runtime_error(std::string(operation) + ": " + (name ? name : "CUDA error") +
                             " " + (detail ? detail : ""));
}
void status_check(Status result, const char *operation) {
    if (result != Status::Succ) throw std::runtime_error(std::string(operation) + " failed");
}
Json::Value read_json(const std::string &path) {
    std::ifstream source(path);
    Json::Value value; Json::CharReaderBuilder builder; std::string error;
    if (!source || !Json::parseFromStream(builder, source, &value, &error))
        throw std::runtime_error("cannot parse " + path + ": " + error);
    return value;
}
void print_record(const char *prefix, const Json::Value &value) {
    Json::StreamWriterBuilder writer; writer["indentation"] = "";
    static std::mutex output_mutex;
    std::lock_guard<std::mutex> lock(output_mutex);
    std::fprintf(stderr, "%s %s\n", prefix, Json::writeString(writer, value).c_str());
}
static uint64_t positive(const Json::Value &value, const char *name) {
    if (!value.isUInt64() || !value.asUInt64())
        throw std::runtime_error(std::string("profile requires positive integer ") + name);
    return value.asUInt64();
}
Profile::Profile(const std::string &path) : Profile(read_json(path)) {}
Profile::Profile(const Json::Value &value) {
    if (!value["schema_version"].isInt() || value["schema_version"] != 1 || value["model"] != "resnet152" ||
        value["gpu_correctness_validated"] != true || !value["kernels"].isArray())
        throw std::runtime_error("profile schema/model/correctness validation mismatch");
    launch_overhead_ns = positive(value["launch_overhead_ns"], "launch_overhead_ns");
    large_after_ns = positive(value["large_after_ns"], "large_after_ns");
    if (!value["small_input_enabled"].isBool() || !value["small_output_enabled"].isBool())
        throw std::runtime_error("profile lacks explicit small-pattern eligibility");
    small_input_enabled = value["small_input_enabled"].asBool();
    small_output_enabled = value["small_output_enabled"].asBool();
    for (const auto &entry : value["kernels"]) {
        KernelProfile k;
        if (!entry["index"].isUInt64() || entry["index"].asUInt64() != kernels.size() ||
            !entry["name"].isString() || entry["name"].asString().empty() ||
            !entry["argument_count"].isUInt() || !entry["argument_count"].asUInt() ||
            !entry["cap"].isUInt() || !entry["cap"].asUInt() ||
            entry["grid"].size() != 3 || entry["block"].size() != 3)
            throw std::runtime_error("malformed indexed kernel profile");
        k.name = entry["name"].asString();
        k.argument_count = entry["argument_count"].asUInt();
        k.cap = entry["cap"].asUInt();
        if (k.name != "nop" && !k.cap) throw std::runtime_error("zero split capacity");
        k.split_ns = positive(entry["split_ns"], "split_ns");
        k.whole_ns = positive(entry["whole_ns"], "whole_ns");
        for (unsigned int axis = 0; axis < 3; ++axis) {
            if (!entry["grid"][axis].isUInt() || !entry["block"][axis].isUInt())
                throw std::runtime_error("invalid kernel dimensions");
            k.grid[axis] = entry["grid"][axis].asUInt();
            k.block[axis] = entry["block"][axis].asUInt();
            if (!k.grid[axis] || !k.block[axis])
                throw std::runtime_error("zero kernel dimension");
        }
        uint64_t total = 1, threads = 1;
        for (unsigned int axis = 0; axis < 3; ++axis) {
            if (k.grid[axis] > UINT32_MAX / total || k.block[axis] > 1024 / threads)
                throw std::runtime_error("unsupported grid or block volume");
            total *= k.grid[axis]; threads *= k.block[axis];
        }
        if (k.cap > total || k.argument_count > 64 ||
            (k.name == "nop" && (total != 1 || threads != 1 || k.cap != 1 || k.argument_count != 2)))
            throw std::runtime_error("profile cap/argument bounds violated");
        kernels.push_back(k);
    }
    if (kernels.empty()) throw std::runtime_error("empty kernel profile");
}
Policy::Policy(bool bpf, const std::string &path) {
    if (!bpf) return;
    std::ifstream source(path, std::ios::binary);
    std::vector<char> code{std::istreambuf_iterator<char>(source), {}};
    if (!source.is_open() || code.empty() || code.size() > 65536)
        throw std::runtime_error("invalid BPF program");
    auto *vm = ebpf_create("ubpf");
    if (!vm) throw std::runtime_error("cannot create ubpf VM");
    char *error = nullptr;
    if (ebpf_load(vm, code.data(), code.size(), &error)) {
        std::string message = error ? error : "BPF load failed";
        std::free(error); ebpf_destroy(vm); throw std::runtime_error(message);
    }
    auto fn = ebpf_compile(vm, &error);
    if (!fn) {
        std::string message = error ? error : "BPF JIT failed";
        std::free(error); ebpf_destroy(vm); throw std::runtime_error(message);
    }
    vm_ = vm; jit_ = reinterpret_cast<decltype(jit_)>(fn);
}
Policy::~Policy() { if (vm_) ebpf_destroy(static_cast<ebpf_vm *>(vm_)); }
hb_output Policy::decide(hb_input input) {
    hb_call call{}; call.input = input;
    const auto begin = now_ns();
    auto action = jit_ ? jit_(&call, sizeof(call)) : hb_decide(&call, sizeof(call));
    decision_ns += now_ns() - begin; ++decisions; if (jit_) ++jit_decisions;
    if (action != call.output.action || action < HB_STOP_LP || action > HB_WHOLE)
        throw std::runtime_error("invalid idle-policy action; no fallback");
    return call.output;
}
void CheckedExecutor::clear() {
    if (model) { status_check(model->clear(), "free model resources"); model.reset(); }
}
Status SplitModel::load_model(std::string path) {
    model = std::make_shared<foo::Model>();
    return model->load_model(path + "/mod.json", path + "/host.json", cubin_);
}
void SplitModel::launch_tile(size_t index, CUstream stream, const Tile &tile) {
    auto &k = kernel(index);
    if (k.name == "nop") { status_check(foo::Executor::launch_kernel(index, stream), "nop copy"); return; }
    uint32_t x = tile.offset[0], y = tile.offset[1], z = tile.offset[2];
    std::vector<void *> arguments(k.args_ptr.begin(), k.args_ptr.end());
    arguments.push_back(&x); arguments.push_back(&y); arguments.push_back(&z);
    cuda_check(cuLaunchKernel(k.handler, tile.grid[0], tile.grid[1], tile.grid[2],
        k.launch_params[3], k.launch_params[4], k.launch_params[5], 0, stream,
        arguments.data(), nullptr), "cuLaunchKernel offset piece");
}
void SplitModel::launch_whole(size_t index, CUstream stream) {
    auto &k = kernel(index);
    if (k.name == "nop") { status_check(foo::Executor::launch_kernel(index, stream), "nop copy"); return; }
    launch_tile(index, stream, {{0, 0, 0}, {k.launch_params[0], k.launch_params[1], k.launch_params[2]}});
}
Status SplitModel::launch_kernel(size_t index, CUstream stream) { launch_whole(index, stream); return Status::Succ; }

IdleExecutor::IdleExecutor(std::string cubin, const Profile &profile, SharedState &state,
                           bool bpf, const std::string &path)
    : SplitModel(std::move(cubin)), profile_(profile), state_(state), policy_(bpf, path) {}
IdleExecutor::~IdleExecutor() {
    try { shutdown(); } catch (const std::exception &e) {
        std::fprintf(stderr, "HUMMINGBIRD_FATAL worker cleanup: %s\n", e.what()); std::terminate();
    }
}
void IdleExecutor::verify_profile() {
    if (profile_.kernels.size() != get_kernel_num()) throw std::runtime_error("profile kernel count changed");
    for (size_t i = 0; i < get_kernel_num(); ++i) {
        auto &actual = kernel(i); const auto &p = profile_.kernels[i];
        if (actual.name != p.name || actual.args_ptr.size() != p.argument_count)
            throw std::runtime_error("profile kernel identity/arguments changed");
        if (actual.name == "nop") continue;
        for (unsigned int a = 0; a < 3; ++a)
            if (actual.launch_params[a] != p.grid[a] || actual.launch_params[a + 3] != p.block[a])
                throw std::runtime_error("profile launch dimensions changed");
    }
}
void IdleExecutor::start(CUcontext context, CUstream stream) {
    if (worker_.joinable() || !context || !stream) throw std::runtime_error("invalid idle worker start");
    verify_profile(); context_ = context; stream_ = stream;
    cuda_check(cuEventCreate(&event_, CU_EVENT_DISABLE_TIMING), "create LP completion event");
    worker_ = std::thread(&IdleExecutor::worker, this);
}
Status IdleExecutor::execute(CUstream stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (failure_) std::rethrow_exception(failure_);
    if (!worker_.joinable() || closing_ || request_ || stream != stream_)
        throw std::runtime_error("LP FIFO full, closed, or wrong stream");
    request_ = true; ++requested_; condition_.notify_all(); return Status::Succ;
}
void IdleExecutor::synchronize() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!condition_.wait_for(lock, std::chrono::seconds(120), [&] { return !request_ || failure_; }))
        throw std::runtime_error("LP request exceeded 120 s bounded drain");
    if (failure_) std::rethrow_exception(failure_);
}
void IdleExecutor::shutdown() {
    if (!worker_.joinable()) return;
    synchronize();
    { std::lock_guard<std::mutex> lock(mutex_); closing_ = true; condition_.notify_all(); }
    worker_.join();
    cuda_check(cuCtxSetCurrent(context_), "LP cleanup context");
    cuda_check(cuEventDestroy(event_), "destroy LP completion event"); event_ = nullptr;
}
void IdleExecutor::worker() {
    try {
        bind_core(2);
        cuda_check(cuCtxSetCurrent(context_), "LP worker context");
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [&] { return request_ || closing_; });
            if (closing_ && !request_) break;
            lock.unlock(); run_request(); lock.lock();
            request_ = false; ++completed_; condition_.notify_all();
        }
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_); failure_ = std::current_exception(); condition_.notify_all();
    }
}
void IdleExecutor::run_request() {
    const auto deadline = now_ns() + 120000000000ULL;
    uint64_t tick_due = 0;
    bool in_flight = false;
    auto done = [&] {
        if (!in_flight) return true;
        ++event_checks_; const auto begin = now_ns(); const auto result = cuEventQuery(event_);
        event_query_ns_ += now_ns() - begin;
        if (result == CUDA_ERROR_NOT_READY) return false;
        cuda_check(result, "query LP completion"); in_flight = false; return true;
    };
    for (size_t i = 0; i < get_kernel_num(); ++i) {
        const auto &p = profile_.kernels[i];
        GridCursor cursor(p.name == "nop" ? std::array<uint32_t, 3>{1, 1, 1} : p.grid,
                          p.name == "nop" ? 1 : p.cap);
        while (!cursor.done()) {
            if (now_ns() > deadline) throw std::runtime_error("LP execution deadline exceeded");
            const bool gpu_done = done();
            const auto wait_begin = now_ns();
            std::unique_lock<std::mutex> lock(state_.admission);
            admission_lock_ns_ += now_ns() - wait_begin;
            hb_input input{};
            input.now_ns = now_ns(); input.last_hp_activity_ns = state_.last_hp_activity_ns;
            input.large_after_ns = profile_.large_after_ns; input.tick_due_ns = tick_due;
            input.launch_overhead_ns = profile_.launch_overhead_ns;
            input.split_ns = p.split_ns; input.whole_ns = p.whole_ns;
            input.hp_pending = state_.hp_pending; input.hp_gpu_done = state_.hp_gpu_done;
            input.lp_pending = 1; input.lp_gpu_done = gpu_done;
            input.kernel_unstarted = cursor.unstarted(); input.consolidate = 1;
            if (state_.small_active && state_.hp_context && !state_.hp_pending) {
                const auto query_begin = now_ns();
                cuda_check(cuCtxPushCurrent(state_.hp_context), "enter HP event context");
                auto start_status = cuEventQuery(state_.small_start);
                auto end_status = cuEventQuery(state_.small_end);
                CUcontext popped = nullptr;
                cuda_check(cuCtxPopCurrent(&popped), "restore LP event context");
                state_.small_event_checks += 2;
                state_.small_event_query_ns += now_ns() - query_begin;
                if (start_status != CUDA_ERROR_NOT_READY) cuda_check(start_status, "query small start");
                if (end_status != CUDA_ERROR_NOT_READY) cuda_check(end_status, "query small end");
                if (end_status == CUDA_SUCCESS) state_.small_active = false;
                input.small_active = state_.small_active;
                input.small_start_done = start_status == CUDA_SUCCESS;
            }
            const auto output = policy_.decide(input);
            if (output.action == HB_SPLIT || output.action == HB_WHOLE) {
                // This lock also guards HP pending publication. No HP enqueue can
                // race between policy observation and this single CUDA submission.
                Tile tile = output.action == HB_WHOLE ? Tile{{0, 0, 0}, p.grid} : cursor.current();
                if (p.name == "nop") { launch_whole(i, stream_); ++nops_; }
                else {
                    launch_tile(i, stream_, tile);
                    ctas_ += uint64_t(tile.grid[0]) * tile.grid[1] * tile.grid[2];
                }
                cuda_check(cuEventRecord(event_, stream_), "record LP completion");
                in_flight = true; max_lp_inflight_ = 1; tick_due = output.next_tick_ns;
                if (output.action == HB_WHOLE) ++whole_launches_; else ++split_launches_;
                if (output.bubble == HB_SMALL_BUBBLE) {
                    ++small_launches_;
                    if (state_.small_is_input) ++input_small_launches_; else ++output_small_launches_;
                } else ++large_launches_;
                lock.unlock();
                if (output.action == HB_WHOLE || p.name == "nop") break;
                cursor.advance();
            } else {
                if (output.action == HB_STOP_LP) ++stops_;
                if (output.wait_reason == HB_WAIT_TICK) ++tick_waits_;
                lock.unlock();
                const auto pause = now_ns();
                std::this_thread::yield();
                if (!gpu_done) { ++event_waits_; event_wait_ns_ += now_ns() - pause; }
            }
        }
    }
    while (!done()) {
        if (now_ns() > deadline) throw std::runtime_error("final LP completion deadline exceeded");
        const auto begin = now_ns(); std::this_thread::yield();
        ++event_waits_; event_wait_ns_ += now_ns() - begin;
    }
}
Json::Value IdleExecutor::report() const {
    Json::Value result;
    result["requests_accepted"] = Json::UInt64(requested_);
    result["requests_completed"] = Json::UInt64(completed_);
    result["split_launches"] = Json::UInt64(split_launches_);
    result["whole_launches"] = Json::UInt64(whole_launches_);
    result["small_launches"] = Json::UInt64(small_launches_);
    result["input_small_launches"] = Json::UInt64(input_small_launches_);
    result["output_small_launches"] = Json::UInt64(output_small_launches_);
    result["large_launches"] = Json::UInt64(large_launches_);
    result["ctas_submitted"] = Json::UInt64(ctas_); result["nop_copies"] = Json::UInt64(nops_);
    result["lp_event_checks"] = Json::UInt64(event_checks_); result["lp_event_waits"] = Json::UInt64(event_waits_);
    result["lp_event_yield_ns"] = Json::UInt64(event_wait_ns_); result["hp_stop_decisions"] = Json::UInt64(stops_);
    result["lp_event_query_ns"] = Json::UInt64(event_query_ns_);
    result["tick_waits"] = Json::UInt64(tick_waits_); result["admission_lock_wait_ns"] = Json::UInt64(admission_lock_ns_);
    result["decisions"] = Json::UInt64(policy_.decisions); result["jit_decisions"] = Json::UInt64(policy_.jit_decisions);
    result["decision_ns"] = Json::UInt64(policy_.decision_ns);
    result["max_lp_inflight"] = Json::UInt64(max_lp_inflight_); result["configured_lp_inflight_bound"] = 1;
    result["completion_fence"] = "event-query-before-next-launch";
    return result;
}
} // namespace hummingbird
