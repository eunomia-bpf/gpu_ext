// SPDX-License-Identifier: GPL-2.0
// Real-GPU calibration only. This executable never runs during timed cells.
#include "idle_executor.h"
#include "client_checks.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <memory>
#include <unistd.h>

using namespace hummingbird;
namespace {
constexpr unsigned int samples = 3, max_halvings = 12, copy_samples = 100;
constexpr double stable_fraction = 0.01;
uint64_t median(std::vector<uint64_t> values) {
    if (values.empty()) throw std::runtime_error("empty calibration sample");
    std::sort(values.begin(), values.end()); return values[values.size() / 2];
}
Json::Value numbers(const std::vector<uint64_t> &values) {
    Json::Value result(Json::arrayValue); for (auto v : values) result.append(Json::UInt64(v)); return result;
}
uint64_t blocks(const foo::KernelInfo &kernel) {
    if (kernel.name == "nop") return 1;
    const auto *p = kernel.launch_params;
    return uint64_t(p[0]) * p[1] * p[2];
}
std::array<uint32_t, 3> grid(const foo::KernelInfo &kernel) {
    return kernel.name == "nop" ? std::array<uint32_t, 3>{1, 1, 1} :
        std::array<uint32_t, 3>{kernel.launch_params[0], kernel.launch_params[1], kernel.launch_params[2]};
}
struct ModelRun {
    std::shared_ptr<foo::BaseExecutor> executor;
    CUstream stream = nullptr;
    void *input = nullptr, *output = nullptr;
    size_t input_size = 0, output_size = 0;
    gpreempt_artifact::OutputCheck check;
    ModelRun(std::string name, std::shared_ptr<foo::BaseExecutor> model) : executor(std::move(model)) {
        status_check(executor->init(name), "profile load real model");
        input_size = executor->get_data_size("data"); output_size = executor->get_data_size("heads");
        cuda_check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create profile stream");
        cuda_check(cuMemHostAlloc(&input, input_size, CU_MEMHOSTALLOC_PORTABLE), "profile pinned input");
        cuda_check(cuMemHostAlloc(&output, output_size, CU_MEMHOSTALLOC_PORTABLE), "profile pinned output");
        check.initialize("profile_" + name, std::string(MODEL_PATH) + "/" + name + "/reference.f32",
                         input, input_size, output_size);
    }
    void begin() {
        status_check(executor->set_input("data", input, input_size, stream), "profile upload input");
        cuda_check(cuStreamSynchronize(stream), "profile input synchronization");
    }
    void verify() {
        status_check(executor->get_output("heads", output, output_size, stream), "profile read full output");
        cuda_check(cuStreamSynchronize(stream), "profile output synchronization");
        check.check(output, output_size);
    }
    void full() { begin(); status_check(executor->execute(stream), "profile unsplit model"); verify(); }
    ~ModelRun() {
        executor->clear(); executor.reset();
        // Destruction errors are fatal: a partial calibration must not publish.
        try {
            cuda_check(cuMemFreeHost(input), "profile free input");
            cuda_check(cuMemFreeHost(output), "profile free output");
            cuda_check(cuStreamDestroy(stream), "profile destroy stream");
        } catch (const std::exception &e) { std::fprintf(stderr, "HUMMINGBIRD_FATAL %s\n", e.what()); std::terminate(); }
    }
};
struct Events {
    CUevent start = nullptr, end = nullptr;
    Events() {
        cuda_check(cuEventCreate(&start, 0), "create timed start event");
        cuda_check(cuEventCreate(&end, 0), "create timed end event");
    }
    uint64_t finish(CUstream stream) {
        cuda_check(cuEventRecord(end, stream), "record timed end");
        cuda_check(cuEventSynchronize(end), "synchronize timed operation");
        float milliseconds = 0;
        cuda_check(cuEventElapsedTime(&milliseconds, start, end), "measure actual operation");
        if (!std::isfinite(milliseconds) || milliseconds < 0)
            throw std::runtime_error("nonfinite/negative GPU duration");
        return std::max<uint64_t>(1, std::llround(double(milliseconds) * 1000000));
    }
    ~Events() {
        try { cuda_check(cuEventDestroy(start), "destroy timed start"); cuda_check(cuEventDestroy(end), "destroy timed end"); }
        catch (...) { std::terminate(); }
    }
};
struct Candidate {
    uint32_t cap = 1;
    std::vector<uint64_t> maximum_piece_ns, sum_piece_ns, piece_count;
    uint64_t conservative_ns() const { return *std::max_element(maximum_piece_ns.begin(), maximum_piece_ns.end()); }
    Json::Value json() const {
        Json::Value v; v["cap"] = cap; v["maximum_piece_ns"] = numbers(maximum_piece_ns);
        v["sum_piece_ns"] = numbers(sum_piece_ns); v["piece_count"] = numbers(piece_count);
        v["selection_median_ns"] = Json::UInt64(median(maximum_piece_ns)); return v;
    }
};
void measure_pass(ModelRun &run, SplitModel &model, std::vector<Candidate> &candidates,
                  Events &events, std::vector<uint64_t> &launch_costs) {
    for (unsigned int sample = 0; sample < samples; ++sample) {
        run.begin();
        for (size_t i = 0; i < model.get_kernel_num(); ++i) {
            auto &kernel = model.kernel(i); auto &candidate = candidates[i];
            GridCursor cursor(grid(kernel), candidate.cap);
            uint64_t maximum = 0, sum = 0, pieces = 0;
            while (!cursor.done()) {
                cuda_check(cuEventRecord(events.start, run.stream), "record timed piece start");
                const auto begin = now_ns();
                model.launch_tile(i, run.stream, cursor.current());
                const auto host_duration = std::max<uint64_t>(1, now_ns() - begin);
                if (kernel.name != "nop") launch_costs.push_back(host_duration);
                const auto duration = events.finish(run.stream);
                maximum = std::max(maximum, duration); sum += duration; ++pieces; cursor.advance();
            }
            candidate.maximum_piece_ns.push_back(maximum); candidate.sum_piece_ns.push_back(sum);
            candidate.piece_count.push_back(pieces);
        }
        // Every full pass runs every original CTA exactly once and checks all
        // outputs. Repeating one isolated/in-place kernel could corrupt state.
        run.verify();
    }
}
Json::Value copy_profile(Events &events) {
    ModelRun hp("vgg", std::make_shared<CheckedExecutor>());
    hp.full(); hp.full();
    std::vector<uint64_t> input_gpu, output_gpu, input_host, output_host;
    for (unsigned int sample = 0; sample < copy_samples; ++sample) {
        const auto input_begin = now_ns();
        cuda_check(cuEventRecord(events.start, hp.stream), "copy-profile input start");
        status_check(hp.executor->set_input("data", hp.input, hp.input_size, hp.stream), "copy-profile input");
        input_gpu.push_back(events.finish(hp.stream)); input_host.push_back(now_ns() - input_begin);
        status_check(hp.executor->execute(hp.stream), "copy-profile foreground inference");
        cuda_check(cuStreamSynchronize(hp.stream), "copy-profile foreground complete");
        const auto output_begin = now_ns();
        cuda_check(cuEventRecord(events.start, hp.stream), "copy-profile output start");
        status_check(hp.executor->get_output("heads", hp.output, hp.output_size, hp.stream), "copy-profile output");
        output_gpu.push_back(events.finish(hp.stream)); output_host.push_back(now_ns() - output_begin);
        hp.check.check(hp.output, hp.output_size);
    }
    Json::Value result; result["samples"] = copy_samples;
    result["input_gpu_ns"] = numbers(input_gpu); result["input_api_sync_ns"] = numbers(input_host);
    result["output_gpu_ns"] = numbers(output_gpu); result["output_api_sync_ns"] = numbers(output_host);
    result["isolated_correctness_checked"] = Json::UInt64(hp.check.count());
    result["interference_eligibility_measured"] = false;
    result["small_patterns_enabled"] = false;
    return result;
}
void write_new(const std::string &path, const Json::Value &result) {
    Json::StreamWriterBuilder builder; builder["indentation"] = "  ";
    std::string data = Json::writeString(builder, result) + "\n";
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) throw std::runtime_error("cannot create fresh profile output: " + path);
    size_t offset = 0;
    while (offset < data.size()) {
        auto n = write(fd, data.data() + offset, data.size() - offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { close(fd); throw std::runtime_error("profile write failed"); }
        offset += static_cast<size_t>(n);
    }
    if (fsync(fd) || close(fd)) throw std::runtime_error("profile durable close failed");
}
}

int main(int argc, char **argv) {
    try {
        if (argc == 2 && !std::strcmp(argv[1], "--help")) {
            std::cout << "hummingbird_profile --split-cubin SPLIT.cubin --output NEW_PROFILE.json\n"
                         "Runs actual GPU kernels: occupancy start, capacity halving, 3 full validated passes per candidate, "
                         "1% stability rule, maximum 12 halvings. Small patterns remain disabled until independent interference calibration.\n";
            return 0;
        }
        std::string cubin, output;
        if (argc != 5) throw std::runtime_error("invalid profile command; use --help");
        for (int i = 1; i < argc; i += 2) {
            const std::string option = argv[i];
            auto *target = option == "--split-cubin" ? &cubin : option == "--output" ? &output : nullptr;
            if (!target || !target->empty()) throw std::runtime_error("unknown/repeated profile option");
            *target = argv[i + 1];
        }
        if (cubin.empty() || output.empty() || access(output.c_str(), F_OK) == 0)
            throw std::runtime_error("missing input or output already exists");
        foo::util::init_cuda();
        CUdevice device; cuda_check(cuDeviceGet(&device, 0), "profile device");
        int sm_count = 0, major = 0, minor = 0;
        cuda_check(cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device), "profile SM count");
        cuda_check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "profile compute major");
        cuda_check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "profile compute minor");
        auto executor = std::make_shared<SplitModel>(cubin);
        Json::Value result; result["schema_version"] = 1; result["model"] = "resnet152";
        result["split_cubin"] = cubin; result["context"] = "isolated-primary";
        result["sm_count"] = sm_count; result["compute_major"] = major; result["compute_minor"] = minor;
        result["samples_per_candidate"] = samples; result["max_halvings"] = max_halvings;
        result["stable_fraction"] = stable_fraction;
        result["selection"] = "halve occupancy-resident cap while median per-request maximum piece duration improves by >=1%; retain previous at first plateau";
        result["tick_estimate"] = "maximum measured selected-piece duration over three complete model passes";
        result["gpu_correctness_validated"] = false;
        Events events;
        {
            ModelRun run("resnet152", executor); run.full(); run.full();
            std::vector<Candidate> full, selected;
            std::vector<int> active_blocks;
            std::vector<uint64_t> launch_costs;
            std::vector<Json::Value> history;
            for (size_t i = 0; i < executor->get_kernel_num(); ++i) {
                auto &kernel = executor->kernel(i); const auto total = blocks(kernel);
                if (!total || total > std::numeric_limits<uint32_t>::max())
                    throw std::runtime_error("unsupported full grid size");
                int active = 0;
                if (kernel.name != "nop") {
                    const auto *p = kernel.launch_params;
                    cuda_check(cuOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel.handler,
                        p[3] * p[4] * p[5], 0), "profile exact-function occupancy");
                    if (active <= 0) throw std::runtime_error("zero occupancy for real function");
                }
                active_blocks.push_back(active); full.push_back(Candidate{static_cast<uint32_t>(total), {}, {}, {}});
                const auto cap = kernel.name == "nop" ? 1 : std::min<uint64_t>(total, uint64_t(sm_count) * active);
                selected.push_back(Candidate{static_cast<uint32_t>(cap), {}, {}, {}});
                history.emplace_back(Json::arrayValue);
            }
            measure_pass(run, *executor, full, events, launch_costs);
            measure_pass(run, *executor, selected, events, launch_costs);
            std::vector<bool> active(selected.size(), true);
            for (size_t i = 0; i < selected.size(); ++i) {
                history[i].append(selected[i].json()); active[i] = selected[i].cap > 1;
            }
            for (unsigned int step = 0; step < max_halvings && std::any_of(active.begin(), active.end(), [](bool x) { return x; }); ++step) {
                std::vector<Candidate> candidate;
                for (size_t i = 0; i < selected.size(); ++i)
                    candidate.push_back(Candidate{active[i] ? std::max(1U, selected[i].cap / 2) : selected[i].cap, {}, {}, {}});
                measure_pass(run, *executor, candidate, events, launch_costs);
                for (size_t i = 0; i < selected.size(); ++i) {
                    if (!active[i]) continue;
                    history[i].append(candidate[i].json());
                    if (double(median(candidate[i].maximum_piece_ns)) <=
                        double(median(selected[i].maximum_piece_ns)) * (1 - stable_fraction)) {
                        selected[i] = candidate[i]; active[i] = selected[i].cap > 1;
                    } else active[i] = false;
                }
                Json::Value progress; progress["halving_step"] = step + 1;
                progress["still_improving"] = Json::UInt64(std::count(active.begin(), active.end(), true));
                progress["validated_requests"] = Json::UInt64(run.check.count());
                print_record("HUMMINGBIRD_PROFILE_PROGRESS", progress);
            }
            if (std::any_of(active.begin(), active.end(), [](bool x) { return x; }))
                throw std::runtime_error("capacity search hit the predeclared bound before reaching a stable minimum");
            // Validate the final mixed set of selected capacities, not merely each
            // candidate individually. Every returned profile has real full-output checks.
            std::vector<Candidate> final;
            for (const auto &candidate : selected) final.push_back(Candidate{candidate.cap, {}, {}, {}});
            measure_pass(run, *executor, final, events, launch_costs);
            Json::Value kernels(Json::arrayValue);
            for (size_t i = 0; i < selected.size(); ++i) {
                auto &kernel = executor->kernel(i); Json::Value row;
                row["index"] = Json::UInt64(i); row["name"] = kernel.name;
                row["argument_count"] = Json::UInt64(kernel.args_ptr.size());
                row["grid"] = Json::Value(Json::arrayValue); row["block"] = Json::Value(Json::arrayValue);
                for (unsigned int axis = 0; axis < 3; ++axis) {
                    row["grid"].append(kernel.name == "nop" ? 1 : kernel.launch_params[axis]);
                    row["block"].append(kernel.name == "nop" ? 1 : kernel.launch_params[axis + 3]);
                }
                row["occupancy_active_blocks_per_sm"] = active_blocks[i]; row["cap"] = final[i].cap;
                row["split_ns"] = Json::UInt64(std::max(selected[i].conservative_ns(), final[i].conservative_ns()));
                row["whole_ns"] = Json::UInt64(full[i].conservative_ns());
                row["whole_samples"] = full[i].json(); row["candidates"] = history[i]; row["selected_validation"] = final[i].json();
                kernels.append(row);
            }
            result["kernels"] = kernels; result["validated_resnet_requests"] = Json::UInt64(run.check.count());
            result["launch_overhead_ns"] = Json::UInt64(median(launch_costs));
            result["launch_cost_samples_ns"] = numbers(launch_costs);
        }
        result["copy_profile"] = copy_profile(events);
        uint64_t max_small = 0;
        for (const auto &key : {"input_api_sync_ns", "output_api_sync_ns"})
            for (const auto &value : result["copy_profile"][key]) max_small = std::max(max_small, value.asUInt64());
        result["large_after_ns"] = Json::UInt64(max_small + std::max<uint64_t>(1000, max_small / 100));
        result["large_threshold_rule"] = "max isolated measured copy/sync API interval + max(1 us,1%)";
        result["small_input_enabled"] = false; result["small_output_enabled"] = false;
        result["small_pattern_eligibility"] = "not-yet-tested; require independent interference calibration before enabling";
        result["gpu_correctness_validated"] = true;
        write_new(output, result);
        Json::Value summary; summary["output"] = output; summary["kernel_count"] = result["kernels"].size();
        summary["gpu_correctness_validated"] = true; summary["small_patterns_enabled"] = false;
        print_record("HUMMINGBIRD_PROFILE_COMPLETE", summary);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "HUMMINGBIRD_PROFILE_FATAL %s\n", e.what()); return 1;
    }
}
