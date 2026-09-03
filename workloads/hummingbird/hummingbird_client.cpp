// SPDX-License-Identifier: GPL-2.0
// Scoped DNN frontend: no changes to the frozen GPreempt sources/libraries.
#include "idle_executor.h"
#include "client_checks.h"
#include "disb.h"
#include "gpreempt.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

using namespace hummingbird;
namespace {
std::string mode, split_cubin, profile_path, bpf_path;
std::unique_ptr<Profile> profile;
SharedState shared;
std::mutex setup_mutex;
bool idle_mode() { return mode == "idle_c" || mode == "idle_bpf"; }
class Client;
std::vector<Client *> clients;

class Client : public DISB::DependentClient {
public:
    explicit Client(const Json::Value &config) {
        role_ = config["priority"].asInt(); model_name_ = config["model_name"].asString();
        if (!config["priority"].isInt() || (role_ != 0 && role_ != 1) ||
            config["batch_size"] != 1 || config["use_cuda_graph"] != false ||
            config["preprocess_time"] != 200 || model_name_ != (role_ ? "resnet152" : "vgg"))
            throw std::runtime_error("only explicit batch-one VGG/ResNet role 0/1 and preprocess 200 are admitted");
        setName(config["name"].asString()); clients.push_back(this);
    }
    ~Client() {
        try { cleanup(); } catch (const std::exception &e) {
            std::fprintf(stderr, "HUMMINGBIRD_FATAL client cleanup: %s\n", e.what()); std::terminate();
        }
    }
    void init() override {
        foo::util::init_cuda(); cuda_check(cuCtxGetCurrent(&context_), "get primary context");
        if (mode == "native") create_streams(true); else create_streams(false);
        initialize_executor(false); allocate();
    }
    void initInThread() override {
        bind_core(role_);
        std::lock_guard<std::mutex> lock(setup_mutex);
        if (mode == "native") {
            cuda_check(cuCtxSetCurrent(context_), "bind native context");
        } else {
            release_resources();
            CUdevice device; cuda_check(cuDeviceGet(&device, 0), "select GPU");
            cuda_check(cuCtxCreate(&context_, 0, device), "create independent context"); owns_context_ = true;
            NvContext nv{}; nv.hClient = util_gettid();
            if (NvRmQuery(&nv) != NV_OK || NvRmModifyTS(nv, 1000000) != NV_OK)
                throw std::runtime_error("owned context 1,000,000 us timeslice setup failed");
            create_streams(false); initialize_executor(idle_mode() && role_ == 1); allocate();
            Json::Value record; record["task"] = getName(); record["role"] = role_;
            record["mode"] = mode; record["timeslice_us"] = Json::UInt64(1000000);
            record["owned_query_ok"] = true; record["timeslice_set_ok"] = true;
            record["hclient"] = nv.hClient; record["hobject"] = nv.hObject;
            record["stream_priority"] = 0; record["context_kind"] = "independent";
            print_record("HUMMINGBIRD_CONTEXT", record);
        }
        cuda_check(cuEventCreate(&compute_end_, CU_EVENT_DISABLE_TIMING), "create HP compute event");
        cuda_check(cuEventCreate(&copy_start_, CU_EVENT_DISABLE_TIMING), "create copy start event");
        cuda_check(cuEventCreate(&copy_end_, CU_EVENT_DISABLE_TIMING), "create copy end event");
        if (auto *idle = dynamic_cast<IdleExecutor *>(executor_.get())) idle->start(context_, stream_);
        if (!role_ && idle_mode()) {
            std::lock_guard<std::mutex> admission(shared.admission);
            shared.hp_context = context_; shared.last_hp_activity_ns = now_ns();
            shared.hp_gpu_done = true; shared.hp_pending = 0;
        }
        initialized_ = true; checks_.begin_timed();
    }
    void prepareInput() override {}
    void preprocess() override { usleep(200); }
    void copyInput() override { copy(true); }
    void infer() override {
        if (initialized_ && !role_ && idle_mode()) {
            { std::lock_guard<std::mutex> lock(shared.admission);
              shared.small_active = false; shared.hp_pending = 1; shared.hp_gpu_done = false;
              shared.last_hp_activity_ns = now_ns(); ++shared.hp_enqueues; }
            // Publish before any HP enqueue; never hold the admission lock for
            // an entire inference or while waiting for GPU execution.
            status_check(executor_->execute(stream_), "HP explicit execution");
            cuda_check(cuEventRecord(compute_end_, stream_), "record HP completion");
            cuda_check(cuStreamSynchronize(stream_), "synchronize HP compute");
            cuda_check(cuEventQuery(compute_end_), "confirm HP compute completion");
            { std::lock_guard<std::mutex> lock(shared.admission);
              shared.hp_pending = 0; shared.hp_gpu_done = true;
              shared.last_hp_activity_ns = now_ns(); ++shared.hp_completions; }
        } else {
            status_check(executor_->execute(stream_), "model execution");
            if (auto *idle = dynamic_cast<IdleExecutor *>(executor_.get())) idle->synchronize();
            cuda_check(cuStreamSynchronize(stream_), "model synchronization");
        }
    }
    void copyOutput() override { copy(false); }
    void postprocess() override { checks_.check(output_, output_size_); }
    int role() const { return role_; }
    void cleanup() {
        if (!executor_) return;
        cuda_check(cuCtxSetCurrent(context_), "client cleanup context");
        if (auto *idle = dynamic_cast<IdleExecutor *>(executor_.get())) {
            idle->shutdown(); Json::Value record = idle->report();
            record["task"] = getName(); record["mode"] = mode;
            print_record("HUMMINGBIRD_EXECUTOR", record);
        }
        if (!role_ && idle_mode() && initialized_) {
            std::lock_guard<std::mutex> lock(shared.admission);
            if (shared.hp_pending || !shared.hp_gpu_done) throw std::runtime_error("HP cleanup while GPU work pending");
            shared.small_active = false; shared.hp_context = nullptr;
            Json::Value record; record["hp_enqueues"] = Json::UInt64(shared.hp_enqueues);
            record["hp_completions"] = Json::UInt64(shared.hp_completions);
            record["input_bubbles"] = Json::UInt64(shared.input_bubbles);
            record["output_bubbles"] = Json::UInt64(shared.output_bubbles);
            record["input_bubble_ns"] = Json::UInt64(shared.input_bubble_ns);
            record["output_bubble_ns"] = Json::UInt64(shared.output_bubble_ns);
            record["small_event_checks"] = Json::UInt64(shared.small_event_checks);
            record["small_event_query_ns"] = Json::UInt64(shared.small_event_query_ns);
            print_record("HUMMINGBIRD_HP_EVENTS", record);
        }
        release_resources();
        if (owns_context_) { cuda_check(cuCtxDestroy(context_), "destroy owned context"); owns_context_ = false; }
        context_ = nullptr;
        Json::Value record; record["task"] = getName(); record["mode"] = mode; record["complete"] = true;
        print_record("HUMMINGBIRD_CLEANUP", record);
    }
private:
    void create_streams(bool native) {
        int least = 0, greatest = 0, actual = 0;
        cuda_check(cuCtxGetStreamPriorityRange(&least, &greatest), "query priority range");
        const int desired = native ? (role_ ? least : greatest) : 0;
        cuda_check(cuStreamCreateWithPriority(&stream_, 0, desired), "create compute stream");
        cuda_check(cuStreamCreateWithPriority(&copy_stream_, 0, desired), "create copy stream");
        cuda_check(cuStreamGetPriority(stream_, &actual), "query actual priority");
        if (actual != desired) throw std::runtime_error("actual stream priority differs from requested");
        if (native) {
            if (least <= greatest) throw std::runtime_error("native GPU priority range is degenerate");
            std::fprintf(stderr, "GPREEMPT_LOAD_PRIORITY task=%s role=%d actual=%d least=%d greatest=%d\n",
                         getName().c_str(), role_, actual, least, greatest);
        }
    }
    void initialize_executor(bool scheduled) {
        if (scheduled) executor_ = std::make_shared<IdleExecutor>(split_cubin, *profile, shared,
                                                               mode == "idle_bpf", bpf_path);
        else if (idle_mode() && role_) executor_ = std::make_shared<SplitModel>(split_cubin);
        else executor_ = std::make_shared<CheckedExecutor>();
        status_check(executor_->init(model_name_), "initialize real model");
        input_size_ = executor_->get_data_size("data"); output_size_ = executor_->get_data_size("heads");
    }
    void allocate() {
        cuda_check(cuMemHostAlloc(&input_, input_size_, CU_MEMHOSTALLOC_PORTABLE), "allocate input");
        cuda_check(cuMemHostAlloc(&output_, output_size_, CU_MEMHOSTALLOC_PORTABLE), "allocate output");
        checks_.initialize(getName(), std::string(MODEL_PATH) + "/" + model_name_ + "/reference.f32",
                           input_, input_size_, output_size_);
    }
    void copy(bool input) {
        const bool hp_idle = initialized_ && !role_ && idle_mode();
        const bool eligible = hp_idle && (input ? profile->small_input_enabled : profile->small_output_enabled);
        CUstream stream = input ? copy_stream_ : stream_;
        uint64_t begin = 0;
        // Publish only after BOTH events have been recorded. A never-recorded
        // event queries successful, which must not be mistaken for completion.
        if (eligible) {
            std::lock_guard<std::mutex> lock(shared.admission);
            if (shared.hp_pending || !shared.hp_gpu_done) throw std::runtime_error("copy bubble overlaps unfinished HP compute");
            cuda_check(cuEventRecord(copy_start_, stream), "record actual copy start");
            if (input) status_check(executor_->set_input("data", input_, input_size_, stream), "copy input");
            else status_check(executor_->get_output("heads", output_, output_size_, stream), "copy output");
            cuda_check(cuEventRecord(copy_end_, stream), "record actual copy end");
            shared.small_start = copy_start_; shared.small_end = copy_end_; shared.small_active = true;
            shared.small_is_input = input;
            begin = now_ns();
            if (input) ++shared.input_bubbles; else ++shared.output_bubbles;
        } else if (input) status_check(executor_->set_input("data", input_, input_size_, stream), "copy input");
        else status_check(executor_->get_output("heads", output_, output_size_, stream), "copy output");
        cuda_check(cuStreamSynchronize(stream), "copy synchronization");
        if (eligible) {
            cuda_check(cuEventQuery(copy_end_), "confirm actual copy completion");
            std::lock_guard<std::mutex> lock(shared.admission); shared.small_active = false;
            if (input) shared.input_bubble_ns += now_ns() - begin;
            else shared.output_bubble_ns += now_ns() - begin;
        }
    }
    void release_resources() {
        if (!context_) return;
        cuda_check(cuCtxSetCurrent(context_), "release resource context");
        if (executor_) { executor_->clear(); executor_.reset(); }
        if (input_) { cuda_check(cuMemFreeHost(input_), "free input"); input_ = nullptr; }
        if (output_) { cuda_check(cuMemFreeHost(output_), "free output"); output_ = nullptr; }
        if (stream_) { cuda_check(cuStreamDestroy(stream_), "destroy compute stream"); stream_ = nullptr; }
        if (copy_stream_) { cuda_check(cuStreamDestroy(copy_stream_), "destroy copy stream"); copy_stream_ = nullptr; }
        for (CUevent *event : {&compute_end_, &copy_start_, &copy_end_})
            if (*event) { cuda_check(cuEventDestroy(*event), "destroy client event"); *event = nullptr; }
    }
    int role_ = -1;
    std::string model_name_;
    bool initialized_ = false, owns_context_ = false;
    CUcontext context_ = nullptr;
    CUstream stream_ = nullptr, copy_stream_ = nullptr;
    CUevent compute_end_ = nullptr, copy_start_ = nullptr, copy_end_ = nullptr;
    std::shared_ptr<foo::BaseExecutor> executor_;
    size_t input_size_ = 0, output_size_ = 0;
    void *input_ = nullptr, *output_ = nullptr;
    gpreempt_artifact::OutputCheck checks_;
};
}

int main(int argc, char **argv) {
    try {
        if (argc == 2 && !std::strcmp(argv[1], "--help")) {
            std::cout << "hummingbird_client CONFIG.json --mode native|timeslice_control|idle_c|idle_bpf "
                         "[--profile PROFILE.json --split-cubin SPLIT.cubin --bpf-program PROGRAM.bin]\n";
            return 0;
        }
        if (argc < 4 || argc % 2) throw std::runtime_error("invalid command; use --help");
        for (int i = 2; i < argc; i += 2) {
            std::string arg = argv[i], value = argv[i + 1];
            std::string *target = arg == "--mode" ? &mode : arg == "--profile" ? &profile_path :
                arg == "--split-cubin" ? &split_cubin : arg == "--bpf-program" ? &bpf_path : nullptr;
            if (!target || !target->empty()) throw std::runtime_error("unknown or repeated option: " + arg);
            *target = value;
        }
        if (mode != "native" && mode != "timeslice_control" && !idle_mode()) throw std::runtime_error("unknown mode");
        if (const char *value = std::getenv("GPREEMPT_POLICY"))
            if (std::string(value) != "original") throw std::runtime_error("idle study rejects unrelated driver BPF policy");
        if (idle_mode()) {
            if (profile_path.empty() || split_cubin.empty() || (mode == "idle_bpf" && bpf_path.empty()))
                throw std::runtime_error("idle mode requires profile/cubin and BPF mode requires real bytecode");
            profile = std::make_unique<Profile>(profile_path);
        } else if (!profile_path.empty() || !split_cubin.empty() || !bpf_path.empty())
            throw std::runtime_error("unsplit control cannot consume a split profile/program");
        Json::Value config = read_json(argv[1]);
        const bool lc_only = mode == "native" && config["tasks"].size() == 1 &&
                             config["tasks"][0]["client"]["priority"] == 0;
        if (config["tasks"].size() != 2 && !lc_only)
            throw std::runtime_error("one HP and one LP task required; only native admits isolated HP calibration");
        bool roles[2] = {};
        for (const auto &task : config["tasks"]) {
            const auto &c = task["client"];
            if (!c["priority"].isInt() || c["priority"].asInt() < 0 || c["priority"].asInt() > 1 || roles[c["priority"].asInt()])
                throw std::runtime_error("missing or duplicate client role");
            roles[c["priority"].asInt()] = true;
        }
        Json::Value setup; setup["mode"] = mode; setup["graph"] = false; setup["profile_path"] = profile_path;
        setup["split_cubin"] = split_cubin; setup["bpf_program"] = bpf_path;
        setup["lp_worker_cpu"] = idle_mode() ? Json::Value(2) : Json::Value();
        setup["small_input_enabled"] = profile ? profile->small_input_enabled : false;
        setup["small_output_enabled"] = profile ? profile->small_output_enabled : false;
        setup["isolated_lc_calibration"] = lc_only;
        setup["gpreempt_driver_policy"] = "original";
        print_record("HUMMINGBIRD_SETUP", setup);
        DISB::BenchmarkSuite suite;
        Json::StreamWriterBuilder writer;
        suite.init(Json::writeString(writer, config), [](const Json::Value &c) -> std::shared_ptr<DISB::Client> {
            return std::make_shared<Client>(c);
        });
        suite.run(); std::cout << suite.generateReport() << std::endl;
        // Stop the LP event reader before destroying HP-owned event/context handles.
        for (int role : {1, 0}) for (auto *client : clients) if (client->role() == role) client->cleanup();
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "HUMMINGBIRD_FATAL %s\n", e.what()); return 1;
    }
}
