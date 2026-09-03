// CPU-only fake-device test of the exact adapter_live.inc control flow.
// No torch/CUDA. Fake copies cannot establish GPU correctness or performance.
#include "adapter_state.h"
#include "finemoe_runtime_safety.h"
#include <atomic>
#include <condition_variable>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace torch {
enum DeviceType { kCPU, kCUDA };
struct Device {
    DeviceType kind;
    int id;
    Device(DeviceType type = kCPU, int index = -1) : kind(type), id(index) {}
    bool is_cuda() const { return kind == kCUDA; }
    bool is_cpu() const { return kind == kCPU; }
    int index() const { return id; }
    bool operator==(const Device &other) const { return kind == other.kind && id == other.id; }
    bool operator!=(const Device &other) const { return !(*this == other); }
};
}
struct Pool {
    std::int64_t capacity = 64, used = 0;
    std::int64_t GetFreeMemory(torch::Device) const { return capacity - used; }
};
static std::unique_ptr<Pool> kDeviceMemoryPool;
struct Node {
    std::uint64_t id = 0;
    std::int64_t byte_size = 16;
    bool is_sparse = true, fail_copy = false;
    std::atomic<unsigned> state{0};
    std::mutex mutex;
    std::condition_variable cv;
    torch::Device device, default_device{torch::kCUDA, 0}, default_host;
    void *device_memory_ptr = nullptr, *host_memory_ptr = this;
    void SetDevice(torch::Device destination, bool = false, void * = nullptr, bool = false) {
        if (fail_copy && destination.is_cuda()) throw std::runtime_error("fake transfer failure");
        if (device.is_cuda() != destination.is_cuda())
            kDeviceMemoryPool->used += destination.is_cuda() ? byte_size : -byte_size;
        device = destination;
        device_memory_ptr = device.is_cuda() ? this : nullptr;
    }
};
using NodePtr = std::shared_ptr<Node>;
using NodePtrList = std::vector<NodePtr>;
struct Topology {
    NodePtrList nodes;
    NodePtrList GetSparseNodes() const { return nodes; }
    NodePtr GetNodeFromTensorID(std::uint32_t id) const { return nodes.at(id); }
    std::int64_t GetSparseCacheLimit(torch::Device, std::int64_t dense = 0) const {
        return kDeviceMemoryPool->capacity - dense;
    }
};
static std::unique_ptr<Topology> kTopologyHandle;
static int GetDeviceCount() { return 1; }
struct Task {
    eb_u64 eb_epoch = 0;
    eb_u32 eb_layer = 0, eb_expert = 0, priority = 0;
    bool on_demand = true;
    NodePtr node;
    torch::Device dst_device{torch::kCUDA, 0};
};
using TaskPtr = std::shared_ptr<Task>;
class ArcherTaskPool {
public:
    void DrainForSnapshot() {} // No asynchronous task queue in this CPU fixture.
    void ConfigureExpertBuffering(const std::string &, eb_u32, const std::string &, const std::string &);
    eb_u64 BeginExpertBuffering(eb_u32, eb_u32, const std::vector<std::uint32_t> &, const std::vector<eb_u32> &);
    void EndExpertBuffering(eb_u32, eb_u64);
    void CheckExpertBufferingResidency(eb_u32, const NodePtrList &);
    void StampExpertBufferingTask(const TaskPtr &);
    bool RemoveExpertBufferingNode(const TaskPtr &);
    void CopyExpertBufferingNode(const TaskPtr &);
    std::int64_t ExpertBufferingResidentBytes() const;
    std::map<std::string, eb_u64> ExpertBufferingStats();
    std::mutex eb_mutex_, exec_mutex_;
    std::unique_ptr<eb_adapter::State> eb_state_;
    std::map<eb_u32, NodePtrList> eb_nodes_;
    std::map<std::uint64_t, TaskPtr> exec_queue_;
};
#include "adapter_live.inc"

static unsigned checks = 0;
static void Check(bool ok) {
    ++checks;
    if (!ok) throw std::runtime_error("fake-device control-flow assertion");
}
template <class F> static void Reject(F action) {
    bool rejected = false;
    try { action(); } catch (const std::runtime_error &) { rejected = true; }
    Check(rejected);
}
static void Reset() {
    kDeviceMemoryPool = std::make_unique<Pool>();
    kTopologyHandle = std::make_unique<Topology>();
    for (unsigned i = 0; i < 4; ++i) {
        auto node = std::make_shared<Node>();
        node->id = 100 + i;
        kTopologyHandle->nodes.push_back(node);
    }
}
static TaskPtr Stamp(ArcherTaskPool &pool, unsigned expert) {
    auto task = std::make_shared<Task>();
    task->node = kTopologyHandle->nodes.at(expert);
    std::lock_guard<std::mutex> incoming(task->node->mutex);
    task->node->state = 1;
    pool.StampExpertBufferingTask(task);
    pool.exec_queue_[task->node->id] = task;
    return task;
}
static void Finish(ArcherTaskPool &pool, const TaskPtr &task) {
    task->node->state = 0;
    pool.exec_queue_.erase(task->node->id);
}
static void Demand(ArcherTaskPool &pool, unsigned expert) {
    auto task = Stamp(pool, expert);
    if (!task->node->device.is_cuda()) {
        Check(pool.RemoveExpertBufferingNode(task));
        finemoe_revision::CompleteDemand(task->node->mutex, task->node->cv, task->node->state, [&] {
            pool.CopyExpertBufferingNode(task); // actual included adapter function
            Check(task->node->state == 1); // admission precedes ready publication
            Check(pool.eb_state_->Get(task->eb_layer).input.experts[expert].flags == EB_RESIDENT);
        });
    }
    Finish(pool, task);
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    try {
        for (const std::string arm : {"fifo", "native", "bpf"}) {
            Reset();
            ArcherTaskPool pool;
            pool.ConfigureExpertBuffering(arm, 2, argv[1], argv[2]);
            auto epoch = pool.BeginExpertBuffering(17, 0, {0, 1, 2, 3}, {1, 1, 1, 1});
            Demand(pool, 0);
            Demand(pool, 1);
            Demand(pool, 0); // hit cannot refresh insertion order
            Demand(pool, 2);
            const unsigned victim = arm == "fifo" ? 0 : 1;
            Check(kTopologyHandle->nodes[victim]->device.is_cpu());
            Check(kTopologyHandle->nodes[2]->device.is_cuda());
            auto stats = pool.ExpertBufferingStats();
            Check(stats.at("admissions") == 3 && stats.at("evictions") == 1);
            Check(stats.at("jit_calls") == (arm == "bpf" ? stats.at("decisions") : 0));
            Check(stats.at("resident_sparse_bytes") == 32);
            pool.EndExpertBuffering(17, epoch);
        }

        Reset();
        ArcherTaskPool pool;
        pool.ConfigureExpertBuffering("bpf", 2, argv[1], argv[2]);
        auto epoch = pool.BeginExpertBuffering(5, 0, {0, 1, 2, 3}, {1, 1, 1, 1});
        Demand(pool, 0);
        Demand(pool, 1);
        auto incoming = Stamp(pool, 2);
        Reject([&] { pool.EndExpertBuffering(5, epoch); });
        std::promise<void> locked, release;
        auto released = release.get_future();
        std::thread holder([&] {
            std::scoped_lock<std::mutex, std::mutex> guard(
                kTopologyHandle->nodes[0]->mutex, kTopologyHandle->nodes[1]->mutex);
            locked.set_value();
            released.wait();
        });
        locked.get_future().wait();
        const bool admitted = pool.RemoveExpertBufferingNode(incoming);
        release.set_value();
        holder.join();
        Check(!admitted && pool.eb_state_->Stats().evictions == 0);
        incoming->eb_epoch++;
        Reject([&] { pool.RemoveExpertBufferingNode(incoming); });
        incoming->eb_epoch--;
        Check(pool.RemoveExpertBufferingNode(incoming));
        incoming->node->fail_copy = true;
        const auto serial = pool.eb_state_->Get(5).serial;
        Reject([&] {
            finemoe_revision::CompleteDemand(incoming->node->mutex, incoming->node->cv,
                incoming->node->state, [&] { pool.CopyExpertBufferingNode(incoming); });
        });
        Check(pool.eb_state_->Get(5).serial == serial && incoming->node->state == 1);
        Check(!incoming->node->device.is_cuda());
        incoming->node->fail_copy = false;
        finemoe_revision::CompleteDemand(incoming->node->mutex, incoming->node->cv,
            incoming->node->state, [&] { pool.CopyExpertBufferingNode(incoming); });
        Finish(pool, incoming);
        auto physical = kTopologyHandle->nodes[2];
        physical->SetDevice(torch::Device(torch::kCPU));
        Reject([&] { pool.ExpertBufferingStats(); });
        physical->SetDevice(torch::Device(torch::kCUDA, 0));
        pool.EndExpertBuffering(5, epoch);

        Reset();
        ArcherTaskPool limited;
        kDeviceMemoryPool->capacity = 16; // K=2 cannot override the byte budget.
        limited.ConfigureExpertBuffering("native", 2, argv[1], argv[2]);
        limited.BeginExpertBuffering(0, 0, {0, 1, 2, 3}, {1, 1, 0, 0});
        Demand(limited, 0);
        auto second = Stamp(limited, 1);
        Check(!limited.RemoveExpertBufferingNode(second));
        Check(limited.eb_state_->Stats().admissions == 1);
        std::cout << "fake_device_control_checks=" << checks << " device_backend=fake\n";
    } catch (const std::exception &error) {
        std::cerr << "EB_FAKE_DEVICE_FAILURE: " << error.what() << "\n";
        return 1;
    }
}
