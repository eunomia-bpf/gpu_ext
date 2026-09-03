# Bounded exact-source excerpts for OpenCode adapter task

## workloads/expert-buffering-policy/section-vi/adapter_state.h (line ranges 1,160)

```
// SPDX-License-Identifier: Apache-2.0
// CUDA-free state used by the private live adapter; caller serializes access.
#pragma once
#include "policy.h"
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace eb_adapter {
enum class Arm { Fifo, Native, Bpf };
using NodeId = std::uint64_t;
using Counts = std::vector<eb_u32>;

struct Cohort {
    std::vector<NodeId> nodes;
    eb_input input{};
    eb_u64 serial = 0;
};

struct Counters {
    eb_u64 decisions = 0, jit_calls = 0, admissions = 0, evictions = 0;
};

class State {
public:
    State(Arm arm, eb_u32 capacity, const std::string &library,
          const std::string &bytecode);
    ~State();
    State(const State &) = delete;
    State &operator=(const State &) = delete;

    // One sequential GPU stream of layer invocations, no overlapping batches.
    eb_u64 Begin(eb_u32 layer, eb_u32 device, const std::vector<NodeId> &nodes,
                 const Counts &counts);
    void End(eb_u32 layer, eb_u64 epoch);
    std::pair<eb_u32, eb_u32> Locate(NodeId node) const;
    const Cohort &Get(eb_u32 layer) const;
    eb_u64 ActiveEpoch(eb_u32 layer, eb_u32 expert) const;
    eb_context Decide(eb_u32 layer, eb_u32 incoming, const std::vector<bool> &eligible);
    void Validate(const eb_context &snapshot) const;
    void Evicted(const eb_context &snapshot);
    void CanAdmit(eb_u32 layer, eb_u64 epoch, eb_u32 expert) const;
    void Admitted(eb_u32 layer, eb_u64 epoch, eb_u32 expert);
    Counters Stats() const;
    static eb_u64 NextSerial(eb_u64 serial);

private:
    void RequireActive(eb_u32 layer, eb_u64 epoch) const;
    Arm arm_;
    eb_u32 capacity_;
    std::map<eb_u32, Cohort> layers_;
    std::map<NodeId, std::pair<eb_u32, eb_u32>> locations_;
    bool active_ = false;
    eb_u32 active_layer_ = 0;
    void *library_ = nullptr;
    void *jit_ = nullptr;
    eb_u64 (*native_)(eb_context *) = nullptr;
    int (*bpf_)(void *, eb_context *) = nullptr;
    eb_u64 (*jit_calls_)(void *) = nullptr;
    void (*close_)(void *) = nullptr;
    Counters counters_;
};
} // namespace eb_adapter
```

## workloads/finemoe/deps/FineMoE-EuroSys26/core/prefetch/task_scheduler.h (line ranges 1,170)

```
// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include <deque>
#include <iostream>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/pytorch.h"
#include "model/model_topology.h"
#include "utils/noncopyable.h"
#include "finemoe_copy_ledger.h"

#define SKIP_TO_NEXT_ITERATION                                  \
    std::this_thread::sleep_for(std::chrono::microseconds(10)); \
    continue;

#define NUM_PRIORITY 20UL

struct Task {
    bool on_demand = false;
    NodePtr node;
    std::vector<NodePtr> remove_nodes;
    std::uint32_t priority;
    std::uint64_t request_id;
    torch::Device src_device = DISK_DEVICE;
    torch::Device dst_device = DISK_DEVICE;

    bool remove_layer = false;

    std::string DebugString()
    {
        std::stringstream ss;
        ss << "Task: node: " << node->str() << ", on_demand: " << on_demand
           << ", priority: " << priority << "[" << src_device.str() << "->" << dst_device.str()
           << "]";
        return ss.str();
    }
};
typedef std::shared_ptr<Task> TaskPtr;

class ArcherTaskPool : public noncopyable {
public:
    void StartExec(const std::uint64_t& request_id, const NodePtr& node);
    void FetchExec(const std::uint64_t& request_id, const NodePtr& node);
    void StopExec(const std::uint64_t& request_id, const NodePtr& node);
    void EnqueueTask(const TaskPtr& task);
    void DrainForSnapshot();

    void ClearQueue()
    {
        std::lock_guard<std::mutex> lock(unified_mutex_);
        for (std::uint32_t priority = 1; priority < NUM_PRIORITY; priority++) {
            finemoe_revision::Ledger().Count("prefetch_queue_canceled", unified_queue_[priority].size());
            unified_queue_[priority].clear();
        }
    }
    void ClearCacheSparseNode(int device_id);

    bool RemoveCachedSparseNode(const NodePtr& node, int device_id = -1, bool speculative = false);
    bool RemoveCachedDenseNode(const NodePtr& node);
    // void RemoveCachedNode(const NodePtr& node);

    void ReplaceCacheCandidates(const NodePtrList& candidates)
    {
        std::lock_guard<std::mutex> lock(unified_mutex_);
        {
            std::lock_guard<std::mutex> lock(this->candidates_mutex_);
            candidates_.clear();
            for (auto& node : candidates) { candidates_.insert(node); }
        }

        for (std::uint32_t priority = 1; priority < NUM_PRIORITY; priority++) {
            finemoe_revision::Ledger().Count("prefetch_queue_canceled", unified_queue_[priority].size());
            unified_queue_[priority].clear();
        }
    }

    DELETE_COPY_AND_ASSIGN(ArcherTaskPool);
    STATIC_GET_INSTANCE(ArcherTaskPool);

    ArcherTaskPool();
    ~ArcherTaskPool()
    {
        std::cout << "ArcherTaskPool destructor" << std::endl;
        main_thread_stop_flag_.store(true);
        // wait for all threads to stop
        for (auto& thread_list : exec_threads_) {
            for (auto& thread : thread_list) { thread.join(); }
        }
    }

private:
    void GPUThreadFunc(int gpu_id, int thread_id);
    void SetNodeDevice(const TaskPtr& task);
    void FinishTask();

    std::string DebugString(const std::vector<std::deque<TaskPtr>>& queue);

private:
    std::vector<std::deque<TaskPtr>> unified_queue_;  // For ordered prefetch
    std::vector<std::vector<std::uint32_t>> gpu_min_priority_;
    std::unordered_map<std::uint64_t, TaskPtr> exec_queue_;
    std::mutex exec_mutex_;
    std::mutex unified_mutex_;
    std::condition_variable drain_cv_;
    std::size_t active_tasks_ = 0;
    std::mutex candidates_mutex_;

    std::vector<std::list<std::thread>> exec_threads_;

    std::unordered_set<NodePtr> candidates_;

    std::atomic<bool> main_thread_stop_flag_;
};

extern std::unique_ptr<ArcherTaskPool> kTaskPool;
```

## workloads/finemoe/deps/FineMoE-EuroSys26/core/prefetch/task_scheduler.cpp (line ranges 129,235;258,344;484,636)

```
void ArcherTaskPool::StartExec(const std::uint64_t& request_id, const NodePtr& node)
{
    auto task = std::make_shared<Task>();
    task->on_demand = true;
    task->node = node;
    task->priority = 0;
    task->src_device = node->device;
    task->dst_device = node->default_device;
    task->request_id = request_id;

    ARCHER_LOG_DEBUG("StartExec: {}", task->DebugString());

    node->visit_count += 1;
    if (node->device.is_cuda()) { node->incache_visit_count++; }
    node->last_access_time = MCIROSECONDS_SINCE_EPOCH;
    node->io_state = static_cast<NodeState>(node->io_state | NODE_STATE_VISITED);

    auto node_body = kTopologyHandle->GetNodeBodyFromCorrID(node->corr_id);

    node_body->visit_cnt += 1;

    /* Observation: expert IO time + GPU inference time is similar to its compute time in CPU.
     * Solution: If the node is on CPU, and no other nodes are running on CPU, do not perform H2D
     * memory copy. Allow two node run concurrently.
     */
    // if ((++cpu_running_nodes_ < 2) && task->src_device.is_cpu() && node->is_sparse) {
    //     task->dst_device = task->src_device;
    //     node_body->cpu_visit_cnt += 1;
    // }

    if (task->dst_device.is_cuda()) { node_body->gpu_visit_cnt += 1; }

    node->last_prefetch_time = MCIROSECONDS_SINCE_EPOCH;
    {
        std::lock_guard<std::mutex> lock(unified_mutex_);
        for (std::size_t i = 0; i < NUM_PRIORITY; ++i) {
            // remove any task that has the same node
            auto it =
                std::remove_if(unified_queue_[i].begin(), unified_queue_[i].end(), [&](auto& t) {
                    const bool remove = (t->node == node) |
                           ((node->corr_id & 0xffffffff) > (t->node->corr_id & 0xffffffff));
                    if (remove && t->priority > 0) finemoe_revision::Ledger().Count("prefetch_queue_canceled");
                    return remove;
                });
            unified_queue_[i].erase(it, unified_queue_[i].end());
        }
    }

    if (task->src_device.is_cuda()) {
        // ARCHER_LOG_DEBUG("StartExec: {} is on the same device", task->DebugString());
        std::lock_guard<std::mutex> lock(exec_mutex_);
        exec_queue_.insert({node->id, task});
        node->state = 0;
        node->cv.notify_all();

        if (task->dst_device.is_cpu()) { node_body->cpu_hit_cnt += 1; }

        if (task->dst_device.is_cuda()) node_body->hit_cnt += 1;

        return;
    }

    if (task->dst_device.is_cpu()) { node_body->cpu_miss_cnt += 1; }
    if (task->dst_device.is_cuda()) { node_body->gpu_miss_cnt += 1; }

    {
        std::lock_guard<std::mutex> lock(exec_mutex_);
        if (exec_queue_.find(node->id) != exec_queue_.end()) {
            std::stringstream ss;
            ss << "Node " << std::hex << node->id << " is already in exec queue";
            ARCHER_LOG_WARN(ss.str().c_str());
            node->state = 0;
            node->cv.notify_all();
            return;
        }
        exec_queue_.insert({node->id, task});
    }

    {
        std::lock_guard<std::mutex> lock(unified_mutex_);
        unified_queue_[task->priority].push_back(task);
    }
}

void ArcherTaskPool::StopExec(const std::uint64_t& request_id, const NodePtr& node)
{
    auto task = std::make_shared<Task>();
    task->on_demand = true;
    task->node = node;
    task->priority = 0;
    task->src_device = node->device;
    task->dst_device = node->default_host;
    task->request_id = request_id;

    ARCHER_LOG_DEBUG("StopExec: {}", task->DebugString());

    node->state = 0;
    node->cv.notify_all();
    {
        std::lock_guard<std::mutex> lock(exec_mutex_);
        exec_queue_.erase(node->id);
    }

    return;
}

void ArcherTaskPool::ClearCacheSparseNode(int device_id)
bool ArcherTaskPool::RemoveCachedSparseNode(const NodePtr& node, int device_id, bool speculative)
{
    // ARCHER_LOG_DEBUG("RemoveCachedSparseNode: {}", node->str());

    if (node->device.is_cuda()) { return true; }

    auto start_time = MILLISECONDS_SINCE_EPOCH;

    auto nodes = kTopologyHandle->GetSparseNodes();

    // get all nodes in exec queue
    std::unordered_set<NodePtr> nodes_exec;
    {
        std::lock_guard<std::mutex> lock(exec_mutex_);
        for (auto& [id, task] : exec_queue_) { nodes_exec.insert(task->node); }
    }

    if (device_id == -1) device_id = node->default_device.index();

    auto cache_limit = kTopologyHandle->GetSparseCacheLimit(
        torch::Device(torch::kCUDA, device_id), node->is_sparse ? 0 : node->byte_size);
    ARCHER_LOG_DEBUG(
        "GetSparseCacheLimit: {}, {}MB",
        device_id,
        cache_limit / MB
    );
    if (node->is_sparse) cache_limit -= node->byte_size;

    int64_t cache_size = 0;
    NodePtrList device_nodes;
    for (auto& n : nodes) {
        if (n->device.is_cuda() && (n->device.index() == device_id)) {
            cache_size += n->byte_size;
            device_nodes.push_back(n);
        }
    }

    ARCHER_LOG_DEBUG(
        "RemoveCachedSparseNode: {} {}MB {}MB {}",
        device_id,
        cache_size / MB,
        cache_limit / MB,
        device_nodes.size()
    );

    if (cache_size > cache_limit) {
        std::vector<std::size_t> node_access_time;
        std::vector<std::size_t> node_index(device_nodes.size());
        std::iota(node_index.begin(), node_index.end(), 0);

        for (auto& n : device_nodes) {
            node_access_time.push_back(n->prob * n->incache_visit_count);
        }
        
        std::sort(node_index.begin(), node_index.end(), [&](int i, int j) {
            return node_access_time[i] < node_access_time[j];
        });
        for (auto i : node_index) {
            auto& n = device_nodes[i];
            {
                std::lock_guard<std::mutex> lock(this->candidates_mutex_);
                if (speculative && candidates_.find(n) != candidates_.end()) { continue; }
            }
            if (nodes_exec.find(n) != nodes_exec.end()) { continue; }
            if (n->mutex.try_lock()) {
                ARCHER_LOG_DEBUG("RemoveCachedSparseNode: {}", n->str());
                n->SetDevice(n->default_host);
                n->incache_visit_count = 0;
                n->prob = 0;
                n->mutex.unlock();
                cache_size -= n->byte_size;
                if ((node->io_state & NODE_STATE_VISITED) == 0) node->unused_count += 1;
            }
            if (cache_size <= cache_limit) { break; }
        }
    }

    auto end_time = MILLISECONDS_SINCE_EPOCH;
    ARCHER_LOG_DEBUG(
        "RemoveCachedSparseNode: cache_size {}MB, cache_limit {}MB, node {}, {}us",
        cache_size / MB,
        cache_limit / MB,
        node->str(),
        end_time - start_time
    );

    return cache_size <= cache_limit;
void ArcherTaskPool::GPUThreadFunc(int gpu_id, int thread_id)
{
    while (!main_thread_stop_flag_.load()) {
        std::uint32_t max_priority = 1000;
        std::unique_lock<std::mutex> lock(unified_mutex_);
        for (std::uint32_t i = 0; i < NUM_PRIORITY; ++i) {
            if (!unified_queue_[i].empty()) {
                max_priority = i;
                break;
            }
        }

        if (max_priority == 1000) {
            lock.unlock();
            SKIP_TO_NEXT_ITERATION
        }

        // Find a task that can be executed on the current GPU
        TaskPtr task = nullptr;
        for (auto& t : unified_queue_[max_priority]) {
            if (t->dst_device.index() == gpu_id) {
                task = t;
                break;
            }
        }

        if (task == nullptr) {
            lock.unlock();
            SKIP_TO_NEXT_ITERATION
        }

        auto node = task->node;
        node->incache_visit_count += 1;

        // remove task from the queue
        for (std::uint32_t i = 0; i < NUM_PRIORITY; ++i) {
            unified_queue_[i].erase(std::remove_if(unified_queue_[i].begin(),
                                                   unified_queue_[i].end(),
                                                   [&, task](auto& t) {
                                                       const bool remove = (t->node == node) &
                                                              (t->dst_device == task->dst_device);
                                                       if (remove && t->priority > 0 && t != task)
                                                           finemoe_revision::Ledger().Count("prefetch_queue_canceled");
                                                       return remove;
                                                   }),
                                    unified_queue_[i].end());
        }

        ARCHER_LOG_DEBUG(("Execute task " + task->DebugString()).c_str());

        ++active_tasks_;
        if (task->priority > 0) finemoe_revision::Ledger().Count("prefetch_queue_dequeued");
        lock.unlock();

        // All H2D transfers use this device's sole worker. Recheck space here,
        // immediately before physical copy, not on the earlier Python thread.
        if (!node->device.is_cuda() && task->dst_device.is_cuda()) {
            if (!node->is_sparse) RemoveCachedDenseNode(node);
            bool success = RemoveCachedSparseNode(node, gpu_id, task->priority > 0);
            if (!success) {
                ARCHER_LOG_DEBUG("{} evict failed, move to CPU", task->DebugString());
                if (task->priority > 0) {
                    finemoe_revision::Ledger().Count("prefetch_eviction_skip");
                } else {
                    // Do not manufacture throughput by silently exceeding the
                    // common memory budget or reusing a locked execution node.
                    std::fprintf(stderr, "FINEMOE_DEMAND_BUDGET_ERROR node=%zu bytes=%ld\n", node->id, node->byte_size);
                    std::abort();
                }
                FinishTask();
                continue;
            }
        }
        if (task->on_demand) {
            finemoe_revision::CompleteDemand(node->mutex, node->cv, node->state,
                                             [&] { SetNodeDevice(task); });
        } else {
            SetNodeDevice(task);
        }
        FinishTask();
    }
}

void ArcherTaskPool::FinishTask()
{
    std::lock_guard<std::mutex> lock(unified_mutex_);
    --active_tasks_;
    drain_cv_.notify_all();
}

void ArcherTaskPool::DrainForSnapshot()
{
    std::unique_lock<std::mutex> lock(unified_mutex_);
    const auto drained = [this] {
        if (active_tasks_ != 0) return false;
        for (const auto &queue : unified_queue_) if (!queue.empty()) return false;
        return true;
    };
    if (!drain_cv_.wait_for(lock, std::chrono::seconds(30), drained))
        throw std::runtime_error("FineMoE task drain timed out");
}

void ArcherTaskPool::SetNodeDevice(const TaskPtr& task)
{
    auto node = task->node;

    ARCHER_LOG_DEBUG("SetNodeDevice: task: {}, node: {}", task->DebugString(), node->str());
    if (!task->on_demand) {
        if (!node->mutex.try_lock()) {
            ARCHER_LOG_DEBUG("SetNodeDevice: task: {}, mutex locked", task->DebugString());
            return;
        }
    }

    if (node->device.type() == task->dst_device.type()) {
        ARCHER_LOG_DEBUG("SetNodeDevice: task: {}, skip same device", task->DebugString());
        if (!task->on_demand) node->mutex.unlock();
        return;
    }

    auto start_time = MCIROSECONDS_SINCE_EPOCH;

    // node->SetDevice(task->dst_device);
    node->SetDevice(task->dst_device, task->on_demand, nullptr, task->priority > 0);
    auto end_time = MCIROSECONDS_SINCE_EPOCH;
    ARCHER_LOG_DEBUG(
        "SetNodeDevice: task: {}, emplace time {} us", task->DebugString(), end_time - start_time);

    node->io_state = NODE_STATE_CACHED;

    if (task->priority > 0 && task->dst_device.is_cuda()) {
        auto node_body = kTopologyHandle->GetNodeBodyFromCorrID(node->corr_id);
        node_body->prefetch_cnt += 1;
        node->io_state = static_cast<NodeState>(node->io_state | NODE_STATE_PREFETCHED);
        node->last_prefetch_time = MCIROSECONDS_SINCE_EPOCH;
        ARCHER_LOG_DEBUG("Prefetch Node: task: {}, prefetch_cnt: {}",
                         task->DebugString(),
                         node_body->prefetch_cnt);
    }
    // Demand's CompleteDemand guard publishes readiness and releases the mutex;
    // the acquire waiter then reclaims it until ReleaseTensor finishes compute.
    if (!task->on_demand) node->mutex.unlock();
}

std::string ArcherTaskPool::DebugString(const std::vector<std::deque<TaskPtr>>& queue)
{
    std::stringstream ss;
    for (std::uint32_t i = 0; i < queue.size(); ++i) {
        ss << "priority " << i << " : ";
        for (auto task : queue[i]) {
            auto node = task->node;
            if (node == nullptr && task->remove_nodes.size() == 0) { continue; }
            if (node == nullptr) { node = task->remove_nodes[0]; }
```

## workloads/finemoe/deps/FineMoE-EuroSys26/core/prefetch/archer_prefetch_handle.h (line ranges 1,90)

```
// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#pragma once

#include "aio/archer_tensor_handle.h"
#include "parallel/expert_dispatcher.h"
#include "model/model_topology.h"

class ArcherPrefetchHandle {
public:
    ArcherPrefetchHandle(const std::string& prefix, const double device_memory_ratio);
    ~ArcherPrefetchHandle();

    bool IsTensorOffloaded(const std::uint32_t tensor_id);

    void AcquireTensor(std::uint64_t& request_id, torch::Tensor& buffer);
    void ReleaseTensor(std::uint64_t& request_id, torch::Tensor& buffer);
    void PrefetchTensors(std::uint64_t& request_id, const std::vector<std::uint32_t>& buffer);
    void FetchTensors(std::uint64_t& request_id, const std::vector<std::uint32_t>& buffer);

    void ReplaceCacheCandidates(const std::vector<std::uint32_t>& tensor_ids);
    void EnqueuePrefetch(const uint32_t tensor_id, int gpu_id, float prob);

    void OffloadTensor(torch::Tensor& tensor, const std::uint32_t tensor_id);
    void RegisterTensor(torch::Tensor& tensor, const std::uint32_t tensor_id);
    void RegisterModule(torch::nn::Module& module);
    void RegisterTensor(torch::Tensor* tensor);

    int GetNodeDefaultDevice(std::vector<std::uint32_t> tensor_ids) const;
    int GetNodeDevice(std::vector<std::uint32_t> tensor_ids) const;

    void SetTensorDevice(torch::Tensor& tensor, torch::Device device) const;

    torch::Tensor GetTrace();
    torch::Tensor GetHitRate();

    std::size_t GetAllDeviceBusyMemory();
    void ClearCache();

    void SetTrace(const torch::Tensor& trace);
    void TraceRequest(const std::uint64_t request_id, const TensorID tensor_id);
    void SetTopology(
        const std::vector<std::tuple<std::string, std::vector<std::vector<TensorID>>>>& topology);
    void UpdateTensorMap(std::uint64_t old_ptr, std::uint64_t new_ptr);
    bool IsTensorIndexInitialized() const;
    bool IsTensorOnDevice(const torch::Tensor& tensor) const;
    bool IsTensorOnDevice(const TensorID tensor_id) const;

    void CleanUpResources();

    // void SetNodeCachePriority(const std::uint64_t corr_id, const float priority);

private:
    std::string prefix_;
    std::unordered_map<std::size_t, std::unordered_set<std::uint32_t>> node_id_to_tensor_ids_;
    std::unordered_set<std::uint32_t> tensors_to_delete_;
    uint64_t last_layer_id_;
    NodePtr last_node_;
    bool has_cleaned_up_resources_;

    std::unordered_map<std::uint64_t, std::unordered_set<NodePtr>> request_id_to_nodes_;

    std::mutex mutex_;
};
```

## workloads/finemoe/deps/FineMoE-EuroSys26/core/prefetch/archer_prefetch_handle.cpp (line ranges 85,185)

```
static void SynchronizeNodeCompute(const NodePtr& node)
{
    if (!node->device.is_cuda()) return;
    // Python's post-forward hook runs after launch, not necessarily completion.
    // The main thread must finish its compute stream before releasing cache
    // residency to a speculative copy/eviction thread.
    const auto stream = c10::cuda::getCurrentCUDAStream(node->device.index());
    const auto error = cudaStreamSynchronize(stream.stream());
    if (error != cudaSuccess) {
        finemoe_revision::Ledger().Count("compute_release_sync_errors");
        std::fprintf(stderr, "FINEMOE_COMPUTE_RELEASE_ERROR node=%zu error=%s\n", node->id, cudaGetErrorString(error));
        std::abort();
    }
    finemoe_revision::Ledger().Count("compute_release_syncs");
}

void ArcherPrefetchHandle::AcquireTensor(std::uint64_t& request_id, torch::Tensor& buffer)
{
    auto tensor_id = kArcherTensorHandle->GetTensorId((void*)buffer.data_ptr());
    void* old_ptr = (void*)buffer.data_ptr();
    ARCHER_LOG_DEBUG("Acquire tensor ", tensor_id, old_ptr);

    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);

    // add node tensor_ids to node_id_to_tensor_ids_
    if (node_id_to_tensor_ids_.find(node->id) == node_id_to_tensor_ids_.end() ||
        node_id_to_tensor_ids_[node->id].size() == 0) {
        node_id_to_tensor_ids_[node->id] = std::unordered_set<std::uint32_t>();
        for (auto& tensor_id : node->tensor_ids) {
            node_id_to_tensor_ids_[node->id].insert(tensor_id);
        }

        auto node_body = kTopologyHandle->GetNodeBodyFromCorrID(node->corr_id);
        if (node->device.is_cuda()) { node_body->gpu_hit_cnt++; }

        // always lock node, wait for previous prefetch task to finish
        node->mutex.lock();
        std::unique_lock<std::mutex> lock(node->mutex, std::adopt_lock);
        node->state = 1;

        // Copy worker performs the authoritative budget check immediately
        // before H2D. No demand-side overflow allocation is permitted.
        const bool ready_at_acquire = node->device.is_cuda();
        kTaskPool->StartExec(request_id, node);
        node->cv.wait(lock, [node] { return node->state == 0; });
        if (node->is_sparse) finemoe_revision::Ledger().DemandUse(node->id, ready_at_acquire);
        // ReleaseTensor owns the matching unlock after all node tensors run.
        // Keep the lock across the forward, not merely this acquisition scope.
        lock.release();
    }

    kArcherTensorHandle->SetTensor(tensor_id, buffer);
    kArcherTensorHandle->UpdateTensorMap(old_ptr, (void*)buffer.data_ptr());
}
void ArcherPrefetchHandle::ReleaseTensor(std::uint64_t& request_id, torch::Tensor& buffer)
{
    auto tensor_id = kArcherTensorHandle->GetTensorId((void*)buffer.data_ptr());
    void* old_ptr = (void*)buffer.data_ptr();
    ARCHER_LOG_DEBUG("Release tensor ", tensor_id, old_ptr);

    auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
    // node->state = 1;

    if (node_id_to_tensor_ids_.find(node->id) == node_id_to_tensor_ids_.end()) {
        ARCHER_LOG_ERROR("Node not found in node_id_to_tensor_ids_", node->str());
        return;
    }

    /*  This needs to go after Release, default host can be changed in TraceRequest
     *   Faulty case: node -> default_host = cpu, node -> default_host = cuda; tensor already
     * released
     */
    // if (node != last_node_) {
    //     // kTaskPool->Prefetch(request_id, node);
    //     TraceRequest(request_id, tensor_id);
    // }
    // TraceRequest(request_id, tensor_id);

    auto current_layer_id = node->corr_id & 0xFFFFFFFF;
    if (current_layer_id != last_layer_id_ && node_id_to_tensor_ids_[last_node_->id].size() != 0) {
        node_id_to_tensor_ids_[last_node_->id].clear();
        SynchronizeNodeCompute(last_node_);
        kTaskPool->StopExec(request_id, last_node_);  // evict last node to cpu or disk
        last_node_->mutex.unlock();
    }
    last_layer_id_ = current_layer_id;
    last_node_ = node;

    node_id_to_tensor_ids_[node->id].erase(tensor_id);
    // ARCHER_LOG_DEBUG(
    //     "Node {} tensor_ids size {}", node->id, node_id_to_tensor_ids_[node->id].size());

    if (node_id_to_tensor_ids_[node->id].size() == 0) {
        SynchronizeNodeCompute(node);
        kTaskPool->StopExec(request_id, node);  // FIXME: change api to add request id
        // always unlock node here since, exec queue do not unlock automatically
        node->mutex.unlock();
    }

    if (kTopologyHandle->IsLastNode(node)) {
        ARCHER_LOG_DEBUG("Node is last, clean up", node->str());
```

## workloads/finemoe/deps/FineMoE-EuroSys26/core/model/model_topology.cpp (line ranges 61,159)

```
void Node::SetDevice(const torch::Device& target_device,
                     bool on_demand,
                     cudaStream_t stream,
                     bool speculative) noexcept
{
    ARCHER_LOG_DEBUG("SetDevice: " + str() + " to " + target_device.str());
    if (device == target_device) {
        ARCHER_LOG_DEBUG("SetDevice: " + str() + " to " + target_device.str() +
                         " but device is the same");
        return;
    }

    if (device.type() == target_device.type()) {
        ARCHER_LOG_WARN("SetDevice: " + str() + " to " + target_device.str() +
                        " but device type is the same");
        return;
    }

    if (kCudaStreamH2D == NULL) {
        auto cudaError = cudaStreamCreateWithFlags(&kCudaStreamH2D, cudaStreamNonBlocking);
        if (cudaError != cudaSuccess) {
            ARCHER_LOG_ERROR("cudaStreamCreate failed: {}", cudaGetErrorString(cudaError));
            exit(-1);
        }
    }

    if (target_device == DISK_DEVICE) {
        SetModuleDisk(tensor_ids);
        if (host_memory_ptr != nullptr) {
            kHostMemoryPool->FreeMemory(id, host_memory_ptr, byte_size, CPU_DEVICE);
            host_memory_ptr = nullptr;
        }
        if (device_memory_ptr != nullptr) {
            kDeviceMemoryPool->FreeMemory(id, device_memory_ptr, byte_size, device);
            device_memory_ptr = nullptr;
        }
    } else {
        // both are null, which means the node is not initialized
        if (host_memory_ptr == nullptr && device_memory_ptr == nullptr) {
            // int numa_id =
            //     default_device.index() / 4;  // TODO: 8 gpus, 2 numa nodes, so 4 gpus per numa
            host_memory_ptr = kHostMemoryPool->AllocateMemory(id, byte_size, CPU_DEVICE);
            assert(host_memory_ptr != nullptr);

            auto start_time = MCIROSECONDS_SINCE_EPOCH;
            SetModuleMemoryFromDisk(tensor_ids, host_memory_ptr, on_demand);
            auto end_time = MCIROSECONDS_SINCE_EPOCH;
            ARCHER_LOG_DEBUG("SetModuleMemoryFromDisk time: {} us", end_time - start_time);
        }

        if (target_device.is_cuda()) {
            // ARCHER_LOG_DEBUG("Allocate GPU Memory for node {}", this->id);
            device_memory_ptr = kDeviceMemoryPool->AllocateMemory(id, byte_size, target_device);
            // ARCHER_LOG_DEBUG("Allocate GPU Memory for node {} done", this->id);
            assert(device_memory_ptr != nullptr);
            assert(host_memory_ptr != nullptr);

            auto start_time = MCIROSECONDS_SINCE_EPOCH;
            const auto copy_id = is_sparse
                ? finemoe_revision::Ledger().Start(id, byte_size, speculative) : 0;
            cudaError_t copy_error;
            if (stream == nullptr) {
                copy_error = cudaMemcpy(device_memory_ptr, host_memory_ptr, byte_size, cudaMemcpyHostToDevice);
            } else {
                copy_error = cudaMemcpyAsync(
                    device_memory_ptr, host_memory_ptr, byte_size, cudaMemcpyHostToDevice, stream);
            }
            // Record completion only after the actual copy stream is complete.
            if (copy_error == cudaSuccess) copy_error = cudaStreamSynchronize(stream);
            if (copy_error != cudaSuccess) {
                finemoe_revision::Ledger().Count("prefetch_copy_errors");
                std::fprintf(stderr, "FINEMOE_COPY_ERROR node=%zu error=%s\n", id, cudaGetErrorString(copy_error));
                std::abort();
            }
            finemoe_revision::Ledger().Complete(copy_id);
            SetModuleCudaMemoryFromCPU(tensor_ids, device_memory_ptr, target_device);
            auto end_time = MCIROSECONDS_SINCE_EPOCH;
            ARCHER_LOG_DEBUG("SetModuleCudaMemoryFromCPU time: {} us", end_time - start_time);
        }

        if (target_device.is_cpu() && device.is_cuda()) {
            assert(host_memory_ptr != nullptr);
            auto start_time = MCIROSECONDS_SINCE_EPOCH;
            SetModuleMemoryFromCuda(tensor_ids, host_memory_ptr);
            kDeviceMemoryPool->FreeMemory(id, device_memory_ptr, byte_size, device);
            device_memory_ptr = nullptr;
            auto end_time = MCIROSECONDS_SINCE_EPOCH;
            ARCHER_LOG_DEBUG("SetModuleMemoryFromCuda time: {} us", end_time - start_time);
        }
    }
    if (is_sparse && device.is_cuda() && !target_device.is_cuda())
        finemoe_revision::Ledger().Evict(id);
    device = target_device;
}

ArcherTopologyHandle::ArcherTopologyHandle() {}

NodePtrList ArcherTopologyHandle::GetLFUNodes(const torch::Device& device)
{
```

## workloads/finemoe/deps/FineMoE-EuroSys26/core/model/model_topology.h (line ranges 20,85)

```
enum NodeState {
    NODE_STATE_NONE = 0x0,
    NODE_STATE_CACHED = 0x1,
    NODE_STATE_PREFETCHED = 0x2,
    NODE_STATE_VISITED = 0x4,
};

extern cudaStream_t kCudaStreamH2D;

struct Node {
    std::vector<TensorID> tensor_ids;
    std::int64_t byte_size;
    std::size_t last_access_time;
    std::size_t last_prefetch_time = 0;

    std::size_t id;
    std::size_t corr_id;

    torch::Device device = DISK_DEVICE;
    torch::Device default_device = DEFAULT_CUDA_DEVICE;  // FIXME: should be set by scheduler
    torch::Device default_host = CPU_DEVICE;
    torch::Device initial_host = DISK_DEVICE;

    std::atomic_uint8_t state{0};  // 0 for ready, 1 for moving

    std::mutex mutex;
    std::condition_variable cv;

    float cache_priority = 0.0;
    std::uint64_t visit_count = 0;
    std::uint64_t incache_visit_count = 0;
    std::uint64_t unused_count = 0;
    float prob = 0;
    bool is_sparse = false;
    NodeState io_state = NODE_STATE_NONE;

    bool is_overflow = false;

    void* host_memory_ptr = nullptr;
    void* device_memory_ptr = nullptr;

public:
    explicit Node();
    const std::string str() noexcept;
    void SetDevice(const torch::Device& target_device,
                   bool ondemand = false,
                   cudaStream_t stream = nullptr,
                   bool speculative = false) noexcept;
};

typedef std::shared_ptr<Node> NodePtr;
typedef std::vector<NodePtr> NodePtrList;
typedef std::tuple<std::int64_t, NodePtrList> FilterResult;

struct NodeBody;
typedef std::shared_ptr<NodeBody> NodeBodyPtr;

struct NodeBody {
    NodePtr node;
    std::vector<NodeBodyPtr> children;
    std::vector<std::size_t> children_visit_cnt;
    std::unordered_set<std::size_t> activate_request;
    std::size_t prefetch_cnt = 0;
    std::size_t visit_cnt = 0;
    std::size_t cpu_visit_cnt = 0;
    std::size_t gpu_visit_cnt = 0;
```

## workloads/finemoe/deps/FineMoE-EuroSys26/finemoe/models/modeling_qwen/modeling_qwen2_moe.py (line ranges 842,926;1233,1245)

```
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        expert_probs = routing_weights.detach()
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        expert_index = selected_experts.reshape(
            batch_size, sequence_length, self.top_k)

        # Record states
        for i, seq_id in enumerate(self.seq_id_list):
            self.expert_tracer.update_entry(
                seq_id=seq_id,
                expert_list=expert_index[i],
                layer_idx=self.layer_id,
                hidden_states=hidden_states[i *
                                            sequence_length:(i+1)*sequence_length],
                expert_probs=expert_probs[i *
                                          sequence_length:(i+1)*sequence_length],
            )

        input_trajs = []
        # A completed request may remain in the tracing archive. Match only
        # the active sequence IDs, in their current batch order.
        for seq_id in self.seq_id_list:
            trace_entry = self.expert_tracer.trace[seq_id]
            chosen_iters = trace_entry.iters[-1:]
            input_trajs.append(torch.stack(
                [it["probs"][:self.layer_id+1] for it in chosen_iters], dim=0))

        input_trajs = torch.stack(input_trajs, dim=0).to(
            dtype=torch.float, device=self.device, non_blocking=True
        )
        assert not (input_trajs == 0).all(
            dim=-1).any(), f"Layer with all-zero expert probs exist"

        assert self.expert_map_matcher is not None
        for seq_id, trajs in zip(self.seq_id_list, input_trajs):
            self.expert_map_matcher.traj_prefetch(seq_id, trajs)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            # Even a zero-row invocation runs offload pre-hooks. Do not turn
            # unrouted experts into demand loads or apparent prefetch hits.
            if top_x.numel() == 0:
                continue

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state).to(
                routing_weights.device) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(
            self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        for seq_id, embeds in zip(self.seq_id_list, inputs_embeds):
            self.expert_tracer.update_embed(seq_id, embeds)

        assert self.expert_map_matcher is not None
        for seq_id, embeds in zip(self.seq_id_list, inputs_embeds):
            self.expert_map_matcher.embed_prefetch(seq_id, embeds)

```

## workloads/finemoe/deps/FineMoE-EuroSys26/finemoe/runtime/model_offload.py (line ranges 345,395;674,719)

```
    def init_expert_map_matcher(self):
        self.expert_map_matcher = ExpertMapMatcher(
            expert_tracer=self.expert_tracer,
            expert_map_store=self.expert_map_store,
            expert_prefetcher=self.expert_prefetcher,
            prefetch_distance=self.prefetch_distance,
        )

    def init(
        self, cls: Type[PreTrainedModel], ar_config: Union[str, Dict, ArcherConfig]
    ):

        self.cls = cls
        self.name_id_map = {}
        self.tensor_id_map = {}
        self.registered_tensors = set()
        self.forward_hooks = []
        self.backward_hooks = []

        self.offload_set = set()

        if isinstance(ar_config, str):
            _archer_config = ArcherConfig.load_from_file(ar_config)
        elif isinstance(ar_config, dict):
            _archer_config = ArcherConfig.load_from_json(ar_config)
        elif isinstance(ar_config, ArcherConfig):
            _archer_config = ar_config
        else:
            raise ValueError(
                "ArcherConfig is not provided. Please provide a path to a config file or a dict."
            )

        # TODO: get trace from trace_path

        self.checkpoint = _archer_config.offload_path

        os.makedirs(self.checkpoint, exist_ok=True)

        self.prefetch_lib = PrefetchBuilder().load() if use_jit else prefetch_op
        self.archer_engine = self.prefetch_lib.prefetch_handle(
            self.checkpoint,
            _archer_config.device_memory_ratio,
        )

        self.archer_config = _archer_config

        self.expert_tracer.offload_engine = self

        return self

    def __enter__(self):
                self.expert_tensor_map = dict()
                for name, id in self.name_id_map.items():
                    layer_id, expert_id = parse_expert_id(name, self.config)
                    if expert_id is not None:
                        self.expert_tensor_map[(layer_id, expert_id)] = id

                self.expert_prefetcher.expert_tensor_map = self.expert_tensor_map

                self.init_expert_map_matcher()

                model.model.expert_prefetcher = self.expert_prefetcher
                model.model.expert_tracer = self.expert_tracer
                model.model.expert_map_matcher = self.expert_map_matcher
                model.model._device = self.device

                module_idx = 0
                self.expert_layer_modules = []
                for module in model.modules():
                    if isinstance(module, SyncQwen2MoeSparseMoeBlock):
                        # module.archer_prefetch = self.archer_prefetch
                        # module.archer_tracer = self.archer_tracer
                        module.archer_engine = self.archer_engine
                        module.archer_config = self.archer_config
                        # module.expert_dispatcher = self.expert_dispatcher
                        self.expert_modules.append(module)
                        # module.expert_executor = self.expert_executor
                        module.expert_prefetcher = self.expert_prefetcher
                        module.expert_tracer = self.expert_tracer
                        module.expert_map_matcher = self.expert_map_matcher
                        module.expert_tensor_map = self.expert_tensor_map
                        module.prefetch_distance = self.prefetch_distance
                        module.device = self.device

                        self.expert_layer_modules.append(module)

                        module.layer_id = module_idx

                        module_idx += 1

                        self.moe_layers.append(module)
                        module.moe_layers = self.moe_layers

                    if isinstance(module, Qwen2MoeMLP):
                        module.offload_engine = self

                self.setup_archer_hooks(model)
```

## workloads/finemoe/deps/FineMoE-EuroSys26/core/python/py_archer_prefetch.cpp (line ranges 45,114)

```
        return result;
    });
    py::class_<ArcherPrefetchHandle>(m, "prefetch_handle")
        .def(py::init<const std::string&, const double>())

        .def("offload", &ArcherPrefetchHandle::OffloadTensor)
        .def("register",
             (void(ArcherPrefetchHandle::*)(torch::Tensor&, const std::uint32_t)) &
                 ArcherPrefetchHandle::RegisterTensor)
        //    .def("register",
        //         (void(ArcherPrefetchHandle::*)(torch::nn::Module&)) &
        //             ArcherPrefetchHandle::RegisterModule)
        .def("register",
             (void(ArcherPrefetchHandle::*)(torch::Tensor*)) & ArcherPrefetchHandle::RegisterTensor)
        .def("set_tensor_device",
             (void(ArcherPrefetchHandle::*)(torch::Tensor&, torch::Device)) &
                 ArcherPrefetchHandle::SetTensorDevice)
        // .def("begin", (void (ArcherPrefetchHandle::*)(torch::nn::Module&))
        // &ArcherPrefetchHandle::AcquireTensor) .def("end", (void
        // (ArcherPrefetchHandle::*)(torch::nn::Module&)) &ArcherPrefetchHandle::ReleaseTensor)
        .def("begin",
             (void(ArcherPrefetchHandle::*)(std::uint64_t&, torch::Tensor&)) &
                 ArcherPrefetchHandle::AcquireTensor)
        .def("end",
             (void(ArcherPrefetchHandle::*)(std::uint64_t&, torch::Tensor&)) &
                 ArcherPrefetchHandle::ReleaseTensor)
        // .def("begin",
        //      (void (ArcherPrefetchHandle::*)(torch::Tensor&, const std::uint32_t)) &
        //          ArcherPrefetchHandle::AcquireTensor)
        // .def("end",
        //      (void (ArcherPrefetchHandle::*)(torch::Tensor&, const std::uint32_t)) &
        //          ArcherPrefetchHandle::ReleaseTensor)
        //    .def("get_trace",
        //    (torch::Tensor(ArcherPrefetchHandle::*)()) & ArcherPrefetchHandle::GetTrace)
        .def("get_hit_rate",
             (torch::Tensor(ArcherPrefetchHandle::*)()) & ArcherPrefetchHandle::GetHitRate)
        .def("get_all_device_busy_memory",
             (std::size_t(ArcherPrefetchHandle::*)()) & ArcherPrefetchHandle::GetAllDeviceBusyMemory)
        .def("clear_cache",
             (void(ArcherPrefetchHandle::*)()) & ArcherPrefetchHandle::ClearCache)
        .def("set_trace",
             (void(ArcherPrefetchHandle::*)(const torch::Tensor&)) & ArcherPrefetchHandle::SetTrace)
        //    .def("trace_request",
        //         (void(ArcherPrefetchHandle::*)(const std::uint64_t, const std::uint32_t)) &
        //             ArcherPrefetchHandle::TraceRequest)
        .def("set_topology",
             (void(ArcherPrefetchHandle::*)(
                 const std::vector<std::tuple<std::string, std::vector<std::vector<TensorID>>>>&)) &
                 ArcherPrefetchHandle::SetTopology)
        .def("update_tensor_map",
             (void(ArcherPrefetchHandle::*)(std::uint64_t, std::uint64_t)) &
                 ArcherPrefetchHandle::UpdateTensorMap)
        .def("is_tensor_offloaded", &ArcherPrefetchHandle::IsTensorOffloaded)
        .def("is_tensor_index_initialized", &ArcherPrefetchHandle::IsTensorIndexInitialized)
        .def("is_tensor_on_device",
             (bool(ArcherPrefetchHandle::*)(const torch::Tensor&) const) &
                 ArcherPrefetchHandle::IsTensorOnDevice)
        .def("is_tensor_on_device",
             (bool(ArcherPrefetchHandle::*)(const std::uint32_t) const) &
                 ArcherPrefetchHandle::IsTensorOnDevice)
        .def("get_node_default_device", &ArcherPrefetchHandle::GetNodeDefaultDevice)
        .def("get_node_device", &ArcherPrefetchHandle::GetNodeDevice)
        .def("prefetch_tensors", &ArcherPrefetchHandle::PrefetchTensors)
        .def("replace_cache_candidates", &ArcherPrefetchHandle::ReplaceCacheCandidates)
        .def("enqueue_prefetch", &ArcherPrefetchHandle::EnqueuePrefetch)
        .def("fetch_tensors", &ArcherPrefetchHandle::FetchTensors)
        .def("clean_up_resources", &ArcherPrefetchHandle::CleanUpResources);
     //    .def("set_node_cache_priority", &ArcherPrefetchHandle::SetNodeCachePriority);

    py::class_<ExpertDispatcher>(m, "expert_dispatcher")
```

## workloads/finemoe/deps/FineMoE-EuroSys26/op_builder/prefetch.py (line ranges 27,58;76,90)

```

    def absolute_name(self):
        return f'finemoe.ops.prefetch.{self.NAME}_op'

    def sources(self):
        return [
            'core/utils/archer_logger.cpp',
            'core/utils/cuda_utils.cpp',
            'core/model/model_topology.cpp',
            'core/prefetch/archer_prefetch_handle.cpp',
            'core/prefetch/task_scheduler.cpp',
            'core/prefetch/task_thread.cpp',
            'core/memory/memory_pool.cpp',
            'core/memory/stream_pool.cpp',
            'core/memory/host_caching_allocator.cpp',
            'core/python/py_archer_prefetch.cpp',
            'core/parallel/expert_dispatcher.cpp',
            'core/parallel/expert_module.cpp',
            'core/aio/archer_aio_thread.cpp',
            'core/aio/archer_prio_aio_handle.cpp',
            'core/aio/archer_aio_utils.cpp',
            'core/aio/archer_aio_threadpool.cpp',
            'core/aio/archer_tensor_handle.cpp',
            'core/aio/archer_tensor_index.cpp',
        ]

    def include_paths(self):
        return ['core']

    def cxx_args(self):
        # -O0 for improved debugging, since performance is bound by I/O
        CPU_ARCH = self.cpu_arch()
            '-lpthread',
        ]

    def extra_ldflags(self):
        return []

    def is_compatible(self, verbose=True):
        return super().is_compatible(verbose)
```

## workloads/finemoe/finemoe_runtime_safety.h (line ranges 1,50)

```
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
```


