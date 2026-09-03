// Included only by the private task_scheduler.cpp after its normal headers.
// Single Python inference, GPU 0, and the existing sole GPU copy worker.
namespace {
void EbRequire(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
}

void ArcherTaskPool::ConfigureExpertBuffering(const std::string &arm, eb_u32 capacity,
                                             const std::string &library,
                                             const std::string &bytecode)
{
    DrainForSnapshot();
    std::lock_guard<std::mutex> guard(eb_mutex_);
    EbRequire(!eb_state_, "EB configure is single-use");
    EbRequire(GetDeviceCount() == 1, "EB private adapter supports one GPU");
    eb_adapter::Arm selected;
    if (arm == "fifo") selected = eb_adapter::Arm::Fifo;
    else if (arm == "native") selected = eb_adapter::Arm::Native;
    else if (arm == "bpf") selected = eb_adapter::Arm::Bpf;
    else throw std::runtime_error("EB arm must be fifo/native/bpf");
    for (const auto &node : kTopologyHandle->GetSparseNodes()) {
        std::unique_lock<std::mutex> lock(node->mutex, std::try_to_lock);
        EbRequire(lock.owns_lock() && !node->device.is_cuda(), "EB requires cold unowned experts");
    }
    eb_state_ = std::make_unique<eb_adapter::State>(selected, capacity, library, bytecode);
}

void ArcherTaskPool::CheckExpertBufferingResidency(eb_u32 layer, const NodePtrList &nodes)
{
    const auto &cohort = eb_state_->Get(layer);
    EbRequire(nodes.size() == cohort.nodes.size(), "EB cohort length changed");
    for (eb_u32 i = 0; i < nodes.size(); ++i) {
        const auto &node = nodes[i];
        std::unique_lock<std::mutex> lock(node->mutex, std::try_to_lock);
        EbRequire(lock.owns_lock(), "EB cohort node still owned");
        EbRequire(node->id == cohort.nodes[i] && node->state.load() == 0,
                  "EB cohort node identity or in-flight state changed");
        const bool expected = cohort.input.experts[i].flags & EB_RESIDENT;
        EbRequire(node->device.is_cuda() == expected &&
                  (!expected || (node->device.index() == 0 && node->device_memory_ptr && node->host_memory_ptr)),
                  "EB metadata differs from whole-expert residency");
    }
}

eb_u64 ArcherTaskPool::BeginExpertBuffering(eb_u32 layer, eb_u32 device,
                                           const std::vector<std::uint32_t> &tensor_ids,
                                           const std::vector<eb_u32> &counts)
{
    DrainForSnapshot();
    std::lock_guard<std::mutex> guard(eb_mutex_);
    EbRequire(eb_state_ != nullptr, "EB not configured");
    {
        std::lock_guard<std::mutex> executing(exec_mutex_);
        for (const auto &item : exec_queue_)
            EbRequire(!item.second->node->is_sparse, "EB prior expert still executing");
    }
    NodePtrList nodes;
    std::vector<eb_adapter::NodeId> ids;
    for (auto tid : tensor_ids) {
        auto node = kTopologyHandle->GetNodeFromTensorID(tid);
        EbRequire(node && node->is_sparse && node->byte_size > 0 &&
                  node->default_device.is_cuda() && node->default_device.index() == 0 &&
                  node->default_host.is_cpu(), "EB requires complete GPU0 experts with CPU backing");
        nodes.push_back(node);
        ids.push_back(node->id);
    }
    const auto epoch = eb_state_->Begin(layer, device, ids, counts);
    auto previous = eb_nodes_.find(layer);
    EbRequire(previous == eb_nodes_.end() || previous->second == nodes, "EB physical cohort changed");
    eb_nodes_[layer] = nodes;
    CheckExpertBufferingResidency(layer, nodes);
    return epoch;
}

void ArcherTaskPool::EndExpertBuffering(eb_u32 layer, eb_u64 epoch)
{
    DrainForSnapshot();
    std::lock_guard<std::mutex> guard(eb_mutex_);
    EbRequire(eb_state_ != nullptr, "EB not configured");
    {
        std::lock_guard<std::mutex> executing(exec_mutex_);
        for (const auto &item : exec_queue_)
            EbRequire(!item.second->node->is_sparse, "EB expert still executing at batch end");
    }
    CheckExpertBufferingResidency(layer, eb_nodes_.at(layer));
    eb_state_->End(layer, epoch);
}

void ArcherTaskPool::StampExpertBufferingTask(const TaskPtr &task)
{
    // AcquireTensor owns this incoming node mutex before entering StartExec.
    std::lock_guard<std::mutex> guard(eb_mutex_);
    const auto location = eb_state_->Locate(task->node->id);
    task->eb_layer = location.first;
    task->eb_expert = location.second;
    task->eb_epoch = eb_state_->ActiveEpoch(task->eb_layer, task->eb_expert);
    const auto &input = eb_state_->Get(task->eb_layer).input;
    const bool expected = input.experts[task->eb_expert].flags & EB_RESIDENT;
    EbRequire(task->node->device.is_cuda() == expected &&
              (!expected || task->node->device.index() == 0), "EB demand residency changed");
    if (expected) {
        auto ctx = eb_state_->Decide(task->eb_layer, task->eb_expert,
                                    std::vector<bool>(input.count, false));
        EbRequire(ctx.output.status == EB_HIT, "EB resident demand did not return hit");
    }
}

std::int64_t ArcherTaskPool::ExpertBufferingResidentBytes() const
{
    std::int64_t bytes = 0;
    for (const auto &layer : eb_nodes_) {
        const auto &input = eb_state_->Get(layer.first).input;
        for (eb_u32 i = 0; i < input.count; ++i)
            if (input.experts[i].flags & EB_RESIDENT) bytes += layer.second[i]->byte_size;
    }
    return bytes;
}

bool ArcherTaskPool::RemoveExpertBufferingNode(const TaskPtr &task)
{
    std::lock_guard<std::mutex> guard(eb_mutex_);
    const auto device = torch::Device(torch::kCUDA, 0);
    EbRequire(task->dst_device == device, "EB copy destination is not GPU0");
    const auto &incoming = task->node;
    if (!incoming->is_sparse) {
        // Do not invent cross-layer victims to compensate for a bad K choice.
        return ExpertBufferingResidentBytes() <=
                   kTopologyHandle->GetSparseCacheLimit(device, incoming->byte_size) &&
               kDeviceMemoryPool->GetFreeMemory(device) >= incoming->byte_size;
    }
    EbRequire(task->on_demand && task->priority == 0 && task->eb_epoch &&
              eb_state_->ActiveEpoch(task->eb_layer, task->eb_expert) == task->eb_epoch,
              "EB unstamped, speculative, or stale demand");
    auto &nodes = eb_nodes_.at(task->eb_layer);
    EbRequire(nodes.at(task->eb_expert) == incoming, "EB incoming node changed");
    const auto &input = eb_state_->Get(task->eb_layer).input;
    EbRequire(!(input.experts[task->eb_expert].flags & EB_RESIDENT), "EB miss became resident");
    std::vector<std::unique_lock<std::mutex>> locks;
    std::vector<bool> eligible(nodes.size(), false);
    // Metadata -> nonblocking node locks cannot cycle with acquire's node ->
    // metadata order. The incoming demand waiter may not yet have entered wait;
    // do not race it for its mutex here. CompleteDemand owns it before the copy.
    for (eb_u32 i = 0; i < nodes.size(); ++i) {
        locks.emplace_back(nodes[i]->mutex, std::defer_lock);
        if (i != task->eb_expert) (void)locks.back().try_lock();
    }
    std::lock_guard<std::mutex> executing(exec_mutex_);
    for (eb_u32 i = 0; i < nodes.size(); ++i) {
        if (!locks[i].owns_lock()) continue;
        const bool expected = input.experts[i].flags & EB_RESIDENT;
        EbRequire(nodes[i]->device.is_cuda() == expected &&
                  (!expected || (nodes[i]->device.index() == 0 && nodes[i]->device_memory_ptr)),
                  "EB candidate physical residency changed");
        eligible[i] = expected && nodes[i]->state.load() == 0 && exec_queue_.count(nodes[i]->id) == 0;
    }
    auto ctx = eb_state_->Decide(task->eb_layer, task->eb_expert, eligible);
    if (ctx.output.status == EB_BLOCKED) return false;
    EbRequire(ctx.output.status == EB_ADMIT || ctx.output.status == EB_EVICT, "EB invalid miss decision");
    eb_state_->Validate(ctx);
    if (ctx.output.status == EB_EVICT) {
        const auto victim = ctx.output.victim;
        auto &node = nodes.at(victim);
        EbRequire(locks[victim].owns_lock() && eligible[victim] &&
                  node->device == device && node->state.load() == 0 &&
                  exec_queue_.count(node->id) == 0, "EB victim lost live eligibility");
        // Epoch/metadata, victim ownership, and exec membership stay locked
        // through the actual whole-node eviction; no Python callback occurs.
        node->SetDevice(node->default_host);
        EbRequire(node->device.is_cpu() && !node->device_memory_ptr, "EB eviction did not complete");
        eb_state_->Evicted(ctx);
    }
    eb_state_->CanAdmit(task->eb_layer, task->eb_epoch, task->eb_expert);
    return ExpertBufferingResidentBytes() + incoming->byte_size <=
               kTopologyHandle->GetSparseCacheLimit(device) &&
           kDeviceMemoryPool->GetFreeMemory(device) >= incoming->byte_size;
}

void ArcherTaskPool::CopyExpertBufferingNode(const TaskPtr &task)
{
    // CompleteDemand owns the incoming node mutex. A metadata holder only
    // TRY_LOCKs nodes; no blocking metadata -> node edge can form a cycle.
    std::lock_guard<std::mutex> guard(eb_mutex_);
    const auto &node = task->node;
    EbRequire(task->on_demand && !task->priority && task->dst_device.is_cuda() &&
              task->dst_device.index() == 0 && !node->device.is_cuda(),
              "EB invalid physical admission task");
    eb_state_->CanAdmit(task->eb_layer, task->eb_epoch, task->eb_expert);
    node->SetDevice(task->dst_device, true, nullptr, false);
    EbRequire(node->device == task->dst_device && node->device_memory_ptr && node->host_memory_ptr,
              "EB whole-expert copy did not publish residency");
    // Node::SetDevice has checked and synchronized the complete H2D. Record
    // before CompleteDemand publishes ready; never infer completion from enqueue.
    eb_state_->Admitted(task->eb_layer, task->eb_epoch, task->eb_expert);
}

std::map<std::string, eb_u64> ArcherTaskPool::ExpertBufferingStats()
{
    std::lock_guard<std::mutex> guard(eb_mutex_);
    EbRequire(eb_state_ != nullptr, "EB not configured");
    for (const auto &layer : eb_nodes_) CheckExpertBufferingResidency(layer.first, layer.second);
    const auto stats = eb_state_->Stats();
    return {{"decisions", stats.decisions}, {"jit_calls", stats.jit_calls},
            {"admissions", stats.admissions}, {"evictions", stats.evictions},
            {"resident_sparse_bytes", static_cast<eb_u64>(ExpertBufferingResidentBytes())},
            {"registered_layers", eb_nodes_.size()}};
}
