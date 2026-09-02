// SPDX-License-Identifier: GPL-2.0
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <unordered_map>

#include "ebpf-vm.h"
#include "bpftime_hpf.h"
#include "xsched/sched/policy/policy.h"

using namespace xsched::sched;

class BpftimeHPF final : public Policy {
    std::unique_ptr<ebpf_vm, decltype(&ebpf_destroy)> vm_{nullptr, ebpf_destroy};
    ebpf_jit_fn execute_ = nullptr;
    std::unordered_map<XQueueHandle, Priority> priorities_;
    uint64_t calls_ = 0, queues_ = 0, suspended_ = 0;

public:
    BpftimeHPF() : Policy(kPolicyHighestPriorityFirst)
    {
        const char *path = std::getenv("GPUBPF_HPF_CODE");
        if (!path) throw std::runtime_error("GPUBPF_HPF_CODE must name HPF bytecode");
        std::ifstream input(path, std::ios::binary);
        std::vector<char> code{std::istreambuf_iterator<char>(input), {}};
        if (!input.is_open() || code.empty() || code.size() > 65536)
            throw std::runtime_error("invalid HPF bytecode file");
        vm_.reset(ebpf_create("ubpf"));
        char *error = nullptr;
        if (ebpf_load(vm_.get(), code.data(), code.size(), &error) != 0) {
            const std::string message = error ? error : "HPF load failed";
            std::free(error);
            throw std::runtime_error(message);
        }
        execute_ = ebpf_compile(vm_.get(), &error);
        if (!execute_) {
            const std::string message = error ? error : "HPF JIT failed";
            std::free(error);
            throw std::runtime_error(message);
        }
        std::cout << "bpftime_hpf_ready: backend=ubpf-jit max_queues="
                  << BPFTIME_HPF_MAX_QUEUES << std::endl;
    }

    ~BpftimeHPF() override
    {
        std::cout << "bpftime_hpf_stats: calls=" << calls_ << " queues=" << queues_
                  << " suspend=" << suspended_ << " resume=" << queues_ - suspended_
                  << std::endl;
    }

    void RecvHint(std::shared_ptr<const Hint> hint) override
    {
        if (hint->Type() != kHintTypePriority) return;
        auto priority = std::dynamic_pointer_cast<const PriorityHint>(hint);
        if (!priority) throw std::runtime_error("invalid priority hint");
        priorities_[priority->Handle()] = std::clamp(priority->Prio(), PRIORITY_MIN, PRIORITY_MAX);
    }

    void Sched(const Status &status) override
    {
        if (status.xqueue_status.size() > BPFTIME_HPF_MAX_QUEUES)
            throw std::runtime_error("HPF snapshot exceeds declared 64-queue limit");
        bpftime_hpf_snapshot snapshot{};
        for (const auto &entry : status.xqueue_status) {
            const auto &queue = *entry.second;
            const auto priority = priorities_.find(queue.handle);
            snapshot.queues[snapshot.count++] = {
                queue.device, priority == priorities_.end() ? PRIORITY_DEFAULT : priority->second,
                static_cast<unsigned int>(queue.ready),
            };
        }
        const uint64_t mask = execute_(&snapshot, sizeof(snapshot));
        unsigned int index = 0;
        for (const auto &entry : status.xqueue_status) {
            if ((mask >> index++) & 1) {
                Suspend(entry.second->handle);
                ++suspended_;
            } else {
                Resume(entry.second->handle);
            }
        }
        ++calls_;
        queues_ += snapshot.count;
    }
};

// Relink the existing upstream server objects; do not patch the vendor tree.
// GNU ld wraps only the policy factory. Queue execution and IPC stay upstream.
extern "C" std::unique_ptr<Policy> __real__ZN6xsched5sched12CreatePolicyE11XPolicyType(XPolicyType);
extern "C" std::unique_ptr<Policy> __wrap__ZN6xsched5sched12CreatePolicyE11XPolicyType(XPolicyType type)
{
    if (type == kPolicyHighestPriorityFirst) return std::make_unique<BpftimeHPF>();
    return __real__ZN6xsched5sched12CreatePolicyE11XPolicyType(type);
}
