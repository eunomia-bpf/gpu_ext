// SPDX-License-Identifier: GPL-2.0
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include "bpftime_hpf.h"
#include "xsched/sched/policy/hpf.h"

using namespace xsched::sched;

int main()
{
    HighestPriorityFirstPolicy original;
    // The test links the same wrapped factory as xserver-bpftime.
    auto bpf = CreatePolicy(kPolicyHighestPriorityFirst);
    std::map<XQueueHandle, bool> expected, actual;
    original.SetSuspendFunc([&](XQueueHandle q) { expected[q] = true; });
    original.SetResumeFunc([&](XQueueHandle q) { expected[q] = false; });
    bpf->SetSuspendFunc([&](XQueueHandle q) { actual[q] = true; });
    bpf->SetResumeFunc([&](XQueueHandle q) { actual[q] = false; });
    const Priority priorities[] = {-999, PRIORITY_MIN, -1, 0, 1, PRIORITY_MAX, 999};
    for (unsigned int i = 0; i < BPFTIME_HPF_MAX_QUEUES; ++i) {
        if (i % 8 == 0) continue; // Includes missing hints/default priority.
        auto hint = std::make_shared<PriorityHint>(i + 1, priorities[i % 7]);
        original.RecvHint(hint);
        bpf->RecvHint(hint);
    }
    auto irrelevant = std::make_shared<TimesliceHint>(1000);
    original.RecvHint(irrelevant);
    bpf->RecvHint(irrelevant);

    std::minstd_rand random(20260902);
    uint64_t decisions = 0;
    for (unsigned int trial = 0; trial < 5000; ++trial) {
        Status status;
        const unsigned int count = trial < 65 ? trial : random() % 65;
        for (unsigned int i = 0; i < count; ++i) {
            auto queue = std::make_unique<XQueueStatus>();
            queue->handle = i + 1;
            queue->device = random() % 5;
            queue->ready = trial % 7 == 0 ? false : random() % 2;
            queue->suspended = random() % 2;
            status.xqueue_status.emplace(i + 1, std::move(queue));
        }
        expected.clear();
        actual.clear();
        original.Sched(status);
        bpf->Sched(status);
        if (actual != expected || actual.size() != count)
            throw std::runtime_error("BPF decision differs from upstream HPF");
        decisions += count;
    }
    Status oversized;
    for (unsigned int i = 0; i <= BPFTIME_HPF_MAX_QUEUES; ++i)
        oversized.xqueue_status.emplace(i + 1, std::make_unique<XQueueStatus>());
    bool rejected = false;
    try { bpf->Sched(oversized); }
    catch (const std::runtime_error &) { rejected = true; }
    if (!rejected) throw std::runtime_error("oversized snapshot was not rejected");
    std::cout << "{\"test\":\"upstream_hpf_vs_bpftime_jit\",\"snapshots\":5000,"
              << "\"queue_decisions\":" << decisions
              << ",\"max_queues\":64,\"devices\":5,\"oversize_rejected\":true,"
              << "\"mismatches\":0}" << std::endl;
}
