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
