// SPDX-License-Identifier: Apache-2.0
#include "adapter_state.h"
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <set>
#include <stdexcept>

namespace eb_adapter {
namespace {
void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
template <class T> T Symbol(void *library, const char *name) {
    auto symbol = dlsym(library, name);
    Require(symbol != nullptr, "EB selector symbol missing");
    return reinterpret_cast<T>(symbol);
}
eb_u32 ResidentCount(const eb_input &input) {
    eb_u32 count = 0;
    for (eb_u32 i = 0; i < input.count; ++i)
        count += !!(input.experts[i].flags & EB_RESIDENT);
    return count;
}
} // namespace

State::State(Arm arm, eb_u32 capacity, const std::string &library,
             const std::string &bytecode) : arm_(arm), capacity_(capacity) {
    Require(arm == Arm::Fifo || arm == Arm::Native || arm == Arm::Bpf, "EB invalid arm");
    Require(capacity > 0 && capacity <= EB_MAX_EXPERTS, "EB invalid capacity");
    library_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    Require(library_ != nullptr, "EB selector library load failed");
    try {
        native_ = Symbol<decltype(native_)>(library_, "eb_select");
        if (arm == Arm::Bpf) {
            auto open = Symbol<void *(*)(const char *, char *, size_t)>(library_, "eb_jit_open");
            bpf_ = Symbol<decltype(bpf_)>(library_, "eb_jit_select");
            jit_calls_ = Symbol<decltype(jit_calls_)>(library_, "eb_jit_calls");
            close_ = Symbol<decltype(close_)>(library_, "eb_jit_close");
            char message[512]{};
            jit_ = open(bytecode.c_str(), message, sizeof(message));
            if (!jit_) throw std::runtime_error(std::string("EB JIT open: ") + message);
        }
    } catch (...) {
        dlclose(library_);
        throw;
    }
}

State::~State() {
    if (jit_) close_(jit_);
    if (library_) dlclose(library_);
}

eb_u64 State::NextSerial(eb_u64 serial) {
    Require(serial != std::numeric_limits<eb_u64>::max(), "EB serial exhausted");
    return serial + 1;
}

eb_u64 State::Begin(eb_u32 layer, eb_u32 device, const std::vector<NodeId> &nodes,
                    const Counts &counts) {
    Require(!active_, "EB overlapping layer invocation");
    Require(device == 0 && !nodes.empty() && nodes.size() <= EB_MAX_EXPERTS &&
            counts.size() == nodes.size() && capacity_ <= nodes.size(), "EB invalid cohort");
    Require(std::set<NodeId>(nodes.begin(), nodes.end()).size() == nodes.size(),
            "EB duplicate whole-expert node");
    auto found = layers_.find(layer);
    if (found == layers_.end()) {
        for (auto node : nodes)
            Require(locations_.count(node) == 0, "EB node belongs to another layer");
        Cohort cohort;
        cohort.nodes = nodes;
        cohort.input.abi_version = EB_ABI_VERSION;
        cohort.input.count = nodes.size();
        cohort.input.capacity = capacity_;
        cohort.input.layer_id = layer;
        cohort.input.device_id = device;
        found = layers_.emplace(layer, std::move(cohort)).first;
        for (eb_u32 i = 0; i < nodes.size(); ++i) locations_.emplace(nodes[i], std::make_pair(layer, i));
    } else {
        Require(found->second.nodes == nodes && found->second.input.device_id == device,
                "EB cohort mapping changed");
    }
    auto &input = found->second.input;
    input.batch_epoch = NextSerial(input.batch_epoch);
    for (eb_u32 i = 0; i < input.count; ++i) input.experts[i].token_count = counts[i];
    active_layer_ = layer;
    active_ = true;
    return input.batch_epoch;
}

void State::RequireActive(eb_u32 layer, eb_u64 epoch) const {
    Require(active_ && active_layer_ == layer && Get(layer).input.batch_epoch == epoch,
            "EB inactive or stale batch epoch");
}

void State::End(eb_u32 layer, eb_u64 epoch) {
    RequireActive(layer, epoch);
    active_ = false;
}

std::pair<eb_u32, eb_u32> State::Locate(NodeId node) const {
    auto found = locations_.find(node);
    Require(found != locations_.end(), "EB sparse demand lacks actual batch routing");
    return found->second;
}

const Cohort &State::Get(eb_u32 layer) const {
    auto found = layers_.find(layer);
    Require(found != layers_.end(), "EB unknown layer");
    return found->second;
}

eb_u64 State::ActiveEpoch(eb_u32 layer, eb_u32 expert) const {
    const auto &input = Get(layer).input;
    RequireActive(layer, input.batch_epoch);
    Require(expert < input.count && input.experts[expert].token_count > 0,
            "EB demand for inactive expert");
    return input.batch_epoch;
}

eb_context State::Decide(eb_u32 layer, eb_u32 incoming, const std::vector<bool> &eligible) {
    eb_context ctx{};
    ctx.input = Get(layer).input;
    RequireActive(layer, ctx.input.batch_epoch);
    Require(incoming < ctx.input.count && eligible.size() == ctx.input.count,
            "EB invalid decision request");
    Require(ctx.input.experts[incoming].token_count > 0, "EB inactive demand");
    ctx.input.incoming = incoming;
    for (eb_u32 i = 0; i < ctx.input.count; ++i)
        if ((ctx.input.experts[i].flags & EB_RESIDENT) && eligible[i])
            ctx.input.experts[i].flags |= EB_ELIGIBLE;
    const auto before = ctx.input;
    // FIFO shares validation/hit/admit/block semantics; only victim order differs.
    const int result = arm_ == Arm::Bpf ? bpf_(jit_, &ctx) : static_cast<int>(native_(&ctx));
    ++counters_.decisions;
    Require(result >= EB_HIT && result <= EB_BLOCKED && ctx.output.status == eb_u32(result) &&
            ctx.output.batch_epoch == before.batch_epoch &&
            std::memcmp(&ctx.input, &before, sizeof(before)) == 0, "EB selector failure");
    if (arm_ == Arm::Fifo && result == EB_EVICT) {
        eb_u64 earliest = std::numeric_limits<eb_u64>::max();
        ctx.output.victim = EB_NO_VICTIM;
        for (eb_u32 i = 0; i < ctx.input.count; ++i) {
            const auto &entry = ctx.input.experts[i];
            if (entry.flags != (EB_RESIDENT | EB_ELIGIBLE)) continue;
            if (ctx.output.victim == EB_NO_VICTIM || entry.admission < earliest) {
                ctx.output.victim = i;
                earliest = entry.admission;
            }
        }
    }
    Validate(ctx);
    return ctx;
}

void State::Validate(const eb_context &snapshot) const {
    const auto &input = snapshot.input;
    RequireActive(input.layer_id, input.batch_epoch);
    auto current = Get(input.layer_id).input;
    Require(input.incoming < current.count && snapshot.output.status <= EB_BLOCKED &&
            snapshot.output.status != EB_INVALID &&
            snapshot.output.batch_epoch == input.batch_epoch, "EB invalid decision metadata");
    current.incoming = input.incoming;
    for (eb_u32 i = 0; i < current.count; ++i) {
        Require((input.experts[i].flags & ~(EB_RESIDENT | EB_ELIGIBLE)) == 0 &&
                (!(input.experts[i].flags & EB_ELIGIBLE) || (current.experts[i].flags & EB_RESIDENT)),
                "EB invalid eligibility flags");
        current.experts[i].flags |= input.experts[i].flags & EB_ELIGIBLE;
    }
    Require(std::memcmp(&current, &input, sizeof(current)) == 0, "EB stale residency snapshot");
    const auto &incoming = input.experts[input.incoming];
    const bool hit = incoming.flags & EB_RESIDENT;
    const auto resident = ResidentCount(input);
    eb_u32 eligible = 0;
    for (eb_u32 i = 0; i < current.count; ++i)
        eligible += input.experts[i].flags == (EB_RESIDENT | EB_ELIGIBLE);
    Require(incoming.token_count > 0 && resident <= input.capacity, "EB invalid demand state");
    const auto status = snapshot.output.status;
    Require((status == EB_HIT && hit) ||
            (status == EB_ADMIT && !hit && resident < input.capacity) ||
            (status == EB_EVICT && !hit && resident == input.capacity && eligible > 0) ||
            (status == EB_BLOCKED && !hit && resident == input.capacity && eligible == 0),
            "EB status contradicts residency or eligibility");
    if (snapshot.output.status == EB_EVICT) {
        auto victim = snapshot.output.victim;
        Require(victim < current.count && victim != input.incoming &&
                input.experts[victim].flags == (EB_RESIDENT | EB_ELIGIBLE), "EB invalid victim");
    } else {
        Require(snapshot.output.victim == EB_NO_VICTIM, "EB unexpected victim");
    }
}

void State::Evicted(const eb_context &snapshot) {
    Validate(snapshot);
    Require(snapshot.output.status == EB_EVICT, "EB eviction without victim decision");
    auto &entry = layers_.at(snapshot.input.layer_id).input.experts[snapshot.output.victim];
    entry.flags = 0;
    entry.admission = 0;
    ++counters_.evictions;
}

void State::CanAdmit(eb_u32 layer, eb_u64 epoch, eb_u32 expert) const {
    RequireActive(layer, epoch);
    const auto &cohort = Get(layer);
    const auto &input = cohort.input;
    Require(expert < input.count && input.experts[expert].token_count &&
            !(input.experts[expert].flags & EB_RESIDENT) && ResidentCount(input) < capacity_,
            "EB admission is not a current active miss with free capacity");
    (void)NextSerial(cohort.serial);
}

void State::Admitted(eb_u32 layer, eb_u64 epoch, eb_u32 expert) {
    CanAdmit(layer, epoch, expert);
    auto &cohort = layers_.at(layer);
    cohort.serial = NextSerial(cohort.serial);
    auto &entry = cohort.input.experts[expert];
    entry.admission = cohort.serial;
    entry.flags = EB_RESIDENT;
    ++counters_.admissions;
}

Counters State::Stats() const {
    auto result = counters_;
    if (jit_) result.jit_calls = jit_calls_(jit_);
    return result;
}
} // namespace eb_adapter
