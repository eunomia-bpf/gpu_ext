// SPDX-License-Identifier: Apache-2.0
// Common runtime instrumentation, not a BPF policy or a synthetic IO model.
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace finemoe_revision {
using u64 = std::uint64_t;
// id,node,bytes,speculative,started_ns,completed_ns,first_use_ns,evicted_ns.
using CopyRecord = std::array<u64, 8>;

inline u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct CopySnapshot {
    std::map<std::string, u64> counters;
    std::vector<CopyRecord> copies;
};

class CopyLedger {
public:
    void Begin() {
        std::lock_guard<std::mutex> guard(mutex_);
        for (const auto &row : copies_)
            if (!row[5]) throw std::runtime_error("cannot reset ledger during a copy");
        copies_.clear();
        resident_.clear();
        counters_.clear();
        enabled_ = true;
    }

    void Count(const char *name, u64 amount = 1) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (enabled_) counters_[name] += amount;
    }

    void Maximum(const char *name, u64 value) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (enabled_ && value > counters_[name]) counters_[name] = value;
    }

    u64 Start(u64 node, u64 bytes, bool speculative) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!enabled_) return 0;
        if (!bytes) throw std::runtime_error("zero-byte expert copy");
        const auto id = copies_.size() + 1;
        copies_.push_back({id, node, bytes, speculative ? 1ULL : 0ULL, NowNs(), 0, 0, 0});
        counters_[speculative ? "prefetch_copy_started" : "demand_copy_started"]++;
        return id;
    }

    void Complete(u64 id) {
        if (!id) return;
        std::lock_guard<std::mutex> guard(mutex_);
        if (id > copies_.size()) throw std::runtime_error("unknown expert copy completion");
        auto &row = copies_.at(id - 1);
        if (row[5]) throw std::runtime_error("duplicate expert copy completion");
        row[5] = NowNs();
        counters_[row[3] ? "prefetch_copy_completed" : "demand_copy_completed"]++;
        counters_[row[3] ? "prefetch_copy_bytes" : "demand_copy_bytes"] += row[2];
        if (row[3]) {
            if (resident_.count(row[1])) throw std::runtime_error("prefetch overwrote resident generation");
            resident_[row[1]] = id - 1;
        }
    }

    void DemandUse(u64 node, bool cache_hit) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!enabled_) return;
        counters_["expert_demand_uses"]++;
        counters_[cache_hit ? "expert_demand_cache_hits" : "expert_demand_cache_misses"]++;
        const auto item = resident_.find(node);
        if (item == resident_.end()) return;  // Includes left-censored warmup copies.
        auto &row = copies_.at(item->second);
        if (!row[6]) {
            row[6] = NowNs();
            counters_["prefetch_first_use_copies"]++;
            counters_["prefetch_first_use_bytes"] += row[2];
        }
    }

    void Evict(u64 node) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!enabled_) return;
        const auto item = resident_.find(node);
        if (item == resident_.end()) return;
        auto &row = copies_.at(item->second);
        row[7] = NowNs();
        if (!row[6]) {
            counters_["prefetch_evicted_unused_copies"]++;
            counters_["prefetch_evicted_unused_bytes"] += row[2];
        }
        resident_.erase(item);
    }

    CopySnapshot Snapshot() {
        std::lock_guard<std::mutex> guard(mutex_);
        CopySnapshot result{counters_, copies_};
        for (const auto *name : {"prefetch_copy_started", "prefetch_copy_completed",
             "prefetch_copy_bytes", "demand_copy_started", "demand_copy_completed", "demand_copy_bytes",
             "prefetch_first_use_copies", "prefetch_first_use_bytes", "prefetch_evicted_unused_copies",
             "prefetch_evicted_unused_bytes", "prefetch_resident_unused_copies", "prefetch_resident_unused_bytes",
             "expert_demand_uses", "expert_demand_cache_hits", "expert_demand_cache_misses",
             "prefetch_queue_enqueued", "prefetch_queue_canceled", "prefetch_queue_dequeued",
             "prefetch_enqueue_resident_skip", "prefetch_eviction_skip", "prefetch_copy_errors",
             "compute_release_syncs", "compute_release_sync_errors"})
            result.counters.emplace(name, 0);
        for (const auto &[node, index] : resident_) {
            (void)node;
            const auto &row = copies_.at(index);
            if (!row[6]) {
                result.counters["prefetch_resident_unused_copies"]++;
                result.counters["prefetch_resident_unused_bytes"] += row[2];
            }
        }
        return result;
    }

private:
    std::mutex mutex_;
    bool enabled_ = false;
    std::vector<CopyRecord> copies_;
    std::map<u64, std::size_t> resident_;
    std::map<std::string, u64> counters_;
};

inline CopyLedger &Ledger() { static CopyLedger ledger; return ledger; }
}  // namespace finemoe_revision
