// SPDX-License-Identifier: Apache-2.0
#include "finemoe_copy_ledger.h"
#include <cassert>
#include <iostream>

int main() {
    using namespace finemoe_revision;
    CopyLedger ledger;
    assert(ledger.Start(1, 64, true) == 0);  // Warmup is left-censored.
    ledger.Begin();
    auto used = ledger.Start(1, 64, true);
    ledger.Complete(used);
    ledger.DemandUse(1, true);
    ledger.DemandUse(1, true);
    ledger.Evict(1);  // Used eviction must not become waste.
    auto wasted = ledger.Start(2, 128, true);
    ledger.Complete(wasted);
    ledger.Evict(2);
    auto resident = ledger.Start(3, 256, true);
    ledger.Complete(resident);
    auto demand = ledger.Start(4, 512, false);
    ledger.Complete(demand);
    ledger.DemandUse(4, false);
    auto result = ledger.Snapshot();
    auto &c = result.counters;
    assert(c["prefetch_copy_started"] == 3 && c["prefetch_copy_completed"] == 3);
    assert(c["prefetch_copy_bytes"] == 448);
    assert(c["prefetch_first_use_bytes"] == 64);
    assert(c["prefetch_evicted_unused_bytes"] == 128);
    assert(c["prefetch_resident_unused_bytes"] == 256);
    assert(c["prefetch_copy_bytes"] == c["prefetch_first_use_bytes"] +
        c["prefetch_evicted_unused_bytes"] + c["prefetch_resident_unused_bytes"]);
    assert(c["demand_copy_bytes"] == 512 && c["expert_demand_uses"] == 3);
    assert(c["expert_demand_cache_hits"] == 2 && c["expert_demand_cache_misses"] == 1);
    assert(result.copies.size() == 4);
    assert(result.copies[0][6] && result.copies[0][7]);
    assert(!result.copies[1][6] && result.copies[1][7]);
    assert(!result.copies[2][6] && !result.copies[2][7]);
    assert(ledger.Snapshot().counters["prefetch_resident_unused_bytes"] == 256);
    ledger.Begin();  // Snapshot did not evict resident experts.
    ledger.DemandUse(3, true);  // Do not credit a copy from the previous epoch.
    assert(ledger.Snapshot().counters["prefetch_first_use_bytes"] == 0);
    auto active = ledger.Start(5, 32, true);
    bool rejected = false;
    try { ledger.Begin(); } catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
    ledger.Complete(active);
    rejected = false;
    try { ledger.Complete(active); } catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
    std::cout << "copy ledger CPU tests: PASS\n";
}
