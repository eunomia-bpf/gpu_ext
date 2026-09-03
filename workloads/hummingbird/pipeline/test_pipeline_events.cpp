// SPDX-License-Identifier: GPL-2.0
// CPU fake-event observations exercise the exact ring used by IdleExecutor.
#include "pipeline_events.h"
#include <array>
#include <iostream>
#include <vector>
using hummingbird::PipelineEvents;
static void require(bool value) { if (!value) throw std::runtime_error("pipeline event test failed"); }
int main() {
    for (unsigned bound : {1U, 2U}) {
        PipelineEvents ring(bound);
        std::array<bool, 2> ready{};
        std::array<unsigned, 2> work{};
        std::vector<unsigned> issued, retired;
        auto submit = [&](unsigned id) {
            auto slot = ring.next_slot(); ready[slot] = false; work[slot] = id;
            issued.push_back(id); ring.issued();
        };
        auto query = [&](unsigned slot) {
            if (!ready[slot]) return false;
            retired.push_back(work[slot]); return true;
        };
        require(ring.retire_completed(query)); // Never query an unrecorded event.
        for (unsigned i = 0; i < bound; ++i) submit(i);
        bool rejected = false;
        try { ring.next_slot(); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected && !ring.retire_completed(query)); // Underpredicted/full.
        require(ring.outstanding() == bound && ring.retired_count == 0);
        ready[0] = true;
        ring.retire_completed(query);
        require(ring.retired_count == 1 && ring.next_slot() == 0);
        // Model/kernel boundary must not reset or overwrite a still-pending slot.
        submit(bound);
        if (bound == 2) {
            require(ring.outstanding() == 2 && !ring.retire_completed(query));
            ready[1] = true;
            require(!ring.retire_completed(query) && retired == std::vector<unsigned>({0, 1}));
        }
        bool propagated = false;
        try { ring.retire_completed([](unsigned) -> bool { throw std::runtime_error("query failure"); }); }
        catch (const std::runtime_error &) { propagated = true; }
        require(propagated && ring.outstanding() == 1);
        ready[0] = true;
        require(ring.retire_completed(query)); // Final request drain.
        require(issued == retired && ring.issued_count == ring.retired_count && ring.peak == bound);
        require((ring.overlap_launches > 0) == (bound == 2));
    }
    for (unsigned bound : {0U, 3U, 100U}) {
        bool rejected = false;
        try { PipelineEvents ring(bound); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected);
    }
    std::cout << "pipeline_events_cpu: bounds=1,2 reuse_retirement_order_drain_errors=passed gpu_run=0\n";
}
