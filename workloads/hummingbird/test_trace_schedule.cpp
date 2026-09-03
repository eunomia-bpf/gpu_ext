#include "trace_schedule.h"
#include <cassert>
#include <cstdio>

int main() {
    const std::vector<int64_t> trace{0, 0, 10, 90}, none;
    hummingbird::Schedule schedule(100, 200, 0, trace);
    gpreempt_load_study::Request request{};
    assert(schedule.offered() == 4 && schedule.finite());
    assert(!schedule.admit(99, request));
    assert(schedule.admit(100, request) && request.id == 0 && request.scheduled_ns == 100);
    assert(schedule.admit(105, request) && request.id == 1 && request.scheduled_ns == 100);
    assert(schedule.admit(150, request) && request.id == 2 && request.scheduled_ns == 110);
    assert(!schedule.admit(189, request));
    assert(schedule.admit(195, request) && request.id == 3 && request.scheduled_ns == 190);
    assert(!schedule.admit(199, request) && schedule.started() == 4 && schedule.next_ns() == 200);
    hummingbird::Schedule backlog(100, 200, 0, trace);
    assert(!backlog.admit(200, request) && backlog.started() == 0);
    hummingbird::Schedule periodic(100, 200, 10, none), continuous(100, 200, 0, none);
    assert(periodic.offered() == 10 && periodic.admit(110, request) && request.scheduled_ns == 100);
    assert(!continuous.finite() && continuous.admit(125, request) && request.scheduled_ns == 125);
    for (const auto& invalid : {std::vector<int64_t>{-1}, {100}, {10,0}}) {
        bool rejected = false;
        try { hummingbird::Schedule bad(100,200,0,invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    std::puts("trace FIFO ties, backlog, deadline, periodic/continuous and invalid offsets PASS");
}
