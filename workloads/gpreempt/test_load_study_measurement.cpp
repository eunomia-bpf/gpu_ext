#include "load_study_measurement.h"
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

using namespace gpreempt_load_study;
static void require(bool ok) { if (!ok) std::abort(); }

int main() {
    Request r{};
    Schedule fifo(100, 200, 10);
    require(fifo.offered() == 10);
    require(!fifo.admit(99, r));
    require(fifo.admit(100, r) && r.id == 0 && r.scheduled_ns == 100);
    // Backlog does not skip to newest: late starts keep original intended arrivals.
    require(fifo.admit(175, r) && r.id == 1 && r.scheduled_ns == 110);
    require(fifo.admit(199, r) && r.id == 2 && r.scheduled_ns == 120);
    require(!fifo.admit(200, r));
    require(!fifo.admit(201, r));
    require(fifo.started() == 3 && fifo.offered() - fifo.started() == 7);

    Schedule exact(100, 120, 10);
    require(exact.admit(100, r));
    require(!exact.admit(109, r));
    require(exact.admit(110, r));
    require(!exact.admit(120, r)); // scheduled == cutoff is not offered.
    require(exact.offered() == 2);
    Schedule partial(100, 121, 10);
    require(partial.offered() == 3);
    Schedule oversleep(100, 120, 10);
    require(!oversleep.admit(125, r) && oversleep.started() == 0);

    Schedule continuous(100, 200, 0);
    require(!continuous.admit(99, r));
    require(continuous.admit(101, r) && r.id == 0 && r.scheduled_ns == 101);
    require(continuous.admit(150, r) && r.id == 1 && r.scheduled_ns == 150);
    require(!continuous.admit(200, r));
    bool threw = false;
    try { (void)continuous.offered(); } catch (const std::logic_error &) { threw = true; }
    require(threw);

    // Independent threads receive exactly one epoch, including the wall-clock report.
    CommonStart start(2, 60000000000LL, 0);
    Window a, b;
    std::thread one([&] { a = start.wait(); });
    std::thread two([&] { b = start.wait(); });
    one.join(); two.join();
    require(a.begin_ns == b.begin_ns && a.end_ns == b.end_ns);
    require(a.end_ns - a.begin_ns == 60000000000LL);
    require(a.wall_begin == b.wall_begin && a.wall_end == b.wall_end);
    require(start.window().begin_ns == a.begin_ns);

    for (int invalid : {0, -1}) {
        threw = false;
        try { Schedule bad(10, 10 + invalid, 1); }
        catch (const std::invalid_argument &) { threw = true; }
        require(threw);
    }
    std::cout << "load-study FIFO/common-window CPU tests passed\n";
}
