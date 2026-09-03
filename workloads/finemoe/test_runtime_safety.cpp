// SPDX-License-Identifier: Apache-2.0
#include "finemoe_runtime_safety.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>

int main() {
    using namespace finemoe_revision;
    assert(SparseBudget(100, 10, 50, 10) == 50); // Dense input must not shrink a binding expert cap.
    assert(SparseBudget(100, 45, 50, 10) == 45); // Physical free-space bound still applies.
    assert(SparseBudget(100, 10, 0, 10) == 80);
    assert(SparseBudget(100, 10, 50) - 10 == 40); // Sparse input reserves inside its own cap.
    assert(SparseBudget(100, 95, 50, 10) == -5); // Cannot fit: fail rather than overflow.
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic_uint8_t state{1};
    std::promise<void> entering;
    auto entered = entering.get_future();
    bool copied = false;
    std::unique_lock<std::mutex> waiter(mutex);
    std::thread producer([&] {
        entering.set_value();
        CompleteDemand(mutex, cv, state, [&] { copied = true; });
    });
    entered.wait();
    // Producer cannot write readiness/copy while acquire owns the mutex. The
    // wait's atomic unlock+sleep is essential even though state is atomic.
    assert(state == 1 && !copied);
    const bool ready = cv.wait_for(waiter, std::chrono::seconds(2), [&] { return state == 0; });
    assert(ready && copied);
    waiter.unlock();
    producer.join();
    std::cout << "budget and demand publication CPU tests: PASS\n";
}
