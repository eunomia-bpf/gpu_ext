#include "client_checks.h"
#include <cassert>
#include <limits>
#include <unistd.h>

template <typename F> void must_fail(F function) {
    bool failed = false;
    try { function(); } catch (const std::runtime_error &) { failed = true; }
    assert(failed);
}

int main() {
    using namespace gpreempt_artifact;
    validate_config(0, 200);
    validate_config(1, 101);
    must_fail([] { validate_config(0, 100); });
    must_fail([] { validate_config(-1, 200); });
    must_fail([] { validate_config(2, 200); });
    float input[258]{};
    initialize_input(input, sizeof(input));
    assert(input[0] == -1 && input[128] == 0 && input[256] == 1 && input[257] == -1);
    must_fail([&] { initialize_input(input, 3); });
    must_fail([&] { initialize_input(nullptr, sizeof(input)); });
    char temporary[] = "/tmp/gpreempt-output-check-XXXXXX";
    int fd = mkstemp(temporary);
    assert(fd >= 0);
    const float reference[] = {0.25f, 0.5f, 0.75f};
    assert(write(fd, reference, sizeof(reference)) == sizeof(reference));
    assert(close(fd) == 0);
    OutputCheck check;
    check.initialize("cpu_test", temporary, input, sizeof(input), sizeof(reference));
    check.check(reference, sizeof(reference));
    check.begin_timed();
    check.check(reference, sizeof(reference));
    assert(check.count() == 2 && check.timed_count() == 1);
    float observed[] = {0.25f + 1e-7f, 0.5f, 0.75f};
    check.check(observed, sizeof(observed));
    observed[1] = 0.6f;
    must_fail([&] { check.check(observed, sizeof(observed)); });
    observed[1] = std::numeric_limits<float>::quiet_NaN();
    must_fail([&] { check.check(observed, sizeof(observed)); });
    must_fail([&] { check.check(reference, sizeof(float)); });
    must_fail([&] { check.initialize("cpu_test", temporary, input, sizeof(input), 16); });
    assert(check.count() == 3 && check.timed_count() == 2);
    OutputCheck missing;
    must_fail([&] { missing.initialize("missing", "/nonexistent/gpreempt-ref", input, sizeof(input), 12); });
    assert(unlink(temporary) == 0);
    std::puts("{\"test\":\"gpreempt_common_numerics\",\"gpu_execution\":false,\"passed\":true}");
}
