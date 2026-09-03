// SPDX-License-Identifier: GPL-2.0
#include "split_grid.h"
#include <cstdio>
#include <limits>
#include <random>
#include <set>
using namespace hummingbird;
static void require(bool condition) { if (!condition) throw std::runtime_error("grid cursor test failed"); }
int main() {
    std::mt19937 random(20260903);
    for (unsigned int trial = 0; trial < 1000; ++trial) {
        const std::array<uint32_t, 3> grid{uint32_t(1 + random() % 11), uint32_t(1 + random() % 9), uint32_t(1 + random() % 7)};
        const uint32_t cap = 1 + random() % 100;
        GridCursor cursor(grid, cap); std::set<std::array<uint32_t, 3>> seen;
        bool first = true;
        while (!cursor.done()) {
            require(cursor.unstarted() == first); first = false;
            auto tile = cursor.current();
            require(uint64_t(tile.grid[0]) * tile.grid[1] * tile.grid[2] <= cap);
            for (uint32_t z = 0; z < tile.grid[2]; ++z)
                for (uint32_t y = 0; y < tile.grid[1]; ++y)
                    for (uint32_t x = 0; x < tile.grid[0]; ++x) {
                        std::array<uint32_t, 3> point{tile.offset[0] + x, tile.offset[1] + y, tile.offset[2] + z};
                        for (unsigned int a = 0; a < 3; ++a) require(point[a] < grid[a]);
                        require(seen.insert(point).second);
                    }
            cursor.advance();
        }
        require(seen.size() == uint64_t(grid[0]) * grid[1] * grid[2]);
    }
    const auto maximum = std::numeric_limits<uint32_t>::max();
    GridCursor large({maximum, 1, 1}, maximum - 1);
    require(large.current().grid[0] == maximum - 1); large.advance();
    require(large.current().grid[0] == 1 && large.current().offset[0] == maximum - 1);
    large.advance(); require(large.done());
    for (const auto grid : {std::array<uint32_t, 3>{0, 1, 1}, {1, 0, 1}, {1, 1, 0}}) {
        bool caught = false; try { GridCursor invalid(grid, 1); } catch (...) { caught = true; } require(caught);
    }
    bool caught = false; try { GridCursor invalid({1, 1, 1}, 0); } catch (...) { caught = true; } require(caught);
    std::puts("split_grid_cpu: random_exact_coverage_cases=1000 overflow_and_rejection=passed gpu_run=0");
}
