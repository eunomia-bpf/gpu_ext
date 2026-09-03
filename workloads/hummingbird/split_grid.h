// SPDX-License-Identifier: GPL-2.0
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

namespace hummingbird {
struct Tile { std::array<uint32_t, 3> offset{}, grid{}; };
// Streaming, bounded-memory XYZ partition. Original CTA coordinates appear once.
class GridCursor {
public:
    GridCursor(std::array<uint32_t, 3> grid, uint32_t cap) : grid_(grid) {
        if (!cap || !grid[0] || !grid[1] || !grid[2])
            throw std::runtime_error("zero grid or split capacity");
        tile_[0] = std::min(grid[0], cap);
        tile_[1] = std::min(grid[1], cap / tile_[0]);
        tile_[2] = std::min(grid[2], cap / tile_[0] / tile_[1]);
    }
    bool done() const { return done_; }
    bool unstarted() const { return offset_ == std::array<uint32_t, 3>{}; }
    Tile current() const {
        if (done_) throw std::runtime_error("exhausted split cursor");
        Tile result{offset_, {}};
        for (unsigned int a = 0; a < 3; ++a)
            result.grid[a] = std::min(tile_[a], grid_[a] - offset_[a]);
        return result;
    }
    void advance() {
        if (done_) throw std::runtime_error("duplicate cursor advance");
        for (unsigned int a = 0; a < 3; ++a) {
            if (grid_[a] - offset_[a] > tile_[a]) { offset_[a] += tile_[a]; return; }
            offset_[a] = 0;
        }
        done_ = true;
    }
private:
    std::array<uint32_t, 3> grid_, tile_{}, offset_{};
    bool done_ = false;
};
} // namespace hummingbird
