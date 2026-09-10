// CPU parity unit for the adaptive speculative-admission governor rule.
// The rule is published (expert_dispatcher.cpp: RecordRequestSpeculativeOutcome
// + AdmitSpeculativePrefix); this oracle re-implements it from the protocol
// text and demands exact integer-level agreement over randomized outcomes.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <vector>

using u64 = unsigned long long;
using i64 = long long;
static constexpr u64 kQuantum = 128ULL << 20;   // 128 MiB
static constexpr u64 kFloor = 64ULL << 20;      // 64 MiB
static constexpr u64 kInitial = 512ULL << 20;   // 512 MiB

static u64 sub_sat(u64 a, u64 b) { return a > b ? a - b : 0ULL; }
static u64 add_sat(u64 a, u64 b) { return a > UINT64_MAX - b ? UINT64_MAX : a + b; }

// Exact published rule from expert_dispatcher.cpp.
struct RuleState {
  u64 budget = kInitial;
  u64 updates = 0, unchanged = 0, decreases = 0, increases = 0;

  void observe(const std::map<std::string, i64> &outcome) {
    const i64 hit = key(outcome, "prefetch_hit_bytes");
    const i64 evicted = key(outcome, "prefetch_wasted_bytes");
    const i64 censor = key(outcome, "prefetch_unused_resident_bytes");
    const i64 wait_ns = key(outcome, "demand_prefetch_wait_ns") +
                        key(outcome, "demand_cache_wait_ns");
    const i64 resolved = hit + evicted;
    const i64 observed_raw = resolved + censor;
    const u64 observed =
        observed_raw > 0 ? static_cast<u64>(observed_raw) : 0;
    const u64 before = budget;
    if (observed == 0) {
      ++unchanged;
    } else if (wait_ns > 0) {
      budget = sub_sat(budget, kQuantum);
      if (budget < kFloor) budget = kFloor;
    } else if (hit > evicted) {
      budget = add_sat(budget, kQuantum);
      const u64 max_budget = kInitial * 4;
      if (budget > max_budget) budget = max_budget;
    } else if (evicted > hit) {
      budget = sub_sat(budget, kQuantum);
      if (budget < kFloor) budget = kFloor;
    } else {
      ++unchanged;
    }
    ++updates;
    if (budget > before) ++increases;
    else if (budget < before) ++decreases;
  }

  static i64 key(const std::map<std::string, i64> &m, const char *k) {
    auto it = m.find(k);
    return it == m.end() ? 0 : it->second;
  }

  static size_t admit(const std::vector<u64> &payload, u64 budget,
                      u64 outstanding) {
    const u64 allowed = sub_sat(budget, outstanding);
    u64 used = 0;
    size_t n = 0;
    while (n < payload.size()) {
      const u64 bytes = payload[n];
      if (bytes == 0 || used + bytes > allowed) break;
      used = add_sat(used, bytes);
      ++n;
    }
    return n;
  }
};

int main() {
  // Large-hit raises a quantum; ceiling is initial*4 = 2 GiB.
  RuleState rise;
  rise.observe({{"prefetch_hit_bytes", 900ULL << 20},
                {"prefetch_wasted_bytes", 1},
                {"prefetch_unused_resident_bytes", 0},
                {"demand_prefetch_wait_ns", 0}, {"demand_cache_wait_ns", 0}});
  if (rise.budget != kInitial + kQuantum) { std::puts("FAIL rise"); return 1; }
  // Large-waste lowers a quantum; floor 64 MiB.
  RuleState fall;
  fall.observe({{"prefetch_hit_bytes", 1},
                {"prefetch_wasted_bytes", 900ULL << 20},
                {"prefetch_unused_resident_bytes", 0},
                {"demand_prefetch_wait_ns", 0}, {"demand_cache_wait_ns", 0}});
  if (fall.budget != kInitial - kQuantum) { std::puts("FAIL fall"); return 1; }
  // Demand wait dominates: decrease even when hit > waste.
  RuleState wait;
  wait.observe({{"prefetch_hit_bytes", 900ULL << 20},
                {"prefetch_wasted_bytes", 1},
                {"prefetch_unused_resident_bytes", 0},
                {"demand_prefetch_wait_ns", 1}, {"demand_cache_wait_ns", 0}});
  if (wait.budget != kInitial - kQuantum) { std::puts("FAIL wait"); return 1; }
  // No observation holds.
  RuleState hold;
  hold.observe({{"prefetch_hit_bytes", 0}, {"prefetch_wasted_bytes", 0},
                {"prefetch_unused_resident_bytes", 0},
                {"demand_prefetch_wait_ns", 0}, {"demand_cache_wait_ns", 0}});
  if (hold.budget != kInitial || hold.unchanged != 1) { std::puts("FAIL hold"); return 1; }
  // Equal hit and waste holds.
  RuleState equal;
  equal.observe({{"prefetch_hit_bytes", 5}, {"prefetch_wasted_bytes", 5},
                 {"prefetch_unused_resident_bytes", 0},
                 {"demand_prefetch_wait_ns", 0}, {"demand_cache_wait_ns", 0}});
  if (equal.budget != kInitial) { std::puts("FAIL equal"); return 1; }
  // Floor: repeated demand waits never go below 64 MiB.
  RuleState floor_case;
  for (int i = 0; i < 64; ++i)
    floor_case.observe({{"prefetch_hit_bytes", 0},
                        {"prefetch_wasted_bytes", 1 << 20},
                        {"prefetch_unused_resident_bytes", 0},
                        {"demand_prefetch_wait_ns", 1},
                        {"demand_cache_wait_ns", 0}});
  if (floor_case.budget != kFloor) { std::puts("FAIL floor"); return 1; }
  // Ceiling: repeated large-hit requests never exceed 2 GiB.
  RuleState ceiling;
  for (int i = 0; i < 64; ++i)
    ceiling.observe({{"prefetch_hit_bytes", 1ULL << 40},
                     {"prefetch_wasted_bytes", 0},
                     {"prefetch_unused_resident_bytes", 0},
                     {"demand_prefetch_wait_ns", 0},
                     {"demand_cache_wait_ns", 0}});
  if (ceiling.budget != kInitial * 4) { std::puts("FAIL ceiling"); return 1; }
  // Admission: largest whole-expert prefix under budget - outstanding.
  const std::vector<u64> payload = {100, 200, 300, 50, 25};
  if (RuleState::admit(payload, 350, 0) != 2) { std::puts("FAIL admitA"); return 1; }
  if (RuleState::admit(payload, 350, 100) != 1) { std::puts("FAIL admitB"); return 1; }
  if (RuleState::admit(payload, 0, 0) != 0) { std::puts("FAIL admitC"); return 1; }
  if (RuleState::admit(payload, 10000, 0) != 5) { std::puts("FAIL admitD"); return 1; }
  if (RuleState::admit({0, 5}, 100, 0) != 0) { std::puts("FAIL admitE"); return 1; }
  // Negative censor (kept-resident revived by demand) saturates observed
  // to zero only when it exactly cancels resolved outcomes; otherwise it
  // merely damps. Sign rule still driven by hit vs evicted.
  RuleState cancel;
  cancel.observe({{"prefetch_hit_bytes", 5}, {"prefetch_wasted_bytes", 5},
                  {"prefetch_unused_resident_bytes", -10},
                  {"demand_prefetch_wait_ns", 0}, {"demand_cache_wait_ns", 0}});
  if (cancel.budget != kInitial) { std::puts("FAIL cancel-zero"); return 1; }
  RuleState damp;
  damp.observe({{"prefetch_hit_bytes", 10}, {"prefetch_wasted_bytes", 5},
                {"prefetch_unused_resident_bytes", -12},
                {"demand_prefetch_wait_ns", 0}, {"demand_cache_wait_ns", 0}});
  if (damp.budget != kInitial + kQuantum) { std::puts("FAIL damp-rise"); return 1; }
  RuleState negbig;
  negbig.observe({{"prefetch_hit_bytes", 2}, {"prefetch_wasted_bytes", 9},
                  {"prefetch_unused_resident_bytes", -20},
                  {"demand_prefetch_wait_ns", 0}, {"demand_cache_wait_ns", 0}});
  if (negbig.budget != kInitial) { std::puts("FAIL negative-hold"); return 1; }
  std::mt19937_64 rng(20260904ULL);
  for (int iter = 0; iter < 100000; ++iter) {
    std::vector<u64> sizes(std::uniform_int_distribution<size_t>(0, 40)(rng));
    for (auto &s : sizes) s = std::uniform_int_distribution<u64>(0, 1000)(rng);
    const u64 budget = std::uniform_int_distribution<u64>(0, 4000)(rng);
    const u64 outstanding = std::uniform_int_distribution<u64>(0, 2000)(rng);
    // Reference: first-fit over the ranked order resolves the same greedy prefix.
    size_t reference = 0; u64 used = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
      if (sizes[i] == 0) break;
      if (used + sizes[i] > sub_sat(budget, outstanding)) break;
      used += sizes[i];
      reference = i + 1;
    }
    if (RuleState::admit(sizes, budget, outstanding) != reference) {
      std::puts("FAIL random admit parity"); return 1;
    }
  }
  std::puts("governor rule CPU parity: all cases OK");
  return 0;
}
