// SPDX-License-Identifier: Apache-2.0
#include "moe_expert_policy.h"
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <thread>
#include <vector>

static std::atomic<unsigned long long> decisions{0};
static void require(bool condition, const char *message)
{
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); std::abort(); }
}

enum class ExecState : unsigned char { IDLE, FETCHING, EXECUTING };
struct Node {
    bool device_cuda = false;
    mep_u64 incache_visit_count = 0;
    std::atomic<int> pending_dispatches{0};
    std::atomic<ExecState> exec_state{ExecState::IDLE};
};
struct Expert { mep_u64 key; Node *node; };

// Independent transcription of b766f8 ExpertDispatcher::FindExpertEvict. The
// cache traversal is represented explicitly, not sorted; snapshot flags are not
// consulted by this oracle. Raw node pointers include nullptr exactly as upstream.
static mep_u64 original(const std::vector<Expert> &cache)
{
    mep_u64 min_visit_count = INT_MAX;
    mep_u64 evict = MOE_EXPERT_NONE;
    for (size_t index = 0; index < cache.size(); ++index) {
        auto node = cache[index].node;
        if (node == nullptr) continue;
        if (node->device_cuda && node->incache_visit_count < min_visit_count &&
            node->pending_dispatches.load(std::memory_order_acquire) == 0 &&
            node->exec_state.load(std::memory_order_acquire) == ExecState::IDLE) {
            evict = index;
            min_visit_count = node->incache_visit_count;
        }
    }
    return evict;
}

static void check(const std::vector<Expert> &cache)
{
    std::vector<moe_expert_candidate> entries;
    for (const Expert &expert : cache) {
        moe_expert_candidate e{expert.key, 0, 0, 0};
        if (expert.node) {
            e.incache_visit_count = expert.node->incache_visit_count;
            e.flags = MOE_EXPERT_NODE_PRESENT;
            if (expert.node->device_cuda) e.flags |= MOE_EXPERT_DEVICE_CUDA;
            if (expert.node->pending_dispatches.load() == 0) e.flags |= MOE_EXPERT_PENDING_ZERO;
            if (expert.node->exec_state.load() == ExecState::IDLE) e.flags |= MOE_EXPERT_EXEC_IDLE;
        }
        entries.push_back(e);
    }
    mep_u64 index = 0;
    require(moe_expert_policy_select_v1(entries.data(), entries.size(), &index) == 0, "JIT failed");
    require(index == original(cache), "original/JIT victim mismatch");
    ++decisions;
}

static void random_cases(unsigned seed, unsigned cases)
{
    std::mt19937_64 random(seed);
    std::unique_ptr<Node[]> nodes(new Node[129]);
    const mep_u64 boundaries[]{0, 1, INT_MAX - 1ULL, INT_MAX, INT_MAX + 1ULL, MOE_EXPERT_NONE};
    for (unsigned trial = 0; trial < cases; ++trial) {
        std::vector<Expert> cache;
        const unsigned count = random() % 129;
        for (unsigned i = 0; i < count; ++i) {
            Node &n = nodes[i];
            n.device_cuda = random() % 2;
            n.incache_visit_count = random() % 3 ? boundaries[random() % 6] : random();
            n.pending_dispatches = static_cast<int>(random() % 3) - 1;
            n.exec_state = static_cast<ExecState>(random() % 3);
            cache.push_back({random(), random() % 4 ? &n : nullptr});
        }
        check(cache);
    }
}

int main(int argc, char **argv)
{
    require(argc == 2, "bytecode argument required");
    unsetenv("MOE_EXPERT_POLICY_CODE");
    require(moe_expert_policy_init_v1(nullptr) < 0, "missing code accepted");
    require(moe_expert_policy_init_v1("relative.bin") < 0, "relative code accepted");
    require(moe_expert_policy_init_v1("/nonexistent/moe-code.bin") < 0, "missing file accepted");
    require(moe_expert_policy_init_v1(argv[1]) == 0, "JIT initialization failed");
    require(moe_expert_policy_init_v1(argv[1]) == 0, "same initialization failed");
    require(moe_expert_policy_init_v1("/different/program.bin") < 0, "program replacement accepted");
    check({});
    Node node;
    const mep_u64 boundaries[]{0, 1, INT_MAX - 1ULL, INT_MAX, INT_MAX + 1ULL, MOE_EXPERT_NONE};
    for (unsigned facts = 0; facts < 16; ++facts) {
        node.device_cuda = facts & 2;
        node.pending_dispatches = facts & 4 ? 0 : 1;
        node.exec_state = facts & 8 ? ExecState::IDLE : ExecState::FETCHING;
        for (mep_u64 count : boundaries) {
            node.incache_visit_count = count;
            check({{91, facts & 1 ? &node : nullptr}});
        }
    }
    node.device_cuda = true;
    node.pending_dispatches = 0;
    node.exec_state = ExecState::IDLE;
    node.incache_visit_count = 7;
    check({{99999, &node}, {0, &node}, {800, &node}}); // first tie, not smallest key
    node.pending_dispatches = -1;
    check({{0, &node}}); // nonzero means ineligible, including negative
    node.pending_dispatches = 0;
    std::vector<Expert> maximum;
    for (unsigned i = 0; i < MOE_EXPERT_MAX_CANDIDATES; ++i) maximum.push_back({i, &node});
    check(maximum);
    random_cases(817, 50000);
    std::vector<std::thread> threads;
    for (unsigned i = 0; i < 4; ++i) threads.emplace_back(random_cases, 900 + i, 10000);
    for (auto &thread : threads) thread.join();

    moe_expert_policy_stats before{};
    moe_expert_policy_stats_v1(&before);
    require(before.calls == decisions && before.selected + before.no_victim == decisions,
            "decision counters do not match actual JIT calls");
    mep_u64 index;
    moe_expert_candidate entry{0, 0, MOE_EXPERT_ELIGIBLE, 0};
    require(moe_expert_policy_select_v1(nullptr, 1, &index) < 0, "null entries accepted");
    require(moe_expert_policy_select_v1(&entry, 1, nullptr) < 0, "null output accepted");
    require(moe_expert_policy_select_v1(&entry, MOE_EXPERT_MAX_CANDIDATES + 1, &index) < 0,
            "unbounded count accepted");
    entry.reserved = 1;
    require(moe_expert_policy_select_v1(&entry, 1, &index) < 0, "reserved field accepted");
    entry.reserved = 0;
    entry.flags = 16;
    require(moe_expert_policy_select_v1(&entry, 1, &index) < 0, "unknown flags accepted");
    moe_expert_policy_stats after{};
    moe_expert_policy_stats_v1(&after);
    require(after.errors == before.errors + 5, "errors not counted");
    std::printf("moe_expert_policy_test: backend=ubpf-jit decisions=%llu mismatch=0 "
                "negative_cases=9 threads=4 maximum_candidates=%u\n", decisions.load(),
                MOE_EXPERT_MAX_CANDIDATES);
}
