#include "deps/MoE-Infinity/core/parallel/revision_fetch_queue.h"

#include <cassert>
#include <functional>
#include <iostream>
#include <limits>
#include <map>

namespace {
using Key = uint64_t;
constexpr Key kNone = std::numeric_limits<Key>::max();

// Small cache executor using the very same protection/epoch helper as the real
// dispatcher. This tests runtime admission, not CUDA transfer or BPF JIT code.
struct Cache {
  RevisionPredictionSet prediction;
  std::map<Key, double> resident;
  unsigned copies = 0;

  Key Select(uint64_t epoch) const {
    Key victim = kNone;
    double minimum = std::numeric_limits<double>::infinity();
    for (const auto& item : resident) {
      if (epoch && !prediction.MayEvict(epoch, item.first)) continue;
      if (item.second < minimum) {
        victim = item.first;
        minimum = item.second;
      }
    }
    return victim;
  }

  bool Evict(Key victim, uint64_t epoch) {
    if (victim == kNone || (epoch && !prediction.MayEvict(epoch, victim)))
      return false;
    return resident.erase(victim) != 0;
  }

  bool Issue(Key incoming, double score, uint64_t epoch,
             const std::function<void()>& after_selection = {},
             const std::function<void()>& before_copy = {},
             const std::function<void()>& after_issue = {}) {
    if (!prediction.Current(epoch)) return false;
    if (resident.size() == 2) {
      auto victim = Select(epoch);
      if (after_selection) after_selection();
      if (!Evict(victim, epoch)) return false;
    }
    if (before_copy) before_copy();
    if (!prediction.Current(epoch)) return false;
    ++copies;  // Protected atomic issue point, immediately before real DMA.
    if (after_issue) after_issue();
    resident[incoming] = score;  // Already-started copy may finish in new epoch.
    return true;
  }
};

void ProtectionAndDemand() {
  Cache cache;
  cache.resident = {{10, 0.9}, {20, 0.1}};
  auto epoch = cache.prediction.Replace({20, 30, 40});
  assert(cache.prediction.Size() == 3);
  assert(cache.Select(epoch) == 10);  // 20 is protected despite lower reuse.
  assert(cache.Issue(30, 0.05, epoch));
  assert(cache.resident.count(20) && cache.resident.count(30));
  assert(cache.Select(epoch) == kNone);
  assert(!cache.Issue(40, 0.01, epoch));
  assert(cache.copies == 1);  // Later candidate cannot evict earlier prediction.
  assert(cache.Select(0) == 30);  // Demand ignores prediction-set protection.
  assert(cache.Evict(30, 0));
  assert(cache.resident.count(20));
}

void StaleBeforeClaimAndBeforeEviction() {
  Cache cache;
  cache.resident = {{10, 0.1}, {20, 0.2}};
  auto epoch = cache.prediction.Replace({30});
  cache.prediction.Invalidate();
  assert(!cache.Issue(30, 0.3, epoch));
  assert(cache.resident.size() == 2 && cache.copies == 0);
  epoch = cache.prediction.Replace({30});
  assert(!cache.Issue(30, 0.3, epoch, [&] {
    cache.prediction.Replace({10, 30});  // Protect selected victim before commit.
  }));
  assert(cache.resident.size() == 2 && cache.resident.count(10));
  assert(cache.copies == 0);
}

void StaleBeforeCopyAndCompletionAfterIssue() {
  Cache cache;
  cache.resident = {{10, 0.1}, {20, 0.2}};
  auto epoch = cache.prediction.Replace({30});
  assert(!cache.Issue(30, 0.3, epoch, {}, [&] {
    cache.prediction.Replace({40});  // Victim already removed, no DMA may start.
  }));
  assert(cache.resident.size() == 1 && cache.copies == 0);
  epoch = cache.prediction.Replace({30});
  assert(cache.Issue(30, 0.3, epoch, {}, {}, [&] {
    cache.prediction.Replace({40});  // DMA started before replacement: finish.
  }));
  assert(cache.resident.count(30) && cache.copies == 1);
}

void QueueReplacementAndDraining() {
  struct Work { Key key; uint64_t epoch; };
  RevisionFetchQueue<Work> queue;
  RevisionPredictionSet prediction;
  auto first = prediction.Replace({10, 11});
  queue.ReplaceBackground({{10, first}, {11, first}});
  Work work{};
  assert(queue.Pop(work) && work.key == 10);
  auto second = prediction.Replace({20});
  queue.ReplaceBackground({{20, second}});
  assert(!prediction.Current(work.epoch));  // Popped does not imply copy issued.
  queue.CompleteBackground();
  Work demand{99, 0};
  queue.Push(demand);
  assert(queue.Pop(work) && work.key == 99);
  assert(queue.Pop(work) && work.key == 20 && prediction.Current(work.epoch));
  prediction.Invalidate();
  assert(!prediction.Current(work.epoch));
  queue.CompleteBackground();
  queue.DrainBackground();
  assert(prediction.Size() == 0);
  queue.Close();
  assert(!queue.Pop(work));
}

void EmptyDuplicateAndIdentityCases() {
  RevisionPredictionSet prediction;
  assert(!prediction.Current(0) && !prediction.MayEvict(0, 1));
  auto first = prediction.Replace({1, 1, (Key(1) << 32) | 1});
  assert(prediction.Size() == 2);
  assert(!prediction.MayEvict(first, 1));
  assert(!prediction.MayEvict(first, (Key(1) << 32) | 1));
  assert(prediction.MayEvict(first, (Key(2) << 32) | 1));
  auto second = prediction.Replace({});
  assert(second > first && !prediction.Current(first));
  assert(prediction.MayEvict(second, 1) && prediction.Size() == 0);
}
}  // namespace

int main() {
  ProtectionAndDemand();
  StaleBeforeClaimAndBeforeEviction();
  StaleBeforeCopyAndCompletionAfterIssue();
  QueueReplacementAndDraining();
  EmptyDuplicateAndIdentityCases();
  std::cout << "prediction protection: 5 scenario groups passed\n";
}
