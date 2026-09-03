#include "deps/MoE-Infinity/core/parallel/revision_fetch_queue.h"
#include <cassert>
#include <future>

int main() {
  RevisionFetchQueue<int> queue;
  queue.ReplaceBackground({10, 11});
  int demand = 1, value = -1;
  queue.Push(demand);
  assert(queue.Pop(value) && value == 1);
  assert(queue.Pop(value) && value == 10);
  queue.CompleteBackground();
  demand = 2;
  queue.Push(demand);
  queue.ReplaceBackground({20});
  assert(queue.Pop(value) && value == 2);
  assert(queue.Pop(value) && value == 20);
  auto drain = std::async(std::launch::async, [&] { queue.DrainBackground(); });
  assert(drain.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout);
  queue.CompleteBackground();
  drain.get();
  auto waiter = std::async(std::launch::async, [&] { return queue.Pop(value); });
  queue.Close();
  assert(!waiter.get());
  queue.ReplaceBackground({30});
  assert(!queue.Pop(value));
}
