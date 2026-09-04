"""Lightweight source-wiring checks; C++/CUDA behavior needs separate tests.

These deliberately do not compile or load any extension and may run while the
coordinator has reserved quiet GPU timing. They are not a substitute for the
actual helper executable, rebuilt store, numerical canaries, or measured runs.
"""
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
CORE = HERE / "deps/MoE-Infinity/core/parallel"
SOURCE = (CORE / "expert_dispatcher.cpp").read_text()
HEADER = (CORE / "expert_dispatcher.h").read_text()


def body(start, end):
    return SOURCE.split(start, 1)[1].split(end, 1)[0]


class PrefetchSourceWiring(unittest.TestCase):
    def test_demand_default_does_not_use_prediction_protection(self):
        self.assertIn("FindExpertEvict(int gpu_id, uint64_t prediction_epoch = 0)", HEADER)
        self.assertIn("EvictActivationVictim(FindExpertEvict(gpu_id), gpu_id)", SOURCE)

    def test_speculation_passes_epoch_to_selection_and_commit(self):
        prefetch = body("void ExpertDispatcher::FetchActivationPrefetch(",
                        "ExpertDispatcher::GetActivationStats()")
        self.assertIn("FindExpertEvict(gpu_id, args.prediction_epoch)", prefetch)
        self.assertIn("EvictActivationVictim(victim, gpu_id, args.prediction_epoch)", prefetch)
        self.assertIn("release(); return;  // never stall demand for speculation", prefetch)

    def test_protection_is_shared_before_either_selector(self):
        select = body("ExpertNodePtr ExpertDispatcher::FindExpertEvict(",
                      "void ExpertDispatcher::ConfigureActivationPolicy(")
        self.assertLess(select.index("activation_prediction_.MayEvict"),
                        select.index("activation_select_("))
        self.assertIn("if (prediction_epoch &&", select)

    def test_selected_victim_rechecked_under_same_lock(self):
        evict = body("bool ExpertDispatcher::EvictActivationVictim(",
                     "void ExpertDispatcher::FetchActivationPrefetch(")
        self.assertLess(evict.index("lock(activation_mutex_)"),
                        evict.index("activation_prediction_.MayEvict"))
        self.assertLess(evict.index("activation_prediction_.MayEvict"),
                        evict.index("node->SetDevice"))

    def test_final_epoch_check_and_copy_issue_share_lock(self):
        prefetch = body("void ExpertDispatcher::FetchActivationPrefetch(",
                        "ExpertDispatcher::GetActivationStats()")
        issue = prefetch.split("cudaEvent_t done = nullptr;", 1)[1]
        self.assertLess(issue.index("lock(activation_mutex_)"),
                        issue.index("activation_prediction_.Current"))
        self.assertLess(issue.index("activation_prediction_.Current"),
                        issue.index("node->SetDevice"))
        self.assertIn("activation_copy_started_.fetch_add(1, std::memory_order_relaxed);\n  }\n  if (done)", issue)

    def test_publisher_protects_before_making_work_available(self):
        submit = body("void ExpertDispatcher::SubmitActivationPrefetch(",
                      "void ExpertDispatcher::RecordActivationUse(")
        enabled = submit.split("std::vector<std::vector<CallArgs>> work", 1)[1]
        self.assertLess(enabled.index("activation_prediction_.Replace(identities)"),
                        enabled.index("ReplaceBackground"))
        self.assertIn("item.prediction_epoch = epoch", enabled)

    def test_native_off_drain_stays_policy_neutral(self):
        drain = body("void ExpertDispatcher::DrainActivationPrefetch()",
                     "bool ExpertDispatcher::EvictActivationVictim(")
        self.assertIn("if (!activation_mode_.load(std::memory_order_acquire)) return;", drain)
        self.assertLess(drain.index("activation_prediction_.Invalidate()"),
                        drain.index("queue.DrainBackground()"))
        self.assertIn("}\n  // Never wait while holding activation_mutex_", drain)


if __name__ == "__main__":
    unittest.main()
