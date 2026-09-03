"""Finite CPU-only validation of actual rebuilt-store counter requirements."""
import unittest

import run_paper_policy as paper


def state(mode):
    active, bpf = mode != "native-off", mode == "paper-bpf"
    controller = {"matched_predictions": 10, "prefetch_candidates_selected": 20,
                  "completed_requests": 3, "rank_mismatches": 0,
                  "match_mismatches": 0, "rank_calls": 10 if bpf else 0,
                  "bpf_match_calls": 10 if bpf else 0} if active else {}
    dispatcher = {"mode": paper.MODES.index(mode), "prefetch_completed": 12 if active else 0,
                  "prefetch_bytes": 1200 if active else 0,
                  "eviction_selections": 20 if active else 0,
                  "prefetch_hits": 9 if active else 0,
                  "prefetch_wasted": 2 if active else 0,
                  "prefetch_unused_resident": 1 if active else 0,
                  "eviction_mismatches": 0, "bpf_eviction_calls": 20 if bpf else 0,
                  **dict.fromkeys(paper.PREFETCH_PROTECTION_COUNTERS, 0)}
    if active:
        dispatcher.update(prefetch_prediction_epoch=100, prefetch_protected_resident_skips=200,
                          prefetch_copy_started=12)
    return {"mode": mode, "controller": controller, "dispatcher": dispatcher}


class ActivationGateTests(unittest.TestCase):
    def test_three_valid_rebuilt_store_modes(self):
        for mode in paper.MODES:
            with self.subTest(mode=mode):
                paper.validate_activation(mode, state(mode))

    def test_old_store_without_new_counters_is_rejected_in_every_arm(self):
        for mode in paper.MODES:
            old = state(mode)
            for key in paper.PREFETCH_PROTECTION_COUNTERS:
                del old["dispatcher"][key]
            with self.subTest(mode=mode), self.assertRaises(paper.base.GateError):
                paper.validate_activation(mode, old)

    def test_each_missing_or_malformed_counter_is_rejected(self):
        for key in paper.PREFETCH_PROTECTION_COUNTERS:
            for value in (None, -1, 1.0, True, "0"):
                malformed = state("paper-native")
                malformed["dispatcher"][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(paper.base.GateError):
                    paper.validate_activation("paper-native", malformed)

    def test_native_off_requires_every_protection_counter_zero(self):
        for key in paper.PREFETCH_PROTECTION_COUNTERS:
            unexpected = state("native-off")
            unexpected["dispatcher"][key] = 1
            with self.subTest(key=key), self.assertRaises(paper.base.GateError):
                paper.validate_activation("native-off", unexpected)

    def test_both_paper_arms_require_real_epochs_and_protection_engagement(self):
        for mode in ("paper-native", "paper-bpf"):
            for key in ("prefetch_prediction_epoch", "prefetch_protected_resident_skips"):
                missing = state(mode)
                missing["dispatcher"][key] = 0
                with self.subTest(mode=mode, key=key), self.assertRaises(paper.base.GateError):
                    paper.validate_activation(mode, missing)

    def test_both_paper_arms_require_drain_and_all_issued_copies_complete(self):
        for mode in ("paper-native", "paper-bpf"):
            for key, value in (("prefetch_protected_candidates", 1),
                               ("prefetch_copy_started", 11), ("prefetch_copy_started", 13)):
                pending = state(mode)
                pending["dispatcher"][key] = value
                with self.subTest(mode=mode, key=key, value=value), self.assertRaises(paper.base.GateError):
                    paper.validate_activation(mode, pending)

    def test_stale_rejections_are_observations_not_automatic_failures(self):
        for mode in ("paper-native", "paper-bpf"):
            observed = state(mode)
            observed["dispatcher"].update(prefetch_stale_discarded=7, prefetch_no_victim=9,
                                          prefetch_victim_recheck_rejected=2)
            paper.validate_activation(mode, observed)

    def test_requested_mode_must_match_response(self):
        for mode in paper.MODES:
            response = state(mode)
            response["mode"] = "another-arm"
            with self.subTest(mode=mode), self.assertRaises(paper.base.GateError):
                paper.validate_activation(mode, response)

    def test_original_transfer_conservation_gate_is_retained(self):
        invalid = state("paper-bpf")
        invalid["dispatcher"]["prefetch_hits"] = 10
        with self.assertRaises(paper.base.GateError):
            paper.validate_activation("paper-bpf", invalid)


if __name__ == "__main__":
    unittest.main()
