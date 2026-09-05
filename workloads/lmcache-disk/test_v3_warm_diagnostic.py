#!/usr/bin/env python3
"""CPU-only tests for the one-prefix LMCache V3 diagnostic."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location(
    "diagnose_v3_warm_divergence", HERE / "diagnose_v3_warm_divergence.py"
)
assert SPEC and SPEC.loader
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class FakeResponse:
    status = 200

    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.lines)


class DiagnosticTests(unittest.TestCase):
    def test_server_arms_separate_external_and_native_prefix_caches(self):
        model = Path("/model")
        external = diagnostic.diagnostic_server_argv("lmcache_cpu", model, 18080)
        native = diagnostic.diagnostic_server_argv("native_prefix", model, 18080)
        self.assertIn("--no-enable-prefix-caching", external)
        self.assertIn("--kv-transfer-config", external)
        self.assertIn("--enable-prefix-caching", native)
        self.assertNotIn("--kv-transfer-config", native)
        for argv in (external, native):
            self.assertIn("--return-tokens-as-token-ids", argv)
            self.assertIn("--enable-prompt-tokens-details", argv)
        with self.assertRaises(diagnostic.runner.GateError):
            diagnostic.diagnostic_server_argv("unknown", model, 18080)

    def test_completion_retains_one_token_id_and_top_logprobs(self):
        event = {
            "id": "cmpl-request-1",
            "choices": [
                {
                    "text": " x",
                    "token_ids": [42],
                    "logprobs": {
                        "tokens": ["token_id:42"],
                        "token_logprobs": [-0.1],
                        "top_logprobs": [
                            {"token_id:42": -0.1, "token_id:43": -0.2}
                        ],
                    },
                }
            ],
        }
        usage = {
            "id": "cmpl-request-1",
            "choices": [],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "total_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
        lines = [
            f"data: {json.dumps(event)}\n".encode(),
            f"data: {json.dumps(usage)}\n".encode(),
            b"data: [DONE]\n",
        ]

        def answer(request, timeout):
            self.assertEqual(timeout, 600)
            payload = json.loads(request.data)
            self.assertEqual(payload["max_tokens"], 1)
            self.assertEqual(payload["logprobs"], 20)
            self.assertTrue(payload["return_token_ids"])
            self.assertTrue(payload["return_tokens_as_token_ids"])
            return FakeResponse(lines)

        with patch.object(diagnostic.urllib.request, "urlopen", side_effect=answer):
            response = diagnostic.diagnostic_completion(18080, [1, 2, 3], "request-1")
        self.assertEqual(response["generated_token_ids"], [42])
        self.assertEqual(response["tokens"], ["token_id:42"])
        self.assertEqual(response["top_logprobs"], [{"token_id:42": -0.1, "token_id:43": -0.2}])
        self.assertEqual(diagnostic._cached_tokens(response), 0)

    def test_native_engagement_requires_exact_second_request_cache_hit(self):
        base = {"usage": {"prompt_tokens_details": {"cached_tokens": 0}}}
        hit = {"usage": {"prompt_tokens_details": {"cached_tokens": 1536}}}
        evidence = diagnostic._validate_native_log("ordinary server output", base, hit, 1536)
        self.assertEqual(evidence["prompt_tokens_details_cached_tokens"], [0, 1536])
        with self.assertRaisesRegex(diagnostic.runner.GateError, "engagement mismatch"):
            diagnostic._validate_native_log("ordinary server output", base, base, 1536)
        with self.assertRaisesRegex(diagnostic.runner.GateError, "unexpectedly engaged LMCache"):
            diagnostic._validate_native_log("LMCache initialized", base, hit, 1536)

    def test_pair_delta_preserves_token_and_logprob_difference(self):
        left = {
            "generated_token_ids": [42],
            "text": "a",
            "token_logprobs": [-0.1],
            "top_logprobs": [{"token_id:42": -0.1, "token_id:43": -0.5}],
        }
        right = {
            "generated_token_ids": [43],
            "text": "b",
            "token_logprobs": [-0.2],
            "top_logprobs": [{"token_id:42": -0.3, "token_id:43": -0.2}],
        }
        comparison = diagnostic._pair_delta(left, right)
        self.assertFalse(comparison["same_generated_token_id"])
        self.assertEqual(comparison["common_top_token_count"], 2)
        self.assertAlmostEqual(comparison["max_absolute_logprob_delta_on_common_tokens"], 0.3)
        self.assertAlmostEqual(comparison["selected_token_logprob_delta"], -0.1)


if __name__ == "__main__":
    unittest.main()
