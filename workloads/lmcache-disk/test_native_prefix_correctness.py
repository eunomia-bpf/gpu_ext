#!/usr/bin/env python3
"""CPU-only tests for the eight-prefix native-vLLM correctness reference."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location(
    "run_native_prefix_correctness", HERE / "run_native_prefix_correctness.py"
)
assert SPEC and SPEC.loader
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)

OUTPUT_TOKENS = native.runner.OUTPUT_TOKENS
PREFIX_TOKENS = native.runner.PREFIX_TOKENS
GATE = native.runner.GateError


def make_response(request_id, token_ids, text, prompt_tokens, cached_tokens=None):
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": OUTPUT_TOKENS,
        "total_tokens": prompt_tokens + OUTPUT_TOKENS,
    }
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return {
        "request_header": request_id,
        "engine_request_id": f"cmpl-{request_id}",
        "input_tokens": prompt_tokens,
        "status": 200,
        "ttft_ms": 1.5,
        "e2e_ms": 4.25,
        "usage": usage,
        "text": text,
        "generated_token_ids": list(token_ids),
    }


def make_prefix(index, hit_tokens=PREFIX_TOKENS):
    cold_ids = list(range(1000 * index, 1000 * index + OUTPUT_TOKENS))
    warm_ids = list(range(2000 * index, 2000 * index + OUTPUT_TOKENS))
    return {
        "prefix_index": index,
        "expected_hit_tokens": hit_tokens,
        "cold": make_response(
            f"native-p{index}-cold", cold_ids, f"COLD-{index}", 1549, 0
        ),
        "warm": make_response(
            f"native-p{index}-warm", warm_ids, f"WARM-{index}", 1550, hit_tokens
        ),
    }


def make_observations(hit_tokens=PREFIX_TOKENS):
    return [make_prefix(index, hit_tokens) for index in range(native.PREFIXES)]


def make_result(config, observations):
    return {
        "schema": 2,
        "config": config,
        "prefix_count": len(observations),
        "observations": observations,
    }


class FakeStream:
    status = 200

    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.lines)


def completion_lines(token_ids, text, prompt_tokens, request_id, include_ids=True):
    events = []
    for token in token_ids:
        choice = {"text": text, "logprobs": {
            "tokens": [f"token_id:{token}"], "token_logprobs": [-1.0], "top_logprobs": []}}
        if include_ids:
            choice["token_ids"] = [token]
        events.append({"id": f"cmpl-{request_id}", "choices": [choice]})
    events.append({
        "id": f"cmpl-{request_id}",
        "choices": [],
        "usage": {"prompt_tokens": prompt_tokens,
                  "completion_tokens": len(token_ids),
                  "total_tokens": prompt_tokens + len(token_ids)},
    })
    return [f"data: {json.dumps(event)}\n".encode() for event in events] + [b"data: [DONE]\n"]


def lcp(left, right):
    return next(
        (index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
        min(len(left), len(right)),
    )


class FrozenPromptTests(unittest.TestCase):
    def test_frozen_prompt_pairs(self):
        prompts = native.runner.load_prompts(native.runner.PROMPTS)
        self.assertEqual(len(prompts["prefixes"]), 8)
        self.assertEqual([item["index"] for item in prompts["prefixes"]], list(range(8)))
        for item in prompts["prefixes"]:
            self.assertEqual(item["expected_hit_tokens"], 1536)
            self.assertEqual(item["expected_store_tokens"], 1536)
            self.assertEqual(len(item["prefix_token_ids"]), 1536)
            self.assertEqual(item["cold_tokens"], len(item["cold_token_ids"]))
            self.assertEqual(item["warm_tokens"], len(item["warm_token_ids"]))
            cold, warm = item["cold_token_ids"], item["warm_token_ids"]
            shared = lcp(cold, warm)
            self.assertEqual(shared, item["lcp_tokens"])
            self.assertEqual(shared - shared % 256, 1536)
            self.assertEqual(cold[:1536], warm[:1536])
            self.assertGreater(len(cold), 1536)
            self.assertGreater(len(warm), 1536)
            self.assertNotEqual(cold[1536:], warm[1536:])

    def test_prompt_observation_gates(self):
        prompts = native.runner.load_prompts(native.runner.PROMPTS)
        native._validate_prompt_observations(prompts, make_observations())
        with self.assertRaisesRegex(GATE, "exactly 8 prompt/observation pairs"):
            native._validate_prompt_observations(prompts, make_observations()[:7])
        with self.assertRaisesRegex(GATE, "exactly 8 prompt/observation pairs"):
            native._validate_prompt_observations(
                {"prefixes": prompts["prefixes"][:7]}, make_observations()
            )
        swapped = make_observations()
        swapped[1]["prefix_index"] = 0
        with self.assertRaisesRegex(GATE, "prompt mismatch for prefix 1"):
            native._validate_prompt_observations(prompts, swapped)
        tampered = make_observations()
        tampered[2]["expected_hit_tokens"] = 768
        with self.assertRaisesRegex(GATE, "prompt mismatch for prefix 2"):
            native._validate_prompt_observations(prompts, tampered)


class NativeServerProtocolTests(unittest.TestCase):
    def test_frozen_native_server_argv(self):
        argv = native._server_argv(Path("/model"), 18080)
        self.assertEqual(argv[2], "/model")
        self.assertEqual(argv[argv.index("--port") + 1], "18080")
        self.assertIn("--enable-prefix-caching", argv)
        self.assertNotIn("--no-enable-prefix-caching", argv)
        self.assertIn("--enable-prompt-tokens-details", argv)
        self.assertIn("--return-tokens-as-token-ids", argv)
        self.assertNotIn("--kv-transfer-config", argv)
        self.assertIn("--enforce-eager", argv)

    def test_streamed_completion_requests_token_ids(self):
        token_ids = list(range(100, 116))
        prompt = list(range(300, 310))
        lines = completion_lines(token_ids, "x", len(prompt), "native-p0-cold")
        captured = {}

        def answer(request, timeout):
            self.assertEqual(timeout, 600)
            captured["payload"] = json.loads(request.data)
            captured["request_id"] = request.headers.get("X-request-id")
            return FakeStream(lines)

        with patch.object(
            native.runner.legacy.urllib.request, "urlopen", side_effect=answer
        ):
            response = native.runner.legacy.streamed_completion(18080, prompt, "native-p0-cold")
        payload = captured["payload"]
        self.assertEqual(payload["model"], native.runner.legacy.MODEL_ID)
        self.assertEqual(payload["prompt"], prompt)
        self.assertEqual(payload["max_tokens"], 16)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["seed"], 0)
        self.assertIs(payload["stream"], True)
        self.assertIs(payload["ignore_eos"], True)
        self.assertIs(payload["return_token_ids"], True)
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertEqual(captured["request_id"], "native-p0-cold")
        self.assertEqual(response["generated_token_ids"], token_ids)
        self.assertEqual(response["text"], "x" * 16)
        self.assertEqual(response["engine_request_id"], "cmpl-native-p0-cold")
        self.assertEqual(response["input_tokens"], len(prompt))

    def test_streamed_completion_rejects_missing_token_ids(self):
        lines = completion_lines(
            list(range(16)), "x", 10, "native-p0-cold", include_ids=False
        )
        with patch.object(
            native.runner.legacy.urllib.request, "urlopen",
            side_effect=lambda *_args, **_kwargs: FakeStream(lines),
        ):
            with self.assertRaisesRegex(GATE, "16 streamed token IDs"):
                native.runner.legacy.streamed_completion(18080, list(range(10)), "native-p0-cold")

    def test_streamed_completion_rejects_short_stream(self):
        lines = completion_lines(list(range(15)), "x", 10, "native-p0-cold")
        with patch.object(
            native.runner.legacy.urllib.request, "urlopen",
            side_effect=lambda *_args, **_kwargs: FakeStream(lines),
        ):
            with self.assertRaisesRegex(GATE, "16 streamed token IDs"):
                native.runner.legacy.streamed_completion(18080, list(range(10)), "native-p0-cold")


class EngagementGateTests(unittest.TestCase):
    def test_exact_cold_zero_warm_full_hit(self):
        observations = make_observations()
        evidence = native._validate_engagement("ordinary server output", observations)
        self.assertEqual(
            evidence["prompt_tokens_details_cached_tokens"],
            [{"prefix_index": index, "cold_warm": [0, 1536]} for index in range(8)],
        )

    def test_rejects_partial_warm_hit(self):
        observations = make_observations()
        observations[3]["warm"]["usage"]["prompt_tokens_details"]["cached_tokens"] = 768
        with self.assertRaisesRegex(GATE, "engagement mismatch for prefix 3"):
            native._validate_engagement("ordinary server output", observations)

    def test_rejects_nonzero_cold_cache(self):
        observations = make_observations()
        observations[2]["cold"]["usage"]["prompt_tokens_details"]["cached_tokens"] = 1536
        with self.assertRaisesRegex(GATE, "engagement mismatch for prefix 2"):
            native._validate_engagement("ordinary server output", observations)

    def test_rejects_missing_prompt_details(self):
        observations = make_observations()
        del observations[4]["warm"]["usage"]["prompt_tokens_details"]
        with self.assertRaisesRegex(GATE, "engagement mismatch for prefix 4"):
            native._validate_engagement("ordinary server output", observations)

    def test_rejects_unexpected_lmcache_engagement(self):
        with self.assertRaisesRegex(GATE, "unexpectedly engaged LMCache"):
            native._validate_engagement(
                "LMCache initialized in 0.1s", make_observations()
            )

    def test_rejects_fatal_log_evidence(self):
        with self.assertRaisesRegex(GATE, "fatal evidence"):
            native._validate_engagement(
                "line\nCUDA error: misaligned address\nline", make_observations()
            )


class TokenIdGateTests(unittest.TestCase):
    def test_valid_response_passes(self):
        response = make_response("native-p0-cold", list(range(16)), "text", 10, 0)
        native._validate_response(response, 10, "native-p0-cold")

    def test_rejects_short_token_ids(self):
        response = make_response("native-p0-cold", list(range(15)), "text", 10, 0)
        with self.assertRaisesRegex(GATE, "generated token IDs"):
            native._validate_response(response, 10, "native-p0-cold")

    def test_rejects_non_integer_token_id(self):
        response = make_response("native-p0-cold", list(range(16)), "text", 10, 0)
        response["generated_token_ids"][-1] = "42"
        with self.assertRaisesRegex(GATE, "generated token IDs"):
            native._validate_response(response, 10, "native-p0-cold")

    def test_rejects_legacy_response_without_token_ids(self):
        response = make_response("native-p0-cold", list(range(16)), "text", 10, 0)
        del response["generated_token_ids"]
        with self.assertRaisesRegex(GATE, "legacy results without token IDs are rejected"):
            native._validate_response(response, 10, "native-p0-cold")

    def test_base_semantics_still_enforced(self):
        response = make_response("native-p0-cold", list(range(16)), "text", 10, 0)
        response["request_header"] = "someone-else"
        with self.assertRaisesRegex(GATE, "request ID mismatch"):
            native._validate_response(response, 10, "native-p0-cold")


class ExactOutputShapeTests(unittest.TestCase):
    def test_rejects_duplicate_prefix_index(self):
        result = make_result("lmcache_cpu", make_observations())
        result["observations"][7]["prefix_index"] = 3
        with self.assertRaisesRegex(GATE, "unique integers 0..7"):
            native._exact_outputs(result, "lmcache_cpu")

    def test_rejects_missing_prefix_index(self):
        result = make_result("lmcache_cpu", make_observations())
        result["observations"][7]["prefix_index"] = None
        with self.assertRaisesRegex(GATE, "unique integers 0..7"):
            native._exact_outputs(result, "lmcache_cpu")

    def test_rejects_out_of_range_prefix_index(self):
        result = make_result("lmcache_cpu", make_observations())
        result["observations"][7]["prefix_index"] = 8
        with self.assertRaisesRegex(GATE, "unique integers 0..7"):
            native._exact_outputs(result, "lmcache_cpu")

    def test_rejects_wrong_observation_count(self):
        result = make_result("lmcache_cpu", make_observations()[:7])
        with self.assertRaisesRegex(GATE, "exactly 8 observations"):
            native._exact_outputs(result, "lmcache_cpu")

    def test_rejects_short_token_id_list(self):
        result = make_result("lmcache_disk", make_observations())
        del result["observations"][2]["cold"]["generated_token_ids"][-1]
        with self.assertRaisesRegex(GATE, "lacks exact generated token IDs"):
            native._exact_outputs(result, "lmcache_disk")

    def test_rejects_long_token_id_list(self):
        result = make_result("lmcache_disk", make_observations())
        result["observations"][5]["warm"]["generated_token_ids"].append(9000)
        with self.assertRaisesRegex(GATE, "lacks exact generated token IDs"):
            native._exact_outputs(result, "lmcache_disk")


class ComparisonTests(unittest.TestCase):
    def test_exact_match_passes(self):
        native_result = {"observations": make_observations()}
        cpu_result = make_result("lmcache_cpu", make_observations())
        disk_result = make_result("lmcache_disk", make_observations())
        summary = native._compare_outputs(
            native._exact_outputs(native_result, "native_prefix"),
            native._exact_outputs(cpu_result, "lmcache_cpu"),
            native._exact_outputs(disk_result, "lmcache_disk"),
            "native-dir", "cpu-dir", "disk-dir",
        )
        self.assertIs(summary["exact_token_ids_equal"], True)
        self.assertIs(summary["exact_text_equal"], True)
        self.assertEqual(summary["prefixes"], 8)
        self.assertEqual(summary["phases_per_prefix"], ["cold", "warm"])
        self.assertEqual(summary["output_tokens_per_request"], 16)

    def test_detects_token_id_mismatch(self):
        native_result = {"observations": make_observations()}
        disk_result = make_result("lmcache_disk", make_observations())
        disk_result["observations"][3]["warm"]["generated_token_ids"][-1] += 1
        with self.assertRaisesRegex(GATE, "lmcache_disk:3:warm:token_ids"):
            native._compare_outputs(
                native._exact_outputs(native_result, "native_prefix"),
                native._exact_outputs(make_result("lmcache_cpu", make_observations()), "lmcache_cpu"),
                native._exact_outputs(disk_result, "lmcache_disk"),
                "native-dir", "cpu-dir", "disk-dir",
            )

    def test_detects_text_mismatch(self):
        native_result = {"observations": make_observations()}
        cpu_result = make_result("lmcache_cpu", make_observations())
        cpu_result["observations"][5]["cold"]["text"] = "WRONG"
        with self.assertRaisesRegex(GATE, "lmcache_cpu:5:cold:text"):
            native._compare_outputs(
                native._exact_outputs(native_result, "native_prefix"),
                native._exact_outputs(cpu_result, "lmcache_cpu"),
                native._exact_outputs(make_result("lmcache_disk", make_observations()), "lmcache_disk"),
                "native-dir", "cpu-dir", "disk-dir",
            )

    def test_reports_every_mismatched_pair(self):
        native_result = {"observations": make_observations()}
        cpu_result = make_result("lmcache_cpu", make_observations())
        disk_result = make_result("lmcache_disk", make_observations())
        cpu_result["observations"][0]["cold"]["text"] = "WRONG-TEXT"
        disk_result["observations"][7]["warm"]["generated_token_ids"][0] += 1
        with self.assertRaisesRegex(GATE, "lmcache_cpu:0:cold:text") as raised:
            native._compare_outputs(
                native._exact_outputs(native_result, "native_prefix"),
                native._exact_outputs(cpu_result, "lmcache_cpu"),
                native._exact_outputs(disk_result, "lmcache_disk"),
                "native-dir", "cpu-dir", "disk-dir",
            )
        self.assertIn("lmcache_disk:7:warm:token_ids", str(raised.exception))

    def test_rejects_legacy_result_missing_token_ids(self):
        legacy = make_result("lmcache_cpu", make_observations())
        for observation in legacy["observations"]:
            for phase in ("cold", "warm"):
                del observation[phase]["generated_token_ids"]
        with self.assertRaisesRegex(GATE, "lacks exact generated token IDs"):
            native._exact_outputs(legacy, "lmcache_cpu")

    def test_rejects_missing_phase(self):
        disk_result = make_result("lmcache_disk", make_observations())
        del disk_result["observations"][4]["warm"]
        with self.assertRaisesRegex(GATE, "lacks the warm request for prefix 4"):
            native._exact_outputs(disk_result, "lmcache_disk")

    def test_rejects_empty_text(self):
        cpu_result = make_result("lmcache_cpu", make_observations())
        cpu_result["observations"][1]["warm"]["text"] = ""
        with self.assertRaisesRegex(GATE, "lacks generated text"):
            native._exact_outputs(cpu_result, "lmcache_cpu")

    def test_rejects_incomplete_phase_set(self):
        native_result = {"observations": make_observations()}
        cpu_result = make_result("lmcache_cpu", make_observations()[:7])
        with self.assertRaisesRegex(GATE, "exactly 8 observations"):
            native._exact_outputs(cpu_result, "lmcache_cpu")
        native_outputs = native._exact_outputs(native_result, "native_prefix")
        partial = dict(list(native_outputs.items())[:15])
        with self.assertRaisesRegex(GATE, "comparison phases differ"):
            native._compare_outputs(
                native_outputs,
                partial,
                native._exact_outputs(make_result("lmcache_disk", make_observations()), "lmcache_disk"),
                "native-dir", "cpu-dir", "disk-dir",
            )


class EndToEndSuccessTests(unittest.TestCase):
    def test_full_success_path_with_frozen_prompts(self):
        prompts = native.runner.load_prompts(native.runner.PROMPTS)
        observations = []
        for item in prompts["prefixes"]:
            index = item["index"]
            observations.append({
                "prefix_index": index,
                "expected_hit_tokens": item["expected_hit_tokens"],
                "cold": make_response(
                    f"native-p{index}-cold",
                    list(range(3000 * index, 3000 * index + OUTPUT_TOKENS)),
                    f"COLD-{index}",
                    len(item["cold_token_ids"]),
                    0,
                ),
                "warm": make_response(
                    f"native-p{index}-warm",
                    list(range(4000 * index, 4000 * index + OUTPUT_TOKENS)),
                    f"WARM-{index}",
                    len(item["warm_token_ids"]),
                    item["expected_hit_tokens"],
                ),
            })
        native._validate_prompt_observations(prompts, observations)
        for item, observation in zip(prompts["prefixes"], observations, strict=True):
            index = item["index"]
            native._validate_response(
                observation["cold"], len(item["cold_token_ids"]), f"native-p{index}-cold"
            )
            native._validate_response(
                observation["warm"], len(item["warm_token_ids"]), f"native-p{index}-warm"
            )
        evidence = native._validate_engagement("ordinary server output", observations)
        self.assertEqual(
            evidence["prompt_tokens_details_cached_tokens"],
            [{"prefix_index": index, "cold_warm": [0, 1536]} for index in range(8)],
        )
        native_outputs = native._exact_outputs({"observations": observations}, "native_prefix")
        self.assertEqual(len(native_outputs), 16)
        summary = native._compare_outputs(
            native_outputs,
            native._exact_outputs({"observations": observations}, "lmcache_cpu"),
            native._exact_outputs({"observations": observations}, "lmcache_disk"),
            "native-dir", "cpu-dir", "disk-dir",
        )
        self.assertIs(summary["exact_token_ids_equal"], True)
        self.assertIs(summary["exact_text_equal"], True)


if __name__ == "__main__":
    unittest.main()
