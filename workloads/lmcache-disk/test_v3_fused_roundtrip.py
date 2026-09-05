#!/usr/bin/env python3
"""Production-path contract and opt-in GPU roundtrip for fused NHD KV."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV3
from lmcache.v1.memory_allocators.pin_memory_allocator import PinMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
import lmcache.lmcache_native as lmcache_native


NB = 8
NL = 3
BS = 16
NH = 4
HS = 128
CS = 2 * HS
TOKENS = 32


def logical_nhd_views(physical_layers: list[torch.Tensor]) -> list[torch.Tensor]:
    """Expose vLLM's logical [NB,NH,BS,CS] view over physical NHD storage."""
    return [layer.permute(0, 2, 1, 3) for layer in physical_layers]


def metadata(chunk_size: int = 256) -> LMCacheMetadata:
    """Return the legacy metadata shape emitted before cache registration."""
    return LMCacheMetadata(
        model_name="v3-fused-roundtrip",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(NL, 2, 256, NH, HS),
        use_mla=False,
        chunk_size=chunk_size,
    )


class V3FusedContractTests(unittest.TestCase):
    def register(self, layout: LMCacheMetadata, caches: list[torch.Tensor]):
        with patch("torch.cuda.Stream", return_value=object()), patch(
            "lmcache.v1.gpu_connector.kv_format.detectors.vllm.torch_device_type",
            "cuda",
        ):
            connector = VLLMPagedMemGPUConnectorV3(
                layout,
                device=torch.device("cuda:0"),
                use_gpu=False,
                layout_hints={"kv_layout": "NHD"},
            )
            connector.register_kv_caches(caches)
        return connector

    def test_registration_precedes_post_init(self):
        events = []
        physical = [
            torch.empty(NB, BS, NH, CS, dtype=torch.bfloat16) for _ in range(NL)
        ]
        caches = {
            f"layer.{index}": tensor
            for index, tensor in enumerate(logical_nhd_views(physical))
        }
        layout = metadata()
        expected_shape = torch.Size((1, NL, TOKENS, NH * CS))
        with patch("torch.cuda.Stream", return_value=object()), patch(
            "lmcache.v1.gpu_connector.kv_format.detectors.vllm.torch_device_type",
            "cuda",
        ):
            gpu_connector = VLLMPagedMemGPUConnectorV3(
                layout,
                device=torch.device("cuda:0"),
                use_gpu=False,
                layout_hints={"kv_layout": "NHD"},
            )
            register = gpu_connector.register_kv_caches

            def record_registration(values):
                events.append(("register", None))
                register(values)

            gpu_connector.register_kv_caches = record_registration
            manager = SimpleNamespace(
                lmcache_engine=SimpleNamespace(gpu_connector=gpu_connector),
                post_init=lambda: events.append(
                    ("first_allocation", layout.get_shapes(TOKENS))
                ),
            )
            adapter = object.__new__(LMCacheConnectorV1Impl)
            adapter.kv_caches = {}
            adapter._manager = manager

            adapter.register_kv_caches(caches)

        self.assertEqual(
            [event[0] for event in events],
            ["register", "first_allocation"],
        )
        self.assertEqual(events[1][1], [expected_shape])

    def test_v3_registration_updates_fused_geometry_before_allocation(self):
        physical = [
            torch.empty(NB, BS, NH, CS, dtype=torch.bfloat16) for _ in range(NL)
        ]
        logical = logical_nhd_views(physical)
        layout = metadata()

        connector = self.register(layout, logical)
        group_manager = layout.kv_layer_groups_manager
        assert group_manager is not None
        fmt = group_manager.kernel_groups[0].engine_kv_format
        normalized = connector._registered_kvcaches

        self.assertEqual(fmt, lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_CS)
        assert isinstance(normalized, list)
        for original, recovered in zip(physical, normalized, strict=True):
            self.assertEqual(tuple(recovered.shape), (NB, BS, NH, CS))
            self.assertEqual(tuple(recovered.stride()), tuple(original.stride()))
            self.assertEqual(recovered.data_ptr(), original.data_ptr())

        self.assertEqual(
            layout.get_shapes(TOKENS),
            [torch.Size((1, NL, TOKENS, NH * CS))],
        )
        self.assertEqual(layout.get_dtypes(), [torch.bfloat16])

    def test_v3_registration_preserves_split_and_legacy_geometry(self):
        layout = metadata()
        legacy_shape = torch.Size((2, NL, TOKENS, NH * HS))
        self.assertEqual(layout.get_shapes(TOKENS), [legacy_shape])
        split = [
            torch.empty(2, NB, BS, NH, HS, dtype=torch.bfloat16)
            for _ in range(NL)
        ]

        self.register(layout, split)

        self.assertEqual(layout.get_shapes(TOKENS), [legacy_shape])
        self.assertEqual(layout.get_dtypes(), [torch.bfloat16])


@unittest.skipUnless(
    os.environ.get("LMCACHE_RUN_V3_FUSED_GPU_ROUNDTRIP") == "1",
    "set LMCACHE_RUN_V3_FUSED_GPU_ROUNDTRIP=1 for the explicit GPU canary",
)
class V3FusedLiveRoundtripTests(unittest.TestCase):
    def test_production_detect_v3_d2h_h2d_roundtrip(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")

        device = torch.device("cuda:0")
        torch.manual_seed(2709)
        source_physical = [
            torch.randn(NB, BS, NH, CS, device=device, dtype=torch.bfloat16)
            for _ in range(NL)
        ]
        target_physical = [
            torch.zeros_like(layer) for layer in source_physical
        ]
        source = logical_nhd_views(source_physical)
        target = logical_nhd_views(target_physical)
        slots = torch.tensor(
            [3, 71, 9, 94, 17, 65, 31, 112, 7, 82, 19, 101, 27, 73, 43, 120,
             1, 69, 11, 92, 21, 67, 35, 110, 5, 84, 23, 99, 29, 75, 47, 118],
            device=device,
            dtype=torch.int64,
        )

        source_metadata = metadata(chunk_size=TOKENS)
        target_metadata = metadata(chunk_size=TOKENS)
        source_connector = VLLMPagedMemGPUConnectorV3(
            source_metadata,
            device=device,
            use_gpu=True,
            layout_hints={"kv_layout": "NHD"},
        )
        target_connector = VLLMPagedMemGPUConnectorV3(
            target_metadata,
            device=device,
            use_gpu=False,
            layout_hints={"kv_layout": "NHD"},
        )

        source_connector.register_kv_caches(source)
        source_connector._initialize_kv_cache_pointers()
        assert source_connector.group_tmp_buffer is not None
        self.assertEqual(
            [tuple(tensor.shape) for tensor in source_connector.group_tmp_buffer],
            [(1, NL, TOKENS, NH * CS)],
        )
        target_connector.register_kv_caches(target)
        target_connector._initialize_kv_cache_pointers()
        expected_shape = torch.Size((1, NL, TOKENS, NH * CS))
        self.assertEqual(source_metadata.get_shapes(TOKENS), [expected_shape])
        self.assertEqual(target_metadata.get_shapes(TOKENS), [expected_shape])

        arena_bytes = expected_shape.numel() * torch.bfloat16.itemsize + 4096
        allocator = PinMemoryAllocator(arena_bytes)
        try:
            memory_obj = allocator.allocate(
                source_metadata.get_shapes(TOKENS),
                source_metadata.get_dtypes(),
                MemoryFormat.KV_2LTD,
            )
            self.assertIsNotNone(memory_obj)
            assert memory_obj is not None
            source_connector.from_gpu(
                memory_obj, 0, TOKENS, kvcaches=source, slot_mapping=slots
            )
            target_connector.to_gpu(
                memory_obj, 0, TOKENS, kvcaches=target, slot_mapping=slots
            )
            torch.cuda.synchronize(device)

            for source_layer, target_layer in zip(
                source_physical, target_physical, strict=True
            ):
                source_tokens = source_layer.view(NB * BS, NH, CS).index_select(0, slots)
                target_tokens = target_layer.view(NB * BS, NH, CS).index_select(0, slots)
                self.assertTrue(torch.equal(source_tokens, target_tokens))
            allocator.free(memory_obj)
            self.assertTrue(allocator.memcheck())
        finally:
            allocator.close()


if __name__ == "__main__":
    unittest.main()
