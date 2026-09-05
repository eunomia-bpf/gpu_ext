#!/usr/bin/env python3
"""Production-path contract and opt-in GPU roundtrip for fused NHD KV."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV3
from lmcache.v1.gpu_connector.kv_format import detect_format
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
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


def metadata() -> LMCacheMetadata:
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
        chunk_size=256,
    )


class V3FusedContractTests(unittest.TestCase):
    def test_production_detection_and_v3_geometry_keep_full_packed_width(self):
        physical = [
            torch.empty(NB, BS, NH, CS, dtype=torch.bfloat16) for _ in range(NL)
        ]
        logical = logical_nhd_views(physical)

        with patch(
            "lmcache.v1.gpu_connector.kv_format.detectors.vllm.torch_device_type",
            "cuda",
        ):
            fmt, normalized = detect_format(
                logical, EngineType.VLLM, {"kv_layout": "NHD"}
            )

        self.assertEqual(fmt, lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_CS)
        for original, recovered in zip(physical, normalized, strict=True):
            self.assertEqual(tuple(recovered.shape), (NB, BS, NH, CS))
            self.assertEqual(tuple(recovered.stride()), tuple(original.stride()))
            self.assertEqual(recovered.data_ptr(), original.data_ptr())

        layout = metadata()
        layout.kv_layer_groups_manager = KVLayerGroupsManager(
            normalized, engine_kv_formats=[fmt] * NL
        )
        self.assertEqual(layout.get_shapes(TOKENS), [torch.Size((1, NL, TOKENS, NH * CS))])
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

        source_metadata = metadata()
        target_metadata = metadata()
        source_connector = VLLMPagedMemGPUConnectorV3(
            source_metadata,
            device=device,
            use_gpu=False,
            layout_hints={"kv_layout": "NHD"},
        )
        target_connector = VLLMPagedMemGPUConnectorV3(
            target_metadata,
            device=device,
            use_gpu=False,
            layout_hints={"kv_layout": "NHD"},
        )

        source_connector.kvcaches = source
        source_connector._initialize_kv_cache_pointers()
        target_connector.kvcaches = target
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
