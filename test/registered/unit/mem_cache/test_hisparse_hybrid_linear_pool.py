"""HiSparse on a hybrid linear-attention pool (e.g. GLM-5.3-Flash).

HybridLinearKVPool wraps a dense full-attention pool. With HiSparse that pool
is a HiSparseDSATokenToKVPool: the wrapper must delegate the HiSparse slot
mapping, and HiSparse per-layer state must use dense layer indices.
"""

import unittest

import torch

from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _hybrid_pool(full_pool, full_layer_ids):
    pool = object.__new__(HybridLinearKVPool)
    pool.full_kv_pool = full_pool
    pool.full_attention_layer_id_mapping = {
        layer_id: i for i, layer_id in enumerate(full_layer_ids)
    }
    return pool


class TestHiSparseHybridLinearPool(unittest.TestCase):
    def test_delegates_mapping_and_translation(self):
        inner = object.__new__(HiSparseDSATokenToKVPool)
        pool = _hybrid_pool(inner, [3, 7, 11])
        self.assertTrue(pool.is_hisparse)

        mapping = torch.tensor([0, 9, 8, 7, 6, -1], dtype=torch.int64)
        pool.register_mapping(mapping)
        self.assertIs(pool.full_to_hisparse_device_index_mapping, mapping)

        locs = torch.tensor([1, 3, 4])
        torch.testing.assert_close(
            pool.translate_loc_to_hisparse_device(locs), torch.tensor([9, 7, 6])
        )
        torch.testing.assert_close(
            pool.translate_loc_from_full_to_hisparse_device(locs),
            torch.tensor([9, 7, 6]),
        )
        torch.testing.assert_close(
            pool.translate_loc_from_full_to_compressed(locs), locs
        )

    def test_dense_layer_index(self):
        pool = _hybrid_pool(object.__new__(HiSparseDSATokenToKVPool), [3, 7, 11])
        self.assertEqual(pool.hisparse_layer_index(3), 0)
        self.assertEqual(pool.hisparse_layer_index(11), 2)
        with self.assertRaises(ValueError):
            pool.hisparse_layer_index(4)

    def test_plain_dsa_pool_is_not_hisparse(self):
        pool = _hybrid_pool(object(), [3, 7])
        self.assertFalse(pool.is_hisparse)


if __name__ == "__main__":
    unittest.main()
