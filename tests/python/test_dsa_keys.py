#!/usr/bin/env python3
"""Host-only regression checks for DSA keys used by BatchGet setup."""

import unittest

from dsa_keys import build_dsa_keys


class DsaKeyTests(unittest.TestCase):
    def test_multiple_pool_rows_and_layers(self):
        # pool_rows=2, n_layers=3, q_capacity=5; selected layer=1.
        self.assertEqual(
            build_dsa_keys(10, pool_rows=2, n_layers=3, q_capacity=5, layer=1),
            [5, 6, 7, 8, 9, 20, 21, 22, 23, 24],
        )
        self.assertEqual(
            build_dsa_keys(4, start=3, pool_rows=2, n_layers=3, q_capacity=5, layer=1),
            [8, 9, 20, 21],
        )

    def test_chunked_population_matches_prepare_order(self):
        for capacity, chunk_size in ((1, 128), (5, 2), (128, 128), (129, 128)):
            with self.subTest(capacity=capacity, chunk_size=chunk_size):
                keys = []
                for start in range(0, capacity, chunk_size):
                    keys.extend(build_dsa_keys(
                        min(chunk_size, capacity - start), start=start,
                        q_capacity=capacity,
                    ))
                self.assertEqual(keys, list(range(capacity)))

    def test_sweep_capacity_last_chunk_stays_inside_region(self):
        capacity = 16_384 * 78
        keys = build_dsa_keys(128, start=capacity - 128, q_capacity=capacity)
        self.assertEqual(keys[0], capacity - 128)
        self.assertEqual(keys[-1], capacity - 1)
        with self.assertRaises(ValueError):
            build_dsa_keys(128, start=capacity - 127, q_capacity=capacity)

    def test_variable_and_main_put_use_legal_dsa_keys(self):
        probe = build_dsa_keys(3, q_capacity=11)
        self.assertEqual(probe, [0, 1, 2])
        active = build_dsa_keys(8, q_capacity=11)
        self.assertEqual(active, list(range(8)))
        self.assertTrue(all(0 <= key < 11 for key in probe + active))

    def test_invalid_geometry_and_ranges_are_rejected(self):
        for options in (
            dict(q_capacity=0), dict(q_capacity=1, pool_rows=0),
            dict(q_capacity=1, n_layers=0), dict(q_capacity=1, layer=-1),
            dict(q_capacity=1, layer=1), dict(q_capacity=1, start=-1),
            dict(q_capacity=1, start=1), dict(q_capacity=1 << 31),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                build_dsa_keys(1, **options)


if __name__ == "__main__":
    unittest.main()
