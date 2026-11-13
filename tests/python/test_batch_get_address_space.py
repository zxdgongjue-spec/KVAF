"""Host-only checks for BatchGet remote address-space budgets."""

import os
import unittest
from unittest.mock import patch

import test_npu_batch_get_perf as perf


class AddressSpaceTests(unittest.TestCase):
    def test_capacity_and_budget_precedence(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(perf.parse_args([]).table_capacity, perf.DEFAULT_TABLE_CAPACITY)
            self.assertEqual(perf.parse_args(["--table-capacity", "7"]).table_capacity, 7)
        env = {"KVAF_BATCH_GET_PERF_TABLE_CAPACITY": "7",
               "KVAF_BATCH_GET_PERF_ADDRESS_SPACE_BYTES": "1048576"}
        with patch.dict(os.environ, env, clear=True):
            args = perf.parse_args(["--table-capacity", "99"])
            self.assertIsNone(perf.validate_args(args))
            self.assertEqual(args.table_capacity, 12)
            args = perf.parse_args(["--address-space-bytes", "83968"])
            self.assertIsNone(perf.validate_args(args))
            self.assertEqual(args.table_capacity, 1)

    def test_invalid_budgets_and_layouts(self):
        for budget, token, block, field in (
            (0, 1, 1, "address-space-bytes"),
            (-1, 1, 1, "address-space-bytes"),
            (1 << 64, 1, 1, "address-space-bytes"),
            (83967, 656, 128, "one complete block"),
            (1 << 31, 1, 1, "table-capacity"),
            (100, 0, 1, "token-bytes"),
            (100, 1, 0, "tokens-per-block"),
        ):
            with self.subTest(budget=budget, token=token, block=block), patch.dict(os.environ, {}, clear=True):
                args = perf.parse_args(["--address-space-bytes", str(budget), "--token-bytes", str(token),
                                        "--tokens-per-block", str(block)])
                self.assertIn(field, perf.validate_args(args))


if __name__ == "__main__":
    unittest.main()
