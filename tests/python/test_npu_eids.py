#!/usr/bin/env python3
"""Portable validation for explicit multi-EID test configuration."""

from __future__ import annotations

import unittest

from npu_eids import parse_npu_eids


class NpuEidParsingTest(unittest.TestCase):
    def test_empty_configuration_preserves_auto_discovery(self) -> None:
        self.assertEqual(parse_npu_eids(None), [])
        self.assertEqual(parse_npu_eids("   "), [])

    def test_multiple_eids_are_normalized_in_order(self) -> None:
        self.assertEqual(
            parse_npu_eids(
                "00112233445566778899AABBCCDDEEFF,"
                "102132435465768798A9BACBDCEDFE0F"
            ),
            [
                "00112233445566778899aabbccddeeff",
                "102132435465768798a9bacbdcedfe0f",
            ],
        )

    def test_invalid_or_duplicate_eids_are_rejected(self) -> None:
        for value in (
            "abc",
            "00:11:22:33:44:55:66:77:88:99:aa:bb:cc:dd:ee:ff",
            "0" * 32,
            "00112233445566778899aabbccddeeff,",
            "00112233445566778899aabbccddeeff,00112233445566778899aabbccddeeff",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_npu_eids(value)


if __name__ == "__main__":
    unittest.main()
