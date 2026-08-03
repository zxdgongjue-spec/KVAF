#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class JfrManagementFlagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.abi = (
            ROOT
            / "access_operator"
            / "a5"
            / "provider"
            / "include"
            / "kvaf_urma_provider_abi.h"
        ).read_text(encoding="utf-8")
        cls.communication = (
            ROOT
            / "access_operator"
            / "a5"
            / "provider"
            / "source_matched_communication.c"
        ).read_text(encoding="utf-8")
        cls.resources = (
            ROOT
            / "access_operator"
            / "a5"
            / "src"
            / "direct_urma_resources.cpp"
        ).read_text(encoding="utf-8")
        cls.fake = (
            ROOT / "tests" / "fake_acl" / "fake_ascendcl.cpp"
        ).read_text(encoding="utf-8")
        cls.harness = (
            ROOT
            / "tests"
            / "fake_acl"
            / "direct_urma_resources_harness.cpp"
        ).read_text(encoding="utf-8")

    def test_flags_extend_existing_management_word(self) -> None:
        self.assertIn(
            "KVAF_URMA_MANAGEMENT_DIAGNOSTIC_JFR_CONTEXT_SVA_SEPARATE = "
            "1U << 4",
            self.abi,
        )
        self.assertIn(
            "KVAF_URMA_MANAGEMENT_DIAGNOSTIC_JFR_USER_ANONYMOUS_MEMORY = "
            "1U << 5",
            self.abi,
        )

    def test_provider_publishes_context_and_memory_policy(self) -> None:
        record = self.communication.index(
            "kvaf_ub_record_management_diagnostic(\n"
            "        stage, wrapper_errno, effective_errno,"
        )
        context_flag = self.communication.index(
            "KVAF_URMA_MANAGEMENT_DIAGNOSTIC_JFR_CONTEXT_SVA_SEPARATE",
            record,
        )
        memory_marker = self.communication.index(
            '"anonymous-pinned-known-working-v1"', record
        )
        memory_flag = self.communication.index(
            "KVAF_URMA_MANAGEMENT_DIAGNOSTIC_JFR_USER_ANONYMOUS_MEMORY",
            memory_marker,
        )
        self.assertLess(record, context_flag)
        self.assertLess(record, memory_marker)
        self.assertLess(memory_marker, memory_flag)

    def test_host_error_formats_both_bits(self) -> None:
        self.assertIn('" jfr_context_sva_separate="', self.resources)
        self.assertIn('" jfr_user_anonymous_memory="', self.resources)

    def test_fake_round_trip_covers_selected_anonymous_policy(self) -> None:
        self.assertIn(
            "KVAF_URMA_MANAGEMENT_DIAGNOSTIC_JFR_USER_ANONYMOUS_MEMORY",
            self.fake,
        )
        self.assertIn('"jfr_context_sva_separate=0"', self.harness)
        self.assertIn('"jfr_user_anonymous_memory=1"', self.harness)


if __name__ == "__main__":
    unittest.main()
