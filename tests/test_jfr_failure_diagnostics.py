from __future__ import annotations

import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def braced_body(source: str, anchor: str) -> str:
    """Return one C declaration/function body without matching its text."""
    anchor_index = source.index(anchor)
    body_start = source.index("{", anchor_index)
    depth = 0
    for index in range(body_start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[body_start + 1:index]
    raise AssertionError(f"unterminated C body after {anchor!r}")


def enum_members(source: str, enum_name: str) -> set[str]:
    body = braced_body(source, f"enum {enum_name}")
    return set(re.findall(r"\b(KVAF_UDMA_JFR_FAILURE_[A-Z_]+)\b", body))


class JfrFailureDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.header = read_source(
            "access_operator/a5/provider/include/kvaf_source_matched_udma.h"
        )
        cls.jfr = read_source(
            "access_operator/a5/provider/source_matched_udma_jfr.c"
        )
        cls.communication = read_source(
            "access_operator/a5/provider/source_matched_communication.c"
        )
        cls.provider = read_source(
            "access_operator/a5/provider/urma_perftest_provider.c"
        )
        cls.perftest = read_source(
            "access_operator/a5/provider/source_matched_perftest.c"
        )
        cls.ops = read_source(
            "access_operator/a5/provider/source_matched_udma_ops.c"
        )

    def test_failure_abi_is_versioned_sized_and_queryable(self) -> None:
        self.assertRegex(
            self.header,
            r"#define\s+KVAF_UDMA_JFR_FAILURE_ABI_VERSION\s+UINT32_C\(1\)",
        )

        structure = braced_body(
            self.header, "struct kvaf_udma_jfr_failure"
        )
        fields = set(
            re.findall(
                r"\b(?:u?int(?:32|64)_t)\s+([a-z][a-z0-9_]*)\s*;",
                structure,
            )
        )
        self.assertTrue(
            {
                "abi_version",
                "structure_size",
                "stage",
                "operation_status",
                "operation_errno",
                "rollback_status",
                "rollback_errno",
                "requested_depth",
                "requested_max_sge",
                "device_max_depth",
                "device_max_sge",
                "context_jfr_sge",
            }.issubset(fields)
        )

        getter = braced_body(
            self.jfr, "int kvaf_udma_source_get_jfr_failure("
        )
        self.assertRegex(
            getter,
            r"failure\s*==\s*NULL[\s\S]*?"
            r"failure->structure_size\s*<\s*sizeof\(\*failure\)",
        )
        self.assertRegex(
            getter,
            r"\*failure\s*=\s*g_kvaf_jfr_failure\s*;",
        )
        self.assertRegex(
            getter,
            r"failure->abi_version\s*=\s*"
            r"KVAF_UDMA_JFR_FAILURE_ABI_VERSION\s*;",
        )
        self.assertRegex(
            getter,
            r"failure->structure_size\s*=\s*sizeof\(\*failure\)\s*;",
        )
        self.assertIn(
            "void kvaf_udma_source_reset_jfr_failure(void);",
            self.header,
        )

    def test_every_create_stage_records_the_first_leaf_failure(self) -> None:
        declared = enum_members(
            self.header, "kvaf_udma_jfr_failure_stage"
        )
        create = braced_body(
            self.jfr, "urma_jfr_t *kvaf_udma_create_jfr_checked("
        )
        recorded = set(
            re.findall(
                r"kvaf_record_jfr_failure\s*\(\s*"
                r"(KVAF_UDMA_JFR_FAILURE_[A-Z_]+)",
                create,
            )
        )
        critical_stages = {
            "KVAF_UDMA_JFR_FAILURE_VALIDATE_ARGUMENTS",
            "KVAF_UDMA_JFR_FAILURE_ALLOCATE_OWNER",
            "KVAF_UDMA_JFR_FAILURE_LOCK_REGISTRY",
            "KVAF_UDMA_JFR_FAILURE_VALIDATE_CONFIG",
            "KVAF_UDMA_JFR_FAILURE_VALIDATE_CONTEXT",
            "KVAF_UDMA_JFR_FAILURE_INIT_SPIN",
            "KVAF_UDMA_JFR_FAILURE_ALLOCATE_INDEX",
            "KVAF_UDMA_JFR_FAILURE_COMPUTE_RQ",
            "KVAF_UDMA_JFR_FAILURE_ALLOCATE_RQ",
            "KVAF_UDMA_JFR_FAILURE_ALLOCATE_RECORD_DB",
            "KVAF_UDMA_JFR_FAILURE_ALLOCATE_SLEEP_DB",
            "KVAF_UDMA_JFR_FAILURE_CREATE_COMMAND",
            "KVAF_UDMA_JFR_FAILURE_INSERT_TABLE",
            "KVAF_UDMA_JFR_FAILURE_UNLOCK_REGISTRY",
        }
        self.assertTrue(critical_stages.issubset(declared))
        self.assertEqual(critical_stages - recorded, set())

        recorder = braced_body(
            self.jfr, "static void kvaf_record_jfr_failure("
        )
        guard = recorder.index(
            "g_kvaf_jfr_failure.stage != KVAF_UDMA_JFR_FAILURE_NONE"
        )
        early_return = recorder.index("return;", guard)
        assignment = recorder.index("g_kvaf_jfr_failure.stage =", guard)
        self.assertLess(guard, early_return)
        self.assertLess(early_return, assignment)

        self.assertGreaterEqual(
            len(re.findall(r"\bkvaf_record_jfr_rollback\s*\(", create)),
            2,
        )
        self.assertGreaterEqual(
            len(re.findall(r"\berrno\s*=\s*root_errno\s*;", create)),
            2,
        )

    def test_management_reports_provider_stage_and_leaf_errno(self) -> None:
        declared_stages = enum_members(
            self.header, "kvaf_udma_jfr_failure_stage"
        )
        stage_names = braced_body(
            self.communication,
            "static const char *kvaf_jfr_failure_stage_name(",
        )
        named_stages = set(
            re.findall(
                r"case\s+(KVAF_UDMA_JFR_FAILURE_[A-Z_]+)\s*:",
                stage_names,
            )
        )
        self.assertEqual(declared_stages - named_stages, set())

        helper = braced_body(
            self.communication, "static int kvaf_ub_jfr_creation_error("
        )
        self.assertRegex(
            helper,
            r"kvaf_udma_source_get_jfr_failure\s*\(\s*&failure\s*\)",
        )
        for validity_guard in (
            "failure.abi_version == KVAF_UDMA_JFR_FAILURE_ABI_VERSION",
            "failure.structure_size == sizeof(failure)",
            "failure.stage != KVAF_UDMA_JFR_FAILURE_NONE",
        ):
            self.assertIn(validity_guard, helper)
        self.assertRegex(
            helper,
            r"failure_valid\s*&&\s*failure\.operation_errno\s*!=\s*0"
            r"[\s\S]*?effective_errno\s*=\s*failure\.operation_errno",
        )
        self.assertRegex(
            helper,
            r"kvaf_ub_record_management_diagnostic\s*\("
            r"[\s\S]*?failure\.operation_status\s*,\s*failure_valid\s*,"
            r"[\s\S]*?\(int\)failure\.stage\s*,\s*failure_valid\s*\)",
        )
        self.assertRegex(
            helper,
            r"return\s+kvaf_ub_management_error\s*\(\s*"
            r"stage\s*,\s*effective_errno\s*,\s*fallback_errno\s*\)",
        )

        create_resources = braced_body(
            self.communication,
            "static int kvaf_ub_create_local_resources_checked(",
        )
        create_call = create_resources.index("urma_create_jfr(")
        failure_call = create_resources.index(
            "kvaf_ub_jfr_creation_error(", create_call
        )
        jetty_call = create_resources.index("urma_create_jetty(")
        self.assertLess(create_call, failure_call)
        self.assertLess(failure_call, jetty_call)
        self.assertRegex(
            create_resources[create_call:jetty_call],
            r"kvaf_ub_jfr_creation_error\s*\(\s*"
            r'"create_jfr"\s*,\s*errno\s*,\s*EIO\s*\)',
        )
        reset_call = create_resources.rindex(
            "kvaf_udma_source_reset_jfr_failure()", 0, create_call
        )
        self.assertLess(reset_call, create_call)

        data_create = braced_body(
            self.perftest, "static int kvaf_create_duplex_queues_public("
        )
        data_create_call = data_create.index("urma_create_jfr(")
        data_reset_call = data_create.rindex(
            "kvaf_udma_source_reset_jfr_failure()", 0, data_create_call
        )
        self.assertLess(data_reset_call, data_create_call)

    def test_unknown_cleanup_never_reenters_orphan_registries(self) -> None:
        aggregate = braced_body(
            self.ops, "int kvaf_udma_source_retry_orphan_cleanup("
        )
        retries = [
            "kvaf_udma_source_retry_orphan_jetties(context)",
            "kvaf_udma_source_retry_orphan_jfrs(context)",
            "kvaf_udma_source_retry_orphan_jfcs(context)",
            "kvaf_udma_source_retry_orphan_segments(context)",
            "kvaf_udma_source_retry_orphan_tokens(context)",
        ]
        def assert_unknown_guard(text: str) -> None:
            self.assertRegex(
                text,
                r"kvaf_udma_source_cleanup_is_unknown\(\)\s*!=\s*0\s*\|\|"
                r"\s*kvaf_source_ummu_cleanup_is_unknown\(\)\s*!=\s*0",
            )

        first_retry = aggregate.index(retries[0])
        assert_unknown_guard(aggregate[:first_retry])
        for current, following in zip(retries, retries[1:]):
            between = aggregate[
                aggregate.index(current) + len(current):
                aggregate.index(following)
            ]
            assert_unknown_guard(between)
        assert_unknown_guard(aggregate[aggregate.index(retries[-1]):])

        management_close = braced_body(
            self.communication,
            "static int kvaf_ub_close_connection_checked(",
        )
        management_retry = management_close.index(
            "kvaf_udma_source_retry_orphan_cleanup(ctx->urma_ctx)"
        )
        assert_unknown_guard(management_close[:management_retry])

        data_teardown = braced_body(
            self.perftest,
            "static int kvaf_destroy_partial_duplex_ctx_checked(",
        )
        orphan_case = data_teardown.index(
            "case KVAF_PERFTEST_TEARDOWN_ORPHAN_RESOURCES:"
        )
        orphan_retry = data_teardown.index(
            "kvaf_udma_source_retry_orphan_cleanup(", orphan_case
        )
        guarded_case = data_teardown[orphan_case:orphan_retry]
        assert_unknown_guard(guarded_case)
        self.assertIn("kvaf_teardown_mark_unsafe", guarded_case)

    def test_data_parse_explicitly_enables_shared_jfr(self) -> None:
        parser = braced_body(
            self.provider, "static int kvaf_parse_config("
        )
        arguments = re.findall(
            r"KVAF_ADD_ARG\s*\(\s*\"([^\"]+)\"\s*\)", parser
        )
        share_positions = [
            index
            for index, argument in enumerate(arguments)
            if argument == "--share_jfr"
        ]
        self.assertEqual(share_positions, [arguments.index("--share_jfr")])
        share_index = share_positions[0]
        self.assertLess(share_index + 1, len(arguments))
        self.assertEqual(arguments[share_index + 1], "true")

        parse_call = parser.index("perftest_parse_args(")
        share_call = parser.index('KVAF_ADD_ARG("--share_jfr")')
        self.assertLess(share_call, parse_call)


if __name__ == "__main__":
    unittest.main()
