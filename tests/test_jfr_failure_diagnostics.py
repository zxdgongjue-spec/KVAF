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
        cls.jfc = read_source(
            "access_operator/a5/provider/source_matched_udma_jfc.c"
        )
        cls.jetty = read_source(
            "access_operator/a5/provider/source_matched_udma_jetty.c"
        )
        cls.cleanup = read_source(
            "access_operator/a5/provider/include/kvaf_source_matched_udma_cleanup.h"
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
            r"#define\s+KVAF_UDMA_JFR_FAILURE_ABI_VERSION\s+UINT32_C\(2\)",
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
                "context_page_size",
                "context_sva_sep_mode",
                "command_size",
                "response_size",
                "udata_in_len",
                "udata_out_len",
                "udata_in_address",
                "udata_out_address",
                "command_flags",
                "jfc_id",
                "jfc_handle",
                "rq_address",
                "rq_length",
                "index_address",
                "index_length",
                "record_db_address",
                "sleep_db_address",
                "context_fd_errno",
                "jfc_registry_flags",
                "jfc_created_handle",
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
        create_call = create_resources.index(
            "kvaf_udma_source_create_management_jfr("
        )
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

    def test_create_command_snapshot_is_observation_only(self) -> None:
        command = braced_body(
            self.jfr, "static int kvaf_create_direct_jfr_command("
        )
        snapshot = command.index("kvaf_snapshot_jfr_command(")
        ioctl = command.index("urma_cmd_create_jfr(", snapshot)
        self.assertLess(snapshot, ioctl)
        between = command[snapshot:ioctl]
        self.assertNotIn("return", between)
        self.assertNotIn("EINVAL", between)
        self.assertEqual(command.count("urma_cmd_create_jfr("), 1)
        self.assertNotIn("kvaf_jfr_command_snapshot_is_valid", self.jfr)

        create = braced_body(
            self.jfr, "urma_jfr_t *kvaf_udma_create_jfr_checked("
        )
        self.assertIn("KVAF_UDMA_JFR_FAILURE_CREATE_COMMAND", create)
        self.assertNotIn("KVAF_UDMA_JFR_FAILURE_VALIDATE_COMMAND", self.header)

        snapshot_body = braced_body(
            self.jfr, "static void kvaf_snapshot_jfr_command("
        )
        for diagnostic in (
            "KVAF_UDMA_JFR_COMMAND_RQ_RANGE_VALID",
            "KVAF_UDMA_JFR_COMMAND_INDEX_RANGE_VALID",
            "KVAF_UDMA_JFR_COMMAND_INPUT_ADDRESS_MATCH",
            "KVAF_UDMA_JFR_COMMAND_OUTPUT_ADDRESS_MATCH",
            "KVAF_UDMA_JFR_COMMAND_JFC_REGISTRY_IDENTITY_MATCH",
            "context_fd_errno",
        ):
            self.assertIn(diagnostic, snapshot_body)
        self.assertIn("UDMA_HW_PAGE_SIZE", snapshot_body)
        self.assertNotRegex(snapshot_body, r"buf_addr\s*%\s*page_size")

    def test_jfc_registry_preserves_creation_time_identity(self) -> None:
        state = braced_body(self.jfc, "struct kvaf_udma_jfc_state")
        for field in (
            "created_owner_context",
            "created_handle",
            "created_id",
            "created_dev_fd",
        ):
            self.assertIn(field, state)
        create = braced_body(
            self.jfc, "urma_jfc_t *kvaf_udma_create_jfc_checked("
        )
        command = create.index("udma_u_jfc_cmd(")
        for assignment in (
            "state->created_owner_context = context",
            "state->created_handle = jfc->base.handle",
            "state->created_id = jfc->base.jfc_id.id",
            "state->created_dev_fd = context->dev_fd",
        ):
            self.assertGreater(create.index(assignment), command)
        snapshot = braced_body(
            self.jfc, "int kvaf_udma_source_snapshot_jfc_identity("
        )
        for required in (
            "KVAF_UDMA_JFC_IDENTITY_REGISTRY_FOUND",
            "KVAF_UDMA_JFC_IDENTITY_KERNEL_CREATED",
            "KVAF_UDMA_JFC_IDENTITY_NOT_ORPHAN",
            "KVAF_UDMA_JFC_IDENTITY_HANDLE_MATCH",
            "KVAF_UDMA_JFC_IDENTITY_ID_MATCH",
            "KVAF_UDMA_JFC_IDENTITY_DEV_FD_MATCH",
        ):
            self.assertIn(required, snapshot)

    def test_device_result_carries_compact_jfr_snapshot(self) -> None:
        abi = read_source(
            "access_operator/a5/provider/include/kvaf_urma_provider_abi.h"
        )
        direct = read_source(
            "access_operator/a5/src/direct_urma_resources.cpp"
        )
        attach = braced_body(
            self.provider,
            "static void kvaf_attach_jfr_command_diagnostic(",
        )
        for required in (
            "KVAF_URMA_JFR_COMMAND_DIAGNOSTIC_RESERVED_HANDLE",
            "KVAF_URMA_JFR_COMMAND_DIAGNOSTIC_RESERVED_LAYOUT",
        ):
            self.assertIn(required, abi)
            self.assertIn(required, attach)
            self.assertIn(required, direct)
        self.assertIn(
            "kvaf_attach_jfr_command_diagnostic(result, diagnostic.stage)",
            braced_body(
                self.provider,
                "static void kvaf_attach_management_diagnostic(",
            ),
        )
        self.assertIn("jfr_command_diagnostic=valid", direct)
        self.assertIn("std::uint64_t jfc_handle", direct)
        self.assertNotIn("failure.jfc_handle == 0U", attach)
        self.assertNotIn("diagnostic.jfc_handle == 0U", direct)
        fake = read_source("tests/fake_acl/fake_ascendcl.cpp")
        harness = read_source(
            "tests/fake_acl/direct_urma_resources_harness.cpp"
        )
        self.assertIn("0x12345678abcdef01ULL", fake)
        self.assertIn("opening_jfr_failed_safe", harness)
        self.assertIn("CREATE_JFR compact diagnostic did not round-trip", harness)

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

    def test_device_management_jfr_uses_one_call_kernel_buffer_policy(
        self,
    ) -> None:
        policy = braced_body(
            self.cleanup, "enum kvaf_udma_queue_memory_policy"
        )
        self.assertIn("KVAF_UDMA_QUEUE_MEMORY_AUTO", policy)
        self.assertIn("KVAF_UDMA_QUEUE_MEMORY_KERNEL_BUFFER", policy)

        allocator = braced_body(
            self.cleanup,
            "static inline int kvaf_udma_checked_alloc_queue_buffer_policy(",
        )
        self.assertRegex(
            allocator,
            r"memory_policy\s*==\s*KVAF_UDMA_QUEUE_MEMORY_AUTO"
            r"[\s\S]*?queue->ctx->hugepage_enable",
        )
        self.assertIn("UDMA_MMAP_KERNEL_BUF", allocator)
        self.assertNotIn("queue->ctx->hugepage_enable =", allocator)

        default_allocator = braced_body(
            self.cleanup,
            "static inline int kvaf_udma_checked_alloc_queue_buffer(",
        )
        self.assertIn(
            "KVAF_UDMA_QUEUE_MEMORY_AUTO", default_allocator
        )

        checked_create = braced_body(
            self.jfr, "urma_jfr_t *kvaf_udma_create_jfr_checked("
        )
        self.assertIn(
            "rq_memory_policy = g_kvaf_next_jfr_rq_memory",
            checked_create,
        )
        self.assertIn(
            "g_kvaf_next_jfr_rq_memory = KVAF_UDMA_QUEUE_MEMORY_AUTO",
            checked_create,
        )
        self.assertRegex(
            checked_create,
            r"kvaf_udma_checked_alloc_queue_buffer_policy\s*\("
            r"[\s\S]*?&jfr->rq[\s\S]*?rq_memory_policy\s*\)",
        )

        management_wrapper = braced_body(
            self.jfr,
            "urma_jfr_t *kvaf_udma_source_create_management_jfr(",
        )
        self.assertIn("KVAF_UDMA_PROVIDER_ROLE_DEVICE", management_wrapper)
        self.assertIn(
            "KVAF_UDMA_QUEUE_MEMORY_KERNEL_BUFFER", management_wrapper
        )
        self.assertEqual(management_wrapper.count("urma_create_jfr("), 2)
        self.assertGreaterEqual(
            management_wrapper.count(
                "g_kvaf_next_jfr_rq_memory = "
                "KVAF_UDMA_QUEUE_MEMORY_AUTO"
            ),
            1,
        )
        self.assertNotIn("hugepage_enable =", management_wrapper)

        management_create = braced_body(
            self.communication,
            "static int kvaf_ub_create_local_resources_checked(",
        )
        self.assertIn(
            "kvaf_udma_source_create_management_jfr(",
            management_create,
        )
        self.assertIn(
            "KVAF_URMA_MGMT_JFR_RQ_MEMORY", management_create
        )

        data_create = braced_body(
            self.perftest, "static int kvaf_create_duplex_queues_public("
        )
        self.assertIn("urma_create_jfr(", data_create)
        self.assertNotIn(
            "kvaf_udma_source_create_management_jfr(", data_create
        )

        for source in (self.jfc, self.jetty):
            self.assertIn("kvaf_udma_checked_alloc_queue_buffer(", source)
            self.assertNotIn(
                "kvaf_udma_checked_alloc_queue_buffer_policy(", source
            )

        cmake = read_source("access_operator/a5/provider/CMakeLists.txt")
        self.assertIn("KVAF_UDMA_PROVIDER_ROLE_HOST=1", cmake)
        self.assertIn("KVAF_UDMA_PROVIDER_ROLE_DEVICE=1", cmake)

        manifest = read_source("scripts/a5_direct_urma_manifest_v3.sh")
        builder = read_source("scripts/build_a5_direct_urma_provider.sh")
        package = read_source("scripts/package_a5_direct_urma_aicpu.sh")
        contract = (
            "management_device_jfr_rq_memory:"
            "kernel-buffer-pinned-v1"
        )
        self.assertIn(contract, manifest)
        self.assertIn(
            "management_device_jfr_rq_memory="
            "kernel-buffer-pinned-v1",
            builder,
        )
        self.assertIn(
            "MANAGEMENT_DEVICE_JFR_RQ_MEMORY="
            "kernel-buffer-pinned-v1",
            package,
        )


if __name__ == "__main__":
    unittest.main()
