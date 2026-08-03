import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROVIDER = ROOT / "access_operator" / "a5" / "provider"


class SourceMatchedUdmaContractTests(unittest.TestCase):
    def read(self, relative: str) -> str:
        return (ROOT / relative).read_text(encoding="utf-8")

    def test_public_and_private_abi_come_from_one_official_umdk_tree(self):
        cmake = self.read("access_operator/a5/provider/CMakeLists.txt")
        self.assertIn(
            'set(KVAF_UMDK_URMA_ROOT "${KVAF_UMDK_ROOT}/src/urma")',
            cmake,
        )
        self.assertIn('"${KVAF_UDMA_SOURCE_DIR}/udma_u_common.h"', cmake)
        self.assertIn('"${KVAF_UDMA_UAPI_INCLUDE_DIR}/udma_abi.h"', cmake)
        self.assertNotIn("KVAF_URMA_PERFTEST_ROOT", cmake)
        self.assertNotIn("udma_public_abi_preinclude.h", cmake)
        self.assertIn("kvaf_urma_public_abi_probe", cmake)
        self.assertIn("kvaf_udma_uapi_probe", cmake)
        self.assertIn("kvaf_udma_private_abi_probe", cmake)

    def test_official_perftest_uses_ub_default_tcp_fallback_and_external_va(self):
        cmake = self.read("access_operator/a5/provider/CMakeLists.txt")
        communication = self.read(
            "access_operator/a5/provider/source_matched_communication.c"
        )
        resources = self.read(
            "access_operator/a5/provider/source_matched_perftest.c"
        )
        provider = self.read(
            "access_operator/a5/provider/urma_perftest_provider.c"
        )

        self.assertIn("perftest_mgmt_tcp.c", cmake)
        self.assertIn("KVAF_PERFTEST_MGMT_TCP_SOURCE", cmake)
        self.assertIn("perftest_mgmt_ub.c", cmake)
        self.assertIn("perftest_mgmt_ub.h", cmake)
        self.assertIn("KVAF_PERFTEST_MGMT_UB_SOURCE", cmake)
        self.assertNotIn("KVAF_PERFTEST_COMMUNICATION_SOURCE", cmake)
        self.assertIn('#include "perftest_mgmt_tcp.h"', communication)
        self.assertIn('#include "perftest_mgmt_ub.h"', communication)
        self.assertIn(
            "#include KVAF_PERFTEST_MGMT_UB_SOURCE", communication
        )
        self.assertIn("kvaf_ub_establish_connection", communication)
        self.assertIn("kvaf_ub_close_connection_checked", communication)
        self.assertIn("kvaf_perftest_control_fd", communication)
        self.assertIn("cfg->v_address", resources)
        self.assertIn("cfg->enable_va", resources)
        self.assertIn("g_kvaf_config.v_address = config->local_address", provider)
        self.assertIn("g_kvaf_config.enable_va = true", provider)
        for source in (communication, resources, provider):
            self.assertNotIn("ai_jettys", source)
            self.assertNotIn("dev_buf", source)
            self.assertNotIn("comm.sock_fd", source)

    def test_tcp_and_urma_management_addresses_are_disjoint(self):
        provider = self.read(
            "access_operator/a5/provider/urma_perftest_provider.c"
        )
        abi = self.read(
            "access_operator/a5/provider/include/kvaf_urma_provider_abi.h"
        )
        communication = self.read(
            "access_operator/a5/provider/source_matched_communication.c"
        )

        for field in (
            "control_server_address",
            "control_bind_address",
            "management_peer_address",
            "management_local_address",
            "management_jetty_id",
        ):
            self.assertIn(field, abi)
            self.assertIn(field, provider)
        self.assertIn(
            "KVAF_URMA_PROVIDER_EXPLICIT_TCP_MANAGEMENT", abi
        )
        self.assertIn("KVAF_URMA_PROVIDER_URMA_MANAGEMENT", abi)
        self.assertIn("config->reserved0 != 0U", provider)
        self.assertIn("config->reserved[0] != 0U", provider)
        self.assertIn("config->reserved[1] != 0U", provider)
        self.assertIn("kvaf_numeric_ip_family", provider)
        self.assertIn("AF_INET", provider)
        self.assertIn("AF_INET6", provider)
        self.assertIn(
            "KVAF_ADD_ARG(input->control_server_address)", provider
        )
        self.assertIn(
            "KVAF_ADD_ARG(input->control_bind_address)", provider
        )
        self.assertIn('KVAF_ADD_ARG("--mgmt_proto")', provider)
        self.assertNotIn('KVAF_ADD_ARG("--mgmt")', provider)
        self.assertIn('KVAF_ADD_ARG("ub")', provider)
        self.assertIn(
            "KVAF_ADD_ARG(input->management_local_address)", provider
        )
        self.assertIn(
            "KVAF_ADD_ARG(input->management_peer_address)", provider
        )
        parse_start = provider.index("static int kvaf_parse_config(")
        parse_end = provider.index(
            "static bool kvaf_ummu_checked_runtime_is_armed", parse_start
        )
        parse = provider[parse_start:parse_end]
        self.assertIn("use_urma_management", parse)
        self.assertIn("input->role == KVAF_URMA_PROVIDER_CLIENT", parse)
        self.assertIn("control_family == AF_INET6", parse)
        self.assertIn("errno = 0", communication)
        self.assertIn("saved_errno", communication)
        self.assertIn("kvaf_negative_errno_or(EIO)", provider)

    def test_ub_management_bootstrap_and_teardown_are_checked(self):
        provider = self.read(
            "access_operator/a5/provider/urma_perftest_provider.c"
        )
        communication = self.read(
            "access_operator/a5/provider/source_matched_communication.c"
        )

        for expected in (
            "g_kvaf_ub_server_recv_posts == 2U",
            '"KVAF_URMA_MGMT_READY role=server jetty_id=%u "',
            "kvaf_ub_peer_matches(expected_peer_eid)",
            "strcmp(ctx->dev->name, expected_device_name) != 0",
            "eid_idx != expected_eid_index",
            "kvaf_deadline_urma_poll_jfc",
            "kvaf_udma_source_retry_orphan_cleanup(ctx->urma_ctx)",
            "kvaf_ub_quiesce_management_jetty(ctx)",
            "kvaf_ub_quiesce_management_jfr(ctx)",
            "kvaf_ub_drain_management_jfc(ctx)",
            "kvaf_tracking_urma_post_jetty_send_wr",
            "kvaf_ub_account_management_completions",
            "kvaf_ub_send_tracker_close_is_quiesced(",
            "kvaf_ub_send_tracker_require_flush_barrier(",
            "#define KVAF_UB_CLOSE_DRAIN_BATCH 1",
            "URMA_CR_WR_FLUSH_ERR_DONE",
            "kvaf_udma_source_cleanup_is_unknown() != 0",
            "kvaf_source_ummu_cleanup_is_unknown() != 0",
            "kvaf_source_ummu_reset_thread_status()",
            '"KVAF_URMA_MGMT_CLOSE_UMMU stage=%s failure_stage=%u "',
            "KVAF_UB_DESTROY_STEP(KVAF_UB_CLOSE_UNIMPORT_TJETTY,",
            "KVAF_UB_DESTROY_STEP(KVAF_UB_CLOSE_DELETE_CONTEXT,",
            "KVAF_PERFTEST_MANAGEMENT_CLOSE_RETRYABLE",
            "KVAF_PERFTEST_MANAGEMENT_CLOSE_UNSAFE",
            '"KVAF_URMA_MGMT_CLOSE_ERROR stage=%s status=%d errno=%d "',
        ):
            self.assertIn(expected, communication)
        self.assertNotIn(
            "kvaf_source_cleanup_state_is_unknown()", communication
        )
        self.assertNotIn(
            "operation_status == EINTR || operation_errno == EINTR",
            communication,
        )
        self.assertNotIn("completion->flag.bs.flush_err_done", communication)
        self.assertIn("current != NULL && current != stop", communication)
        self.assertIn("if (current != stop)", communication)
        self.assertIn(
            "KVAF_UB_MANAGEMENT_SEND_WR_SCAN_LIMIT", communication
        )
        self.assertIn(
            "kvaf_ub_send_tracker_mark_post_unknown(", communication
        )
        jetty_create = communication.index(
            "ctx->pair.jetty = urma_create_jetty("
        )
        jetty_null_check = communication.index(
            "if (ctx->pair.jetty == NULL)", jetty_create
        )
        barrier_required = communication.index(
            "kvaf_ub_send_tracker_require_flush_barrier(",
            jetty_null_check,
        )
        self.assertLess(jetty_create, jetty_null_check)
        self.assertLess(jetty_null_check, barrier_required)
        actual_id_check = communication.index(
            "ctx->pair.jetty->jetty_id.id != cfg->dst_jetty_id",
            jetty_null_check,
        )
        self.assertLess(jetty_null_check, actual_id_check)
        self.assertLess(actual_id_check, barrier_required)
        self.assertIn(
            "g_kvaf_ub_server_jetty_id = "
            "ctx->pair.jetty->jetty_id.id",
            communication,
        )
        self.assertLess(
            communication.index("g_kvaf_ub_server_recv_posts == 2U"),
            communication.index(
                '"KVAF_URMA_MGMT_READY role=server jetty_id=%u "'
            ),
        )

        open_start = provider.index("int kvaf_urma_provider_open_v2(")
        close_start = provider.index(
            "static int kvaf_close_unlocked(", open_start
        )
        open_body = provider[open_start:close_start]
        self.assertLess(
            open_body.index("kvaf_udma_source_runtime_init()"),
            open_body.index("kvaf_perftest_establish_connection("),
        )
        close_body = provider[close_start:]
        self.assertLess(
            close_body.index("kvaf_perftest_close_connection("),
            close_body.index("kvaf_udma_source_runtime_uninit()"),
        )

    def test_ub_management_reports_the_exact_failed_stage(self):
        communication = self.read(
            "access_operator/a5/provider/source_matched_communication.c"
        )

        self.assertIn(
            '"KVAF_URMA_MGMT_ERROR stage=%s raw_errno=%d "',
            communication,
        )
        self.assertIn(
            '"fallback_errno=%d effective_errno=%d\\n"',
            communication,
        )
        for stage in (
            "validate_establish_arguments",
            "resolve_device_eid",
            "validate_device_eid",
            "create_context",
            "create_jfc",
            "allocate_send_token",
            "allocate_receive_token",
            "allocate_handshake_token",
            "register_send_segment",
            "register_receive_segment",
            "register_handshake_segment",
            "create_jfr",
            "create_jetty",
            "validate_server_jetty_id",
            "server_handshake",
            "client_parse_peer_eid",
            "client_import_peer_jetty",
            "client_post_handshake_send",
            "client_poll_handshake_cqe",
            "validate_peer_eid",
            "post_client_receive",
        ):
            self.assertIn(f'"{stage}"', communication)

        # The source-linked URMA core intentionally leaves ENOENT while it
        # suppresses dynamic-provider scans. Every independently checked UB
        # operation must start with a clean errno, while the error reporter
        # retains a deterministic fallback for APIs that do not set errno.
        for operation in (
            "ub_resolve_dev_and_eid_idx(",
            "urma_create_context(",
            "urma_create_jfc(",
            "urma_register_seg(",
            "urma_create_jfr(",
            "urma_create_jetty(",
            "ub_server_handshake(",
            "if (kvaf_ub_client_handshake_checked(",
            "ub_post_one_recv(",
        ):
            operation_index = communication.index(operation)
            self.assertIn(
                "errno = 0;",
                communication[max(0, operation_index - 120):operation_index],
            )
        helper_start = communication.index(
            "static int kvaf_ub_management_error("
        )
        helper_end = communication.index(
            "static int kvaf_ub_create_local_resources_checked(",
            helper_start,
        )
        helper = communication[helper_start:helper_end]
        self.assertLess(helper.index("(void)fflush(stderr);"), helper.index(
            "errno = concrete_errno;"
        ))

    def test_segment_failure_survives_public_wrapper_cleanup(self):
        header = self.read(
            "access_operator/a5/provider/include/kvaf_source_matched_udma.h"
        )
        segment = self.read(
            "access_operator/a5/provider/source_matched_udma_segment.c"
        )
        communication = self.read(
            "access_operator/a5/provider/source_matched_communication.c"
        )

        self.assertIn("struct kvaf_udma_segment_failure", header)
        self.assertIn("KVAF_UDMA_SEGMENT_FAILURE_UMMU_GRANT", header)
        self.assertIn("KVAF_UDMA_SEGMENT_FAILURE_KERNEL_REGISTER", header)
        self.assertIn("static _Thread_local struct kvaf_udma_segment_failure", segment)
        self.assertIn("kvaf_udma_source_get_segment_failure", segment)
        self.assertIn("rollback_status = ummu_ungrant(", segment)
        self.assertIn("KVAF_URMA_SEGMENT_ERROR management_stage=%s", communication)
        self.assertIn("provider_stage=%s operation_status=%d", communication)
        self.assertIn("ummu_failure_stage=%u", communication)
        self.assertIn("ummu_state_unknown=%u", communication)
        for stage in (
            "register_send_segment",
            "register_receive_segment",
            "register_handshake_segment",
        ):
            call = communication.index(f'"{stage}", errno, EIO')
            self.assertIn(
                "kvaf_ub_segment_registration_error(",
                communication[max(0, call - 120):call],
            )

    def test_retryable_grant_failure_retains_exact_segment_owner(self):
        segment = self.read(
            "access_operator/a5/provider/source_matched_udma_segment.c"
        )

        retry_start = segment.index(
            "static int kvaf_retry_orphan_segment_locked("
        )
        retry_end = segment.index(
            "\nint kvaf_udma_source_retry_orphan_segments(", retry_start
        )
        retry = segment[retry_start:retry_end]
        register_start = segment.index("urma_target_seg_t *udma_u_register_seg(")
        register_end = segment.index(
            "\n urma_status_t udma_u_unregister_seg(".lstrip(),
            register_start,
        )
        register = segment[register_start:register_end]

        self.assertIn("grant_retry_needed", segment)
        self.assertIn("grant_state_unknown", segment)
        self.assertIn("kvaf_source_ummu_get_thread_status", register)
        self.assertIn("no_grant_resource_acquired", register)
        self.assertIn("state->grant_retry_needed = true", register)
        self.assertIn("udma_u_grant_segment(state->segment)", retry)
        self.assertLess(
            retry.index("udma_u_grant_segment(state->segment)"),
            retry.index("ummu_ungrant("),
        )
        self.assertIn("kvaf_udma_source_mark_cleanup_unknown", segment)

    def test_ub_management_explicitly_owns_segment_tokens(self):
        communication = self.read(
            "access_operator/a5/provider/source_matched_communication.c"
        )

        self.assertIn("g_kvaf_ub_management_tokens", communication)
        self.assertIn(
            "seg_cfg.flag.bs.token_id_valid = URMA_TOKEN_ID_VALID",
            communication,
        )
        self.assertIn(
            "KVAF_URMA_TOKEN_ERROR management_stage=%s",
            communication,
        )
        close = communication[communication.index(
            "static int kvaf_ub_close_connection_checked(void)"
        ):]
        handshake_segment = close.index(
            "ctx->tseg_recv_hs, urma_unregister_seg"
        )
        handshake_token = close.index(
            "g_kvaf_ub_management_tokens[KVAF_UB_MANAGEMENT_HANDSHAKE]",
            handshake_segment,
        )
        receive_segment = close.index(
            "ctx->tseg_recv, urma_unregister_seg",
            handshake_token,
        )
        receive_token = close.index(
            "g_kvaf_ub_management_tokens[KVAF_UB_MANAGEMENT_RECEIVE]",
            receive_segment,
        )
        send_segment = close.index(
            "ctx->tseg_send, urma_unregister_seg",
            receive_token,
        )
        send_token = close.index(
            "g_kvaf_ub_management_tokens[KVAF_UB_MANAGEMENT_SEND]",
            send_segment,
        )
        self.assertLess(handshake_segment, handshake_token)
        self.assertLess(handshake_token, receive_segment)
        self.assertLess(receive_segment, receive_token)
        self.assertLess(receive_token, send_segment)
        self.assertLess(send_segment, send_token)

    def test_tcp_connect_retries_share_one_monotonic_deadline(self):
        communication = self.read(
            "access_operator/a5/provider/source_matched_communication.c"
        )
        connect = communication.index("static int kvaf_deadline_connect(")
        sleep = communication.index("static int kvaf_deadline_usleep(")
        include = communication.index(
            "#include KVAF_PERFTEST_MGMT_TCP_SOURCE"
        )
        self.assertLess(connect, sleep)
        self.assertLess(sleep, include)
        self.assertIn(
            "kvaf_deadline_remaining_ms() < 0", communication[connect:sleep]
        )
        self.assertIn(
            "remaining_microseconds", communication[sleep:include]
        )
        self.assertIn("#define usleep kvaf_deadline_usleep", communication)
        self.assertIn("#undef usleep", communication)

    def test_provider_config_layout_remains_fixed_and_semantically_pinned(self):
        abi = self.read(
            "access_operator/a5/provider/include/kvaf_urma_provider_abi.h"
        )
        self.assertIn(
            "sizeof(kvaf_urma_provider_config_v2) == 256", abi
        )
        self.assertIn("control_server_address) == 104", abi)
        self.assertIn("control_bind_address) == 168", abi)
        self.assertIn("management_peer_address) == 104", abi)
        self.assertIn("management_local_address) == 168", abi)
        self.assertIn("management_jetty_id) == 232", abi)

    def test_embedded_perftest_legacy_dependencies_are_private_and_closed(self):
        cmake = self.read("access_operator/a5/provider/CMakeLists.txt")
        resources = self.read(
            "access_operator/a5/provider/source_matched_perftest.c"
        )

        include = resources.index("#include KVAF_PERFTEST_RESOURCES_SOURCE")
        rearm_define = resources.index(
            "#define rearm_jfc kvaf_legacy_rearm_jfc_unsupported"
        )
        close_define = resources.index(
            "#define close_connection kvaf_legacy_close_connection"
        )
        rearm_undef = resources.index("#undef rearm_jfc", include)
        close_undef = resources.index("#undef close_connection", include)

        self.assertIn(
            "static int kvaf_legacy_rearm_jfc_unsupported(", resources
        )
        self.assertIn(
            "static void kvaf_legacy_close_connection(", resources
        )
        self.assertIn("kvaf_perftest_close_connection(cfg);", resources)
        for field in ("server_ip", "bind_ip", "mgmt_addr"):
            self.assertIn(f"free(cfg->{field});", resources)
            self.assertIn(f"cfg->{field} = NULL;", resources)
        self.assertLess(rearm_define, include)
        self.assertLess(close_define, include)
        self.assertGreater(rearm_undef, include)
        self.assertGreater(close_undef, include)
        self.assertNotIn('"${KVAF_PERFTEST_SOURCE_DIR}/urma_perftest.c"', cmake)
        self.assertNotIn('"${KVAF_PERFTEST_SOURCE_DIR}/perftest_mgmt.c"', cmake)
        self.assertIn('"-Wl,-z,defs"', cmake)

    def test_upstream_warning_exceptions_are_source_scoped(self):
        cmake = self.read("access_operator/a5/provider/CMakeLists.txt")
        resources = self.read(
            "access_operator/a5/provider/source_matched_perftest.c"
        )

        self.assertIn(
            '"${KVAF_URMA_COMMON_SOURCE_DIR}/ub_dstring.c"\n'
            "  APPEND PROPERTY COMPILE_OPTIONS\n"
            '  "-Wno-sign-compare"',
            cmake,
        )
        sign_block_start = cmake.index(
            'set_property(SOURCE\n'
            '  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_cmd.c"'
        )
        sign_block_end = cmake.index("\n)", sign_block_start)
        sign_block = cmake[sign_block_start:sign_block_end]
        for source in ("urma_cmd.c", "urma_log.c", "urma_perf.c"):
            self.assertIn(
                f'"${{KVAF_URMA_CORE_SOURCE_DIR}}/{source}"', sign_block
            )
        self.assertIn('"-Wno-sign-compare"', sign_block)

        unused_block_start = cmake.index(
            'set_property(SOURCE\n'
            '  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_cp_api.c"'
        )
        unused_block_end = cmake.index("\n)", unused_block_start)
        unused_block = cmake[unused_block_start:unused_block_end]
        for source in ("urma_cp_api.c", "urma_device.c", "urma_perf.c"):
            self.assertIn(
                f'"${{KVAF_URMA_CORE_SOURCE_DIR}}/{source}"', unused_block
            )
        self.assertIn('"-Wno-unused-parameter"', unused_block)

        # Never hide upstream warnings by weakening a complete KVAF target.
        for target in (
            "kvaf_source_urma_common",
            "kvaf_source_urma_core",
            "kvaf_urma_provider_common",
        ):
            compile_options = cmake.index(
                f"target_compile_options({target} PRIVATE"
            )
            compile_options_end = cmake.index(")", compile_options)
            target_flags = cmake[compile_options:compile_options_end]
            self.assertIn("-Wall -Wextra", target_flags)
            self.assertNotIn("-Wno-", target_flags)

        include = resources.index("#include KVAF_PERFTEST_RESOURCES_SOURCE")
        push = resources.rindex("#pragma GCC diagnostic push", 0, include)
        sign = resources.rindex(
            '#pragma GCC diagnostic ignored "-Wsign-compare"', 0, include
        )
        unused = resources.rindex(
            '#pragma GCC diagnostic ignored "-Wunused-parameter"',
            0,
            include,
        )
        pop = resources.index("#pragma GCC diagnostic pop", include)
        self.assertLess(push, sign)
        self.assertLess(sign, unused)
        self.assertLess(unused, include)
        self.assertLess(include, pop)

    def test_provider_lifecycle_rejects_foreign_global_state(self):
        main = self.read(
            "access_operator/a5/provider/source_matched_udma_main.c"
        )
        ops = self.read(
            "access_operator/a5/provider/source_matched_udma_ops.c"
        )
        self.assertIn("status = urma_init(&init_attr);", main)
        self.assertIn("status != URMA_SUCCESS", main)
        self.assertIn("status = urma_uninit();", main)
        self.assertIn("g_kvaf_udma_runtime_uninit_failed", main)
        self.assertIn("kvaf_source_ummu_checked_contract_ready()", main)
        self.assertIn("kvaf_source_ummu_pending_operation_count()", main)
        self.assertIn("kvaf_source_ummu_retained_mapping_count()", main)
        self.assertIn("kvaf_udma_source_provider_owns_device(device)", ops)
        self.assertLess(
            ops.index("kvaf_udma_source_provider_owns_device(device)"),
            ops.index("urma_cmd_create_context("),
        )

    def test_private_liburma_core_cannot_scan_system_providers(self):
        cmake = self.read("access_operator/a5/provider/CMakeLists.txt")
        wrapper = self.read(
            "access_operator/a5/provider/source_linked_urma_main.c"
        )
        builder = self.read("scripts/build_a5_direct_urma_provider.sh")
        preflight = self.read(
            "scripts/a5_direct_urma_runtime_preflight.sh"
        )

        self.assertIn("add_library(kvaf_source_urma_core OBJECT", cmake)
        self.assertIn("$<TARGET_OBJECTS:kvaf_source_urma_core>", cmake)
        provider_function = cmake[cmake.index(
            "function(kvaf_add_urma_provider"
        ):cmake.index("kvaf_add_urma_provider(\n  kvaf_urma_provider_host")]
        self.assertNotIn("KVAF_URMA_LIBRARY", provider_function)
        self.assertNotIn("KVAF_URMA_COMMON_LIBRARY", provider_function)
        self.assertIn("$<TARGET_OBJECTS:kvaf_source_urma_common>", provider_function)
        self.assertIn("#define SO_LINKED 1", wrapper)
        self.assertLess(
            wrapper.index("#define _GNU_SOURCE"),
            wrapper.index("#include <dirent.h>"),
        )
        self.assertIn(
            "#define opendir(path) kvaf_urma_restricted_opendir(path)",
            wrapper,
        )
        self.assertIn('strcmp(path, "/dev/uburma") == 0', wrapper)
        real_open = wrapper.index("return opendir(path);")
        macro = wrapper.index("#define opendir(path)")
        self.assertLess(real_open, macro)
        self.assertIn("errno = ENOENT;", wrapper[real_open:macro])
        self.assertIn(
            "kvaf_urma_core_has_dynamic_provider_scan_disabled", wrapper
        )
        self.assertIn(
            "urma_core_mode=official-umdk-source-linked-no-dynamic-providers",
            builder,
        )
        self.assertIn("urma_common_mode=official-umdk-source-built-internal", builder)
        self.assertIn("source_layout=official-umdk-libummu-paired-v1", builder)
        self.assertIn("umdk_libummu_pairing=verified", builder)
        self.assertIn("unexpectedly links external URMA/UMMU code", builder)
        self.assertIn("lib(hcomm|urma_perftest|urma_common|urma|urma-udma|ummu|udma)", preflight)

    def test_private_core_scan_stub_has_a_real_null_declaration(self):
        wrapper = self.read(
            "access_operator/a5/provider/source_linked_urma_main.c"
        )
        null_header = wrapper.index("#include <stddef.h>")
        scan_stub = wrapper.index(
            "static DIR *kvaf_urma_restricted_opendir"
        )
        self.assertLess(null_header, scan_stub)

    def test_jetty_delete_is_single_shot_and_event_ack_resumable(self):
        source = self.read(
            "access_operator/a5/provider/source_matched_udma_jetty.c"
        )
        command = source.index("status = urma_ioctl_delete_jetty(")
        issued = source.index(
            "state->kernel_delete_pending = true;", command
        )
        wait = source.index(
            "kvaf_udma_drain_async_events_bounded(", issued
        )
        local_cleanup = source.index(
            "kvaf_cleanup_local_jetty_checked(state)", wait
        )
        free_local = source.index("kvaf_free_local_jetty(state)", local_cleanup)
        self.assertLess(command, issued)
        self.assertLess(issued, wait)
        self.assertLess(local_cleanup, free_local)
        self.assertIn("bool kernel_delete_pending;", source)
        self.assertIn("delete_async_events_reported", source)
        self.assertIn("kvaf_udma_destroy_decide(", source)
        self.assertIn(
            "decision == KVAF_UDMA_DESTROY_ISSUE_IOCTL", source
        )
        self.assertNotIn("urma_cmd_delete_jetty(", source)
        self.assertIn("URMA_NO_SHARE_JFR", source)
        self.assertIn("sq->trans_mode == URMA_TM_RC", source)
        delete = source[source.index(
            "urma_status_t kvaf_udma_delete_jetty_checked("
        ):]
        self.assertIn(
            "state->kernel_created || state->kernel_delete_pending",
            delete,
        )
        self.assertNotIn("udma_u_free_jetty(public_jetty)", delete)

    def test_direct_jetty_creation_does_not_use_broken_legacy_union_path(self):
        source = self.read(
            "access_operator/a5/provider/source_matched_udma_jetty.c"
        )
        create_start = source.index(
            "urma_jetty_t *kvaf_udma_create_jetty_checked("
        )
        create_end = source.index(
            "urma_status_t kvaf_udma_delete_jetty_checked", create_start
        )
        create = source[create_start:create_end]
        self.assertNotIn("kvaf_legacy_udma_u_create_jetty", create)
        self.assertIn("cfg->jfr_cfg->trans_mode != URMA_TM_RM", create)
        self.assertIn("jetty->jfr = shared_jfr", create)
        self.assertIn("cfg->shared.jfr", create)
        self.assertIn("kvaf_create_direct_jetty_command", create)
        self.assertIn("kvaf_delete_kernel_jetty_nonblocking", create)

    def test_segment_and_token_retain_state_on_ummu_failure(self):
        segment = self.read(
            "access_operator/a5/provider/source_matched_udma_segment.c"
        )
        token = self.read(
            "access_operator/a5/provider/source_matched_udma_tid.c"
        )
        self.assertIn("bool kernel_registered;", segment)
        self.assertIn("bool ummu_granted;", segment)
        self.assertIn("uint32_t ummu_tid;", segment)
        self.assertIn("ummu_ungrant(state->ummu_tid", segment)
        self.assertIn("if (ummu_ungrant(", segment)
        self.assertLess(
            segment.index("if (ummu_ungrant("),
            segment.index("free(segment);", segment.index("if (ummu_ungrant(")),
        )
        self.assertIn("bool kernel_allocated;", token)
        self.assertIn("bool ummu_allocated;", token)
        self.assertIn("status = ummu_free_tid(token->tid);", token)
        self.assertIn("if (status != 0)", token)
        self.assertIn("errno = kvaf_errno_from_ummu_status(status);", token)

    def test_failed_create_orphans_are_context_owned_and_retryable(self):
        segment = self.read(
            "access_operator/a5/provider/source_matched_udma_segment.c"
        )
        token = self.read(
            "access_operator/a5/provider/source_matched_udma_tid.c"
        )
        jetty = self.read(
            "access_operator/a5/provider/source_matched_udma_jetty.c"
        )
        jfr = self.read(
            "access_operator/a5/provider/source_matched_udma_jfr.c"
        )
        ops = self.read(
            "access_operator/a5/provider/source_matched_udma_ops.c"
        )

        for source in (segment, token, jetty, jfr):
            self.assertIn("urma_context_t *owner_context;", source)
            self.assertIn("bool orphan;", source)
            self.assertIn("state->owner_context != context", source)

        self.assertIn("kvaf_udma_source_orphan_segment_count", segment)
        self.assertIn("kvaf_udma_source_orphan_token_count", token)
        self.assertIn("kvaf_udma_source_orphan_jetty_count", jetty)
        self.assertIn("kvaf_udma_source_orphan_jfr_count", jfr)
        self.assertIn("kvaf_udma_source_retry_orphan_jetties", jetty)
        self.assertIn("kvaf_udma_source_retry_orphan_jfrs", jfr)
        self.assertIn("kvaf_udma_source_retry_orphan_cleanup", ops)
        drain = ops.index("kvaf_udma_source_retry_orphan_cleanup(")
        live_gate = ops.index(
            "kvaf_udma_source_context_has_live_resources(", drain
        )
        self.assertLess(drain, live_gate)

    def test_context_delete_gates_only_exact_context_sidecars(self):
        header = self.read(
            "access_operator/a5/provider/include/kvaf_source_matched_udma.h"
        )
        ops = self.read(
            "access_operator/a5/provider/source_matched_udma_ops.c"
        )
        delete_start = ops.index(
            "static urma_status_t kvaf_udma_delete_context_checked("
        )
        delete_end = ops.index(
            "int kvaf_udma_source_context_has_live_resources(",
            delete_start,
        )
        delete_context = ops[delete_start:delete_end]

        self.assertIn(
            "kvaf_udma_source_context_has_live_resources(public_context)",
            delete_context,
        )
        for kind, relative in (
            ("segment", "source_matched_udma_segment.c"),
            ("token", "source_matched_udma_tid.c"),
            ("jetty", "source_matched_udma_jetty.c"),
            ("jfr", "source_matched_udma_jfr.c"),
            ("jfc", "source_matched_udma_jfc.c"),
        ):
            symbol = f"kvaf_udma_source_live_{kind}_count_for_context"
            source = self.read(f"access_operator/a5/provider/{relative}")
            self.assertIn(symbol, header)
            self.assertIn(symbol, source)
            helper_start = source.index(symbol)
            helper = source[helper_start:helper_start + 900]
            self.assertIn("state->owner_context == context", helper)
            self.assertIn("pthread_mutex_lock(", helper)
            self.assertIn("pthread_mutex_unlock(", helper)

        aggregate_start = ops.index(
            "int kvaf_udma_source_context_has_live_resources("
        )
        aggregate_end = ops.index(
            "int kvaf_udma_source_retry_orphan_cleanup(", aggregate_start
        )
        aggregate = ops[aggregate_start:aggregate_end]
        for kind in ("segment", "token", "jetty", "jfr", "jfc"):
            self.assertIn(
                f"kvaf_udma_source_live_{kind}_count_for_context(context)",
                aggregate,
            )
            self.assertNotIn(
                f"kvaf_udma_source_live_{kind}_count()", delete_context
            )

    def test_context_create_is_failure_atomic_without_legacy_wrapper(self):
        ops = self.read(
            "access_operator/a5/provider/source_matched_udma_ops.c"
        )
        provider = self.read(
            "access_operator/a5/provider/urma_perftest_provider.c"
        )
        create_start = ops.index(
            "static urma_context_t *kvaf_udma_create_context_checked("
        )
        create_end = ops.index(
            "static urma_status_t kvaf_udma_delete_context_checked(",
            create_start,
        )
        create = ops[create_start:create_end]

        self.assertNotIn("kvaf_legacy_udma_u_create_context(", create)
        self.assertIn("state->orphan = true;", create)
        self.assertIn("state->kernel_context_created = true;", create)
        self.assertIn("state->db_allocated = true;", create)
        self.assertIn("state->table_layout_initialized = true;", ops)
        self.assertIn("state->jfr_table_lock_initialized = true;", ops)
        self.assertIn("state->jetty_table_lock_initialized = true;", ops)
        self.assertIn("kvaf_retry_orphan_context_locked(state)", create)
        self.assertIn("kvaf_udma_source_retry_orphan_contexts()", provider)
        self.assertLess(
            provider.index("kvaf_udma_source_retry_orphan_contexts()"),
            provider.index("kvaf_udma_source_live_context_count()"),
        )

    def test_context_orphan_retry_advances_only_completed_phases(self):
        ops = self.read(
            "access_operator/a5/provider/source_matched_udma_ops.c"
        )
        retry_start = ops.index(
            "static int kvaf_retry_orphan_context_locked("
        )
        retry_end = ops.index(
            "int kvaf_udma_source_retry_orphan_contexts(void)",
            retry_start,
        )
        retry = ops[retry_start:retry_end]
        self.assertLess(
            retry.index("if (state->kernel_context_created)"),
            retry.index("if (state->db_allocated)"),
        )
        self.assertIn("kvaf_udma_checked_free_doorbell(", retry)
        self.assertIn("state->db_allocated = false;", retry)
        self.assertIn("urma_cmd_delete_context(&context->urma_ctx)", retry)
        self.assertIn("state->kernel_context_created = false;", retry)
        self.assertNotIn("free(state)", retry)

    def test_local_cleanup_ebusy_is_unsafe_not_watchdog_retry(self):
        cleanup = self.read(
            "access_operator/a5/provider/include/"
            "kvaf_source_matched_udma_cleanup.h"
        )
        ops = self.read(
            "access_operator/a5/provider/source_matched_udma_ops.c"
        )

        pthread_start = cleanup.index(
            "static inline int kvaf_udma_pthread_cleanup_failed("
        )
        pthread_end = cleanup.index(
            "static inline int kvaf_udma_classify_cleanup_failure(",
            pthread_start,
        )
        pthread_body = cleanup[pthread_start:pthread_end]
        self.assertIn("kvaf_udma_source_mark_cleanup_unknown();", pthread_body)
        self.assertNotIn(
            "!kvaf_udma_cleanup_errno_is_retryable", pthread_body
        )

        classify_start = pthread_end
        classify_end = cleanup.index(
            "static inline int kvaf_udma_snapshot_event_counters_checked(",
            classify_start,
        )
        classify_body = cleanup[classify_start:classify_end]
        self.assertIn(
            "kvaf_udma_source_mark_cleanup_unknown();", classify_body
        )
        self.assertIn("kvaf_udma_cleanup_invariant_failed()", ops)

    def test_direct_destroy_local_invariants_fail_closed(self):
        for kind, public_type, orphan_name in (
            ("jetty", "urma_jetty_t", "jetties"),
            ("jfr", "urma_jfr_t", "jfrs"),
            ("jfc", "urma_jfc_t", "jfcs"),
        ):
            source = self.read(
                f"access_operator/a5/provider/source_matched_udma_{kind}.c"
            )
            lock_name = f"kvaf_lock_{kind}_registry_checked"
            unlock_name = f"kvaf_unlock_{kind}_registry_checked"

            lock_start = source.index(f"static int {lock_name}(void)")
            unlock_start = source.index(
                f"static int {unlock_name}(void)", lock_start
            )
            kernel_start = source.index(
                f"static int kvaf_delete_kernel_{kind}_nonblocking("
            )
            kernel_end = source.index("\n}\n", kernel_start) + 3
            kernel = source[kernel_start:kernel_end]
            delete_start = source.index(
                f"urma_status_t kvaf_udma_delete_{kind}_checked("
                f"{public_type} *public_{kind})"
            )
            delete_end = source.index("\n}\n", delete_start) + 3
            delete = source[delete_start:delete_end]
            create_start = source.index(
                f"kvaf_udma_create_{kind}_checked("
            )
            create = source[create_start:delete_start]
            orphan_start = source.index(
                f"int kvaf_udma_source_retry_orphan_{orphan_name}("
            )
            orphan_end = source.index("\n}\n", orphan_start) + 3
            orphan = source[orphan_start:orphan_end]

            self.assertIn(
                "kvaf_udma_pthread_cleanup_failed(status)",
                source[lock_start:kernel_start],
            )
            self.assertIn("state->owner_context == NULL", kernel)
            self.assertIn(
                f"{kind}->urma_ctx != state->owner_context", kernel
            )
            self.assertIn(f"{kind}->urma_ctx->dev_fd < 0", kernel)
            self.assertIn(
                "return kvaf_udma_cleanup_invariant_failed();", kernel
            )
            self.assertIn(
                "if (!kvaf_udma_destroy_ioctl_retry_is_proven(", kernel
            )

            for cleanup_path in (delete, orphan):
                self.assertIn(f"{lock_name}()", cleanup_path)
                self.assertIn(f"{unlock_name}()", cleanup_path)
                self.assertIn(
                    "kvaf_udma_cleanup_invariant_failed()", cleanup_path
                )
                self.assertNotIn(
                    f"pthread_mutex_lock(&g_kvaf_{kind}_lock)",
                    cleanup_path,
                )
                self.assertNotIn(
                    f"pthread_mutex_unlock(&g_kvaf_{kind}_lock)",
                    cleanup_path,
                )

            create_unlock = create.index(
                f"if ({unlock_name}() != 0)"
            )
            create_unlock_failure = create[create_unlock:]
            self.assertIn(f"{lock_name}()", create)
            self.assertIn("state->orphan = true;", create_unlock_failure)
            self.assertIn(
                f"++g_kvaf_{kind}_orphan_count;", create_unlock_failure
            )
            self.assertIn("return NULL;", create_unlock_failure)
            self.assertNotIn(
                f"pthread_mutex_lock(&g_kvaf_{kind}_lock)", create
            )
            self.assertNotIn(
                f"pthread_mutex_unlock(&g_kvaf_{kind}_lock)", create
            )

    def test_userspace_mappings_have_checked_retry_phases(self):
        cleanup = self.read(
            "access_operator/a5/provider/include/"
            "kvaf_source_matched_udma_cleanup.h"
        )
        ops = self.read(
            "access_operator/a5/provider/source_matched_udma_ops.c"
        )
        jetty = self.read(
            "access_operator/a5/provider/source_matched_udma_jetty.c"
        )
        jfc = self.read(
            "access_operator/a5/provider/source_matched_udma_jfc.c"
        )
        jfr = self.read(
            "access_operator/a5/provider/source_matched_udma_jfr.c"
        )

        doorbell = cleanup[cleanup.index(
            "static inline int kvaf_udma_checked_free_doorbell("
        ):cleanup.index(
            "static inline void kvaf_udma_unlink_hugepage("
        )]
        self.assertLess(doorbell.index("munmap("),
                        doorbell.index("doorbell->addr = NULL;"))
        self.assertIn("return -1;", doorbell)
        self.assertIn("kvaf_udma_checked_free_queue_buffer", cleanup)
        self.assertIn("kvaf_udma_checked_free_sw_db", cleanup)
        self.assertIn("kvaf_udma_checked_alloc_queue_buffer", cleanup)
        self.assertIn("kvaf_udma_checked_alloc_sw_db", cleanup)
        self.assertIn("kvaf_udma_checked_alloc_hugepage_queue", cleanup)
        self.assertIn("queue->ctx->hugepage_enable", cleanup)
        self.assertIn("required_bytes > UINT32_MAX", cleanup)
        self.assertIn("rounded_bytes > UINT32_MAX / 2U", cleanup)
        self.assertIn("base_block_size - 1U", cleanup)
        self.assertIn("page_size - 1U", cleanup)
        self.assertLess(
            cleanup.index("queue->qbuf = mmap("),
            cleanup.index("madvise(queue->qbuf"),
        )

        self.assertNotIn("udma_u_free_db(public_context", ops)
        self.assertNotIn("udma_u_destroy_hugepage(context)", ops)
        self.assertIn("kvaf_udma_checked_free_doorbell(", ops)
        self.assertIn("kvaf_udma_context_has_db_pages(context)", ops)

        local_jetty = jetty[jetty.index(
            "static int kvaf_cleanup_local_jetty_checked("
        ):jetty.index("static void kvaf_free_local_jetty(")]
        self.assertIn("kvaf_udma_checked_free_doorbell(", local_jetty)
        self.assertIn("kvaf_udma_checked_free_queue_buffer(", local_jetty)
        self.assertNotIn("udma_u_free_db(", local_jetty)
        self.assertNotIn("udma_u_delete_sq(", local_jetty)
        direct_create = jetty[jetty.index(
            "static int kvaf_create_direct_sq_checked("
        ):jetty.index("static int kvaf_cleanup_local_jetty_checked(")]
        self.assertIn("kvaf_udma_checked_alloc_queue_buffer(", direct_create)
        self.assertNotIn("udma_u_create_sq(", direct_create)
        self.assertIn("UINT32_MAX / config->depth", direct_create)

        self.assertNotIn(
            "kvaf_legacy_udma_u_create_jfc(context, cfg)", jfc
        )
        self.assertNotIn(
            "kvaf_legacy_udma_u_delete_jfc(public_jfc)", jfc
        )
        self.assertIn("bool kernel_created;", jfc)
        self.assertIn("bool sw_db_allocated;", jfc)
        self.assertIn("bool cq_created;", jfc)
        self.assertIn("kvaf_udma_source_retry_orphan_jfcs", jfc)
        self.assertIn("kvaf_udma_checked_free_sw_db(", jfc)
        self.assertIn("kvaf_udma_checked_free_queue_buffer(", jfc)
        self.assertIn("kvaf_udma_checked_alloc_sw_db(", jfc)
        self.assertIn("kvaf_udma_checked_alloc_queue_buffer(", jfc)
        self.assertNotIn("udma_u_create_cq(&jfc->cq", jfc)
        self.assertNotIn("udma_u_alloc_sw_db(\n        udma_context", jfc)

        self.assertIn("bool index_buffer_mapped;", jfr)
        self.assertIn("bool index_bitmap_allocated;", jfr)
        self.assertIn("bool table_inserted;", jfr)
        self.assertIn("bool table_lock_unknown;", jfr)
        self.assertIn("kvaf_udma_checked_free_sw_db(", jfr)
        self.assertIn("kvaf_udma_checked_free_queue_buffer(", jfr)
        self.assertIn("kvaf_insert_jfr_table_checked", jfr)
        self.assertIn("kvaf_remove_jfr_table_checked", jfr)
        self.assertNotIn("udma_u_free_idx_que(&jfr->idx_que)", jfr)

    def test_table_unlock_ambiguity_is_process_lifetime_unsafe(self):
        header = self.read(
            "access_operator/a5/provider/include/"
            "kvaf_source_matched_udma.h"
        )
        main = self.read(
            "access_operator/a5/provider/source_matched_udma_main.c"
        )
        provider = self.read(
            "access_operator/a5/provider/urma_perftest_provider.c"
        )
        jetty = self.read(
            "access_operator/a5/provider/source_matched_udma_jetty.c"
        )
        jfr = self.read(
            "access_operator/a5/provider/source_matched_udma_jfr.c"
        )

        self.assertIn("kvaf_udma_source_mark_cleanup_unknown", header)
        self.assertIn("kvaf_udma_source_cleanup_is_unknown", header)
        self.assertIn("static uint32_t g_kvaf_udma_cleanup_unknown;", main)
        self.assertIn("__atomic_store_n", main)
        self.assertIn("__atomic_load_n", main)

        runtime_init = main[main.index(
            "int kvaf_udma_source_runtime_init(void)"
        ):main.index("int kvaf_udma_source_runtime_uninit(void)")]
        runtime_uninit = main[main.index(
            "int kvaf_udma_source_runtime_uninit(void)"
        ):main.index("int kvaf_udma_source_runtime_is_owned(void)")]
        self.assertIn("kvaf_udma_source_cleanup_is_unknown()", runtime_init)
        self.assertIn("kvaf_udma_source_cleanup_is_unknown()", runtime_uninit)
        self.assertLess(
            runtime_uninit.index("kvaf_udma_source_cleanup_is_unknown()"),
            runtime_uninit.index("kvaf_udma_source_retry_orphan_contexts()"),
        )

        for source, helper in (
            (jetty, "kvaf_unlock_jetty_table_checked"),
            (jfr, "kvaf_unlock_jfr_table_checked"),
        ):
            helper_start = source.index(f"static int {helper}(")
            helper_end = source.index("\n}\n", helper_start)
            helper_body = source[helper_start:helper_end]
            self.assertIn("state->table_lock_unknown = true;", helper_body)
            self.assertIn(
                "kvaf_udma_source_mark_cleanup_unknown();", helper_body
            )

        no_live = provider[provider.index(
            "static bool kvaf_udma_source_has_no_live_resources(void)"
        ):provider.index("int kvaf_urma_provider_open_v2(")]
        close_retry = provider[provider.index(
            "if (teardown_status =="
        ):provider.index(
            "if (teardown_status != KVAF_PERFTEST_CHECKED_OK)"
        )]
        self.assertIn("kvaf_udma_source_cleanup_is_unknown()", no_live)
        self.assertIn("kvaf_source_cleanup_state_is_unknown()", close_retry)
        self.assertIn("KVAF_URMA_PROVIDER_CLOSE_UNSAFE", close_retry)
        self.assertIn("KVAF_URMA_PROVIDER_CLOSE_RETRYABLE", close_retry)

    def test_kernel_destroy_event_waits_never_use_blocking_umdk_helpers(self):
        cleanup = self.read(
            "access_operator/a5/provider/include/"
            "kvaf_source_matched_udma_cleanup.h"
        )
        self.assertIn("poll(&descriptor, 1, 0)", cleanup)
        self.assertIn("KVAF_UDMA_ASYNC_DRAIN_BUDGET", cleanup)
        self.assertIn("urma_get_async_event(context, &event)", cleanup)
        self.assertIn("urma_ack_async_event(&event)", cleanup)

        for kind in ("jetty", "jfr", "jfc"):
            source = self.read(
                f"access_operator/a5/provider/source_matched_udma_{kind}.c"
            )
            self.assertIn("bool kernel_delete_pending;", source)
            self.assertIn(f"urma_ioctl_delete_{kind}(", source)
            self.assertNotIn(f"urma_cmd_delete_{kind}(", source)
            self.assertIn("kvaf_udma_destroy_decide(", source)
            self.assertIn(
                "decision == KVAF_UDMA_DESTROY_ISSUE_IOCTL", source
            )
            self.assertIn(
                "kvaf_udma_destroy_ioctl_retry_is_proven(", source
            )
            self.assertIn(
                "kvaf_udma_drain_async_events_bounded(", source
            )
            self.assertIn("bool event_condition_initialized;", source)
            self.assertIn("bool event_mutex_initialized;", source)
            self.assertIn("kvaf_udma_destroy_event_sync_checked(", source)
            self.assertIn(
                "kvaf_udma_snapshot_event_counters_checked(", source
            )

        jetty = self.read(
            "access_operator/a5/provider/source_matched_udma_jetty.c"
        )
        command = jetty.index("status = kvaf_create_direct_jetty_command(")
        created_guard = jetty.index("if (state->kernel_created)", command)
        failure_guard = jetty.index("if (status != 0)", command)
        self.assertLess(created_guard, failure_guard)
        self.assertIn(
            "state->event_condition_initialized = true;",
            jetty[created_guard:failure_guard],
        )

    def test_checked_jfr_and_shared_jetty_are_the_only_direct_paths(self):
        cmake = self.read("access_operator/a5/provider/CMakeLists.txt")
        builder = self.read("scripts/build_a5_direct_urma_provider.sh")
        ops = self.read(
            "access_operator/a5/provider/source_matched_udma_ops.c"
        )
        main = self.read(
            "access_operator/a5/provider/source_matched_udma_main.c"
        )
        provider = self.read(
            "access_operator/a5/provider/urma_perftest_provider.c"
        )
        perftest = self.read(
            "access_operator/a5/provider/source_matched_perftest.c"
        )
        jetty = self.read(
            "access_operator/a5/provider/source_matched_udma_jetty.c"
        )

        common_sources = cmake[cmake.index(
            "set(KVAF_UDMA_COMMON_SOURCES"
        ):cmake.index(
            "function(kvaf_add_source_udma_object"
        )]
        self.assertNotIn("udma_u_jfr.c", common_sources)
        self.assertIn("source_matched_udma_jfr.c", cmake)
        self.assertIn("KVAF_UDMA_JFR_SOURCE", cmake)

        self.assertIn(
            "g_udma_ops.create_jfr = kvaf_udma_create_jfr_checked", ops
        )
        self.assertIn(
            "g_udma_ops.delete_jfr = kvaf_udma_delete_jfr_checked", ops
        )
        for unsafe_op in (
            "delete_jfr_batch", "alloc_jfr", "active_jfr",
            "deactive_jfr", "free_jfr",
        ):
            self.assertIn(f"g_udma_ops.{unsafe_op} = NULL;", ops)
        self.assertIn("kvaf_udma_source_retry_orphan_jfrs", ops)
        for lifecycle in (main, provider):
            self.assertIn("kvaf_udma_source_live_jfr_count()", lifecycle)
        self.assertIn(
            "kvaf_udma_source_live_jfr_count_for_context(context)", ops
        )
        for symbol in (
            "kvaf_udma_source_live_jfr_count",
            "kvaf_udma_source_orphan_jfr_count",
            "kvaf_udma_source_retry_orphan_jfrs",
        ):
            self.assertIn(symbol, builder)

        self.assertIn("if (cfg->share_jfr)", perftest)
        self.assertIn("jetty_cfg.shared.jfc", perftest)
        self.assertIn("jetty_cfg.jfr_cfg->jfc", perftest)
        self.assertIn("URMA_SHARE_JFR", jetty)
        self.assertIn("URMA_NO_SHARE_JFR", jetty)
        self.assertIn("kvaf_insert_jetty_table_checked", jetty)
        self.assertIn("kvaf_remove_jetty_table_checked", jetty)
        free_local = jetty[jetty.index(
            "static void kvaf_free_local_jetty("
        ):jetty.index(
            "int kvaf_udma_source_retry_orphan_jetties("
        )]
        self.assertIn("URMA_NO_SHARE_JFR", free_local)
        self.assertNotIn("free(state->jetty->base.jetty_cfg.jfr_cfg)",
                         free_local)

    def test_orphan_retry_never_selects_successfully_returned_objects(self):
        for relative in (
            "access_operator/a5/provider/source_matched_udma_segment.c",
            "access_operator/a5/provider/source_matched_udma_tid.c",
            "access_operator/a5/provider/source_matched_udma_jetty.c",
            "access_operator/a5/provider/source_matched_udma_jfr.c",
        ):
            source = self.read(relative)
            retry_loop = source.index("state->owner_context != context")
            successful_handoff = source.index("state->orphan = false;")
            self.assertIn("!state->orphan", source[retry_loop - 120:retry_loop])
            self.assertGreater(successful_handoff, 0)

    def test_system_ummu_build_stays_partial_and_fail_closed(self):
        cmake = self.read("access_operator/a5/provider/CMakeLists.txt")
        script = self.read("scripts/build_a5_direct_urma_provider.sh")
        if '"${KVAF_UMMU_LIBRARY}"' in cmake:
            self.assertIn(
                "udma_teardown_contract=source-owned-partial-retry-v1",
                script,
            )
            self.assertNotIn(
                "udma_teardown_contract=checked-retryable-v1", script
            )
        self.assertNotIn(
            "udma_teardown_contract=unverified-system-provider", script
        )


if __name__ == "__main__":
    unittest.main()
