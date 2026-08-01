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

    def test_official_perftest_uses_tcp_management_and_external_va(self):
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
        self.assertNotIn("KVAF_PERFTEST_COMMUNICATION_SOURCE", cmake)
        self.assertIn('#include "perftest_mgmt_tcp.h"', communication)
        self.assertIn("kvaf_perftest_control_fd", communication)
        self.assertIn("cfg->v_address", resources)
        self.assertIn("cfg->enable_va", resources)
        self.assertIn("g_kvaf_config.v_address = config->local_address", provider)
        self.assertIn("g_kvaf_config.enable_va = true", provider)
        for source in (communication, resources, provider):
            self.assertNotIn("ai_jettys", source)
            self.assertNotIn("dev_buf", source)
            self.assertNotIn("comm.sock_fd", source)

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

    def test_jetty_delete_is_retryable_for_direct_rm_nonshared_path(self):
        source = self.read(
            "access_operator/a5/provider/source_matched_udma_jetty.c"
        )
        command = source.index("status = urma_cmd_delete_jetty(public_jetty);")
        restore = source.index(
            "public_jetty->jetty_cfg.jfr_cfg = original_jfr_cfg;",
            command,
        )
        local_cleanup = source.index(
            "kvaf_cleanup_local_jetty_checked(state)", restore
        )
        free_local = source.index("kvaf_free_local_jetty(state)", local_cleanup)
        self.assertLess(command, restore)
        self.assertLess(restore, local_cleanup)
        self.assertLess(local_cleanup, free_local)
        self.assertIn("command_jfr_cfg", source)
        self.assertIn("URMA_NO_SHARE_JFR", source)
        self.assertIn("sq->trans_mode == URMA_TM_RC", source)
        delete = source[source.index(
            "urma_status_t kvaf_udma_delete_jetty_checked("
        ):]
        self.assertIn("if (state->kernel_created)", delete)
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
        self.assertIn("kvaf_delete_kernel_jetty_preserve_cfg", create)

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
        self.assertIn("if (ummu_free_tid(token->tid) != 0)", token)

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
            "kvaf_udma_source_live_segment_count()", drain
        )
        self.assertLess(drain, live_gate)

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
            self.assertEqual(
                source.count("kvaf_udma_source_mark_cleanup_unknown();"), 1
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
        for lifecycle in (ops, main, provider):
            self.assertIn("kvaf_udma_source_live_jfr_count()", lifecycle)
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
