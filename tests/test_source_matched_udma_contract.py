import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROVIDER = ROOT / "access_operator" / "a5" / "provider"


class SourceMatchedUdmaContractTests(unittest.TestCase):
    def read(self, relative: str) -> str:
        return (ROOT / relative).read_text(encoding="utf-8")

    def test_public_abi_is_single_source_with_private_preinclude(self):
        cmake = self.read("access_operator/a5/provider/CMakeLists.txt")
        shim = self.read(
            "access_operator/a5/provider/include/udma_public_abi_preinclude.h"
        )
        self.assertIn(
            'set(KVAF_URMA_PUBLIC_ROOT\n  "${KVAF_URMA_PERFTEST_ROOT}/urma/code")',
            cmake,
        )
        self.assertNotIn(
            "${KVAF_UBENGINE_ROOT}/kernelspace/urma/code", cmake
        )
        self.assertGreaterEqual(
            cmake.count("udma_public_abi_preinclude.h"), 3
        )
        self.assertIn("#define udma_u_jetty_queue", shim)
        self.assertIn('#include "urma_provider.h"', shim)
        self.assertIn("#undef udma_u_jetty_queue", shim)
        self.assertIn("kvaf_urma_public_abi_probe", cmake)

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
        runner = self.read("scripts/run_a5_direct_urma_h2d_smoke.sh")

        self.assertIn("add_library(kvaf_source_urma_core OBJECT", cmake)
        self.assertIn("$<TARGET_OBJECTS:kvaf_source_urma_core>", cmake)
        provider_function = cmake[cmake.index(
            "function(kvaf_add_urma_provider"
        ):cmake.index("kvaf_add_urma_provider(\n  kvaf_urma_provider_host")]
        self.assertNotIn('"${KVAF_URMA_LIBRARY}"', provider_function)
        self.assertIn('"${KVAF_URMA_COMMON_LIBRARY}"', provider_function)
        self.assertIn("#define SO_LINKED 1", wrapper)
        self.assertLess(
            wrapper.index("#define _GNU_SOURCE"),
            wrapper.index("#include <dirent.h>"),
        )
        self.assertIn(
            "#define opendir(path) kvaf_urma_dynamic_provider_scan_forbidden(path)",
            wrapper,
        )
        self.assertIn(
            "kvaf_urma_core_has_dynamic_provider_scan_disabled", wrapper
        )
        self.assertIn(
            "urma_core_mode=source-linked-no-dynamic-providers", builder
        )
        self.assertIn("urma_library_linkage=abi-reference-only", builder)
        self.assertIn("external liburma/UMMU/UDMA code", builder)
        self.assertIn(
            "urma_core_mode=source-linked-no-dynamic-providers", runner
        )
        self.assertIn("lib(urma|ummu|urma-udma)\\.so", runner)

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
        self.assertIn("jetty->jfr = NULL", create)
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
        ops = self.read(
            "access_operator/a5/provider/source_matched_udma_ops.c"
        )

        for source in (segment, token, jetty):
            self.assertIn("urma_context_t *owner_context;", source)
            self.assertIn("bool orphan;", source)
            self.assertIn("state->owner_context != context", source)

        self.assertIn("kvaf_udma_source_orphan_segment_count", segment)
        self.assertIn("kvaf_udma_source_orphan_token_count", token)
        self.assertIn("kvaf_udma_source_orphan_jetty_count", jetty)
        self.assertIn("kvaf_udma_source_retry_orphan_jetties", jetty)
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
        self.assertIn("state->node_table_initialized = true;", create)
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

    def test_orphan_retry_never_selects_successfully_returned_objects(self):
        for relative in (
            "access_operator/a5/provider/source_matched_udma_segment.c",
            "access_operator/a5/provider/source_matched_udma_tid.c",
            "access_operator/a5/provider/source_matched_udma_jetty.c",
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
