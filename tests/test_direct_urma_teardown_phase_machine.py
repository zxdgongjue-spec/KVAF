#!/usr/bin/env python3

from __future__ import annotations

import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
PROVIDER = ROOT / "access_operator" / "a5" / "provider"


class DirectUrmaTeardownPhaseMachineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.header = (
            PROVIDER / "include" / "kvaf_source_matched_perftest.h"
        ).read_text(encoding="utf-8")
        cls.perftest_source = (
            PROVIDER / "source_matched_perftest.c"
        ).read_text(encoding="utf-8")
        cls.provider_source = (
            PROVIDER / "urma_perftest_provider.c"
        ).read_text(encoding="utf-8")

    def test_public_state_header_is_valid_c_and_statuses_are_distinct(
        self,
    ) -> None:
        compiler = shutil.which("cc") or shutil.which("clang")
        if compiler is None:
            self.skipTest("no C compiler is installed")
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            (temp / "perftest_parameters.h").write_text(
                "typedef struct perftest_config perftest_config_t;\n",
                encoding="utf-8",
            )
            (temp / "perftest_resources.h").write_text(
                textwrap.dedent(
                    """
                    #include "perftest_parameters.h"
                    typedef struct perftest_context perftest_context_t;
                    """
                ),
                encoding="utf-8",
            )
            unit = temp / "state_header_test.c"
            unit.write_text(
                textwrap.dedent(
                    """
                    #include "kvaf_source_matched_perftest.h"
                    _Static_assert(
                        KVAF_PERFTEST_CHECKED_DESTROY_RETRYABLE == -3,
                        "retryable status changed");
                    _Static_assert(
                        KVAF_PERFTEST_CHECKED_DESTROY_UNSAFE == -4,
                        "unsafe status must remain distinct");
                    _Static_assert(
                        KVAF_PERFTEST_TEARDOWN_COMPLETE <
                            KVAF_PERFTEST_TEARDOWN_UNSAFE,
                        "complete and unsafe phases overlap");
                    int main(void) {
                      struct kvaf_perftest_teardown_state state = {0};
                      return (int)state.object_handle;
                    }
                    """
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    compiler,
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-fsyntax-only",
                    "-I",
                    str(temp),
                    "-I",
                    str(PROVIDER / "include"),
                    str(unit),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_peer_barriers_are_separate_resumable_phases(self) -> None:
        required = (
            "KVAF_PERFTEST_TEARDOWN_DISCONNECT_LOCAL",
            "KVAF_PERFTEST_TEARDOWN_DISCONNECT_PEER",
            "KVAF_PERFTEST_TEARDOWN_UNIMPORT_SEG_LOCAL",
            "KVAF_PERFTEST_TEARDOWN_UNIMPORT_SEG_PEER",
        )
        for marker in required:
            self.assertIn(marker, self.header)
            self.assertIn(f"case {marker}:", self.perftest_source)

        disconnect_local = self.perftest_source.index(
            "case KVAF_PERFTEST_TEARDOWN_DISCONNECT_LOCAL:"
        )
        disconnect_peer = self.perftest_source.index(
            "case KVAF_PERFTEST_TEARDOWN_DISCONNECT_PEER:",
            disconnect_local,
        )
        local_body = self.perftest_source[
            disconnect_local:disconnect_peer
        ]
        self.assertIn("kvaf_disconnect_jettys_checked", local_body)
        self.assertIn("kvaf_sync_phase_status_detailed", local_body)
        self.assertIn("kvaf_teardown_retryable", local_body)

    def test_every_destroy_handle_is_nulled_only_after_success(self) -> None:
        ordered_pairs = (
            (
                "state->last_status = urma_unimport_seg",
                "ctx->import_tseg[i] = NULL;",
            ),
            (
                "urma_unimport_jetty(ctx->import_tjetty[i]);",
                "ctx->import_tjetty[i] = NULL;",
            ),
            (
                "state->last_status = urma_unregister_seg",
                "ctx->local_tseg[i] = NULL;",
            ),
            (
                "state->last_status = urma_free_token_id",
                "ctx->token_id[i] = NULL;",
            ),
            (
                "state->last_status = urma_delete_jetty",
                "ctx->jetty[i] = NULL;",
            ),
            (
                "state->last_status = urma_delete_jfr",
                "ctx->jfr[i] = NULL;",
            ),
            (
                "status = urma_delete_context(ctx->urma_ctx);",
                "ctx->urma_ctx = NULL;",
            ),
        )
        for operation, clear in ordered_pairs:
            operation_offset = self.perftest_source.index(operation)
            failure_return = self.perftest_source.index(
                "return", operation_offset
            )
            clear_offset = self.perftest_source.index(
                clear, operation_offset
            )
            self.assertLess(failure_return, clear_offset)

    def test_retryable_close_poisoning_requires_unknown_ummu_state(self) -> None:
        retryable = self.provider_source.index(
            "KVAF_PERFTEST_CHECKED_DESTROY_RETRYABLE"
        )
        unsafe = self.provider_source.index(
            "teardown_status != KVAF_PERFTEST_CHECKED_OK", retryable
        )
        retryable_body = self.provider_source[retryable:unsafe]
        self.assertIn(
            "return KVAF_URMA_PROVIDER_CLOSE_RETRYABLE;",
            retryable_body,
        )
        unknown = retryable_body.index(
            "if (kvaf_source_ummu_state_is_unknown())"
        )
        poison = retryable_body.index(
            "g_kvaf_cleanup_poisoned = true", unknown
        )
        unsafe_return = retryable_body.index(
            "return KVAF_URMA_PROVIDER_CLOSE_UNSAFE;", poison
        )
        retry_return = retryable_body.index(
            "return KVAF_URMA_PROVIDER_CLOSE_RETRYABLE;", unsafe_return
        )
        self.assertLess(unknown, poison)
        self.assertLess(poison, unsafe_return)
        self.assertLess(unsafe_return, retry_return)
        self.assertNotIn("g_kvaf_session_id = 0", retryable_body)

    def test_each_fallible_local_phase_retries_before_advance(self) -> None:
        cases = (
            (
                "KVAF_PERFTEST_TEARDOWN_LOCAL_SEGMENTS",
                "kvaf_unregister_segments_checked",
                "KVAF_PERFTEST_TEARDOWN_TOKEN_IDS",
            ),
            (
                "KVAF_PERFTEST_TEARDOWN_TOKEN_IDS",
                "kvaf_free_tokens_checked",
                "KVAF_PERFTEST_TEARDOWN_LOCAL_BUFFER_VECTOR",
            ),
            (
                "KVAF_PERFTEST_TEARDOWN_JETTYS",
                "kvaf_delete_jettys_checked",
                "KVAF_PERFTEST_TEARDOWN_JFRS",
            ),
            (
                "KVAF_PERFTEST_TEARDOWN_JFRS",
                "kvaf_delete_jfrs_checked",
                "KVAF_PERFTEST_TEARDOWN_RECV_JFCS",
            ),
            (
                "KVAF_PERFTEST_TEARDOWN_RECV_JFCS",
                "kvaf_delete_recv_jfcs_checked",
                "KVAF_PERFTEST_TEARDOWN_SEND_JFCS",
            ),
            (
                "KVAF_PERFTEST_TEARDOWN_SEND_JFCS",
                "kvaf_delete_send_jfcs_checked",
                "KVAF_PERFTEST_TEARDOWN_CONTEXT",
            ),
        )
        for phase, operation, next_phase in cases:
            start = self.perftest_source.index(f"case {phase}:")
            end = self.perftest_source.index(
                f"case {next_phase}:", start
            )
            body = self.perftest_source[start:end]
            operation_offset = body.index(operation)
            retry_offset = body.index(
                "return kvaf_teardown_retryable", operation_offset
            )
            advance_offset = body.index(
                "kvaf_teardown_advance", retry_offset
            )
            self.assertLess(operation_offset, retry_offset)
            self.assertLess(retry_offset, advance_offset)

        context = self.perftest_source.index(
            "case KVAF_PERFTEST_TEARDOWN_CONTEXT:"
        )
        complete = self.perftest_source.index(
            "case KVAF_PERFTEST_TEARDOWN_COMPLETE:", context
        )
        context_body = self.perftest_source[context:complete]
        self.assertLess(
            context_body.index("return kvaf_teardown_retryable"),
            context_body.index("KVAF_PERFTEST_TEARDOWN_COMPLETE"),
        )

    def test_provider_passes_persistent_state_back_on_every_retry(self) -> None:
        self.assertIn(
            "static struct kvaf_perftest_teardown_state "
            "g_kvaf_teardown_state;",
            self.provider_source,
        )
        destroy_call = self.provider_source.index(
            "kvaf_perftest_destroy_duplex_ctx_checked("
        )
        destroy_end = self.provider_source.index(");", destroy_call)
        self.assertIn(
            "&g_kvaf_teardown_state",
            self.provider_source[destroy_call:destroy_end],
        )
        state_reset = self.provider_source.index(
            "kvaf_perftest_teardown_state_init(&g_kvaf_teardown_state)"
        )
        self.assertLess(state_reset, destroy_call)

    def test_runtime_provider_ownership_precedes_queue_export(self) -> None:
        ownership = self.provider_source.index(
            "kvaf_udma_source_provider_owns_context"
        )
        queue_extract = self.provider_source.index(
            "kvaf_udma_extract_direct_queue_info", ownership
        )
        self.assertLess(ownership, queue_extract)
        self.assertIn(
            "result->reserved[1] = KVAF_UDMA_SOURCE_CAPABILITY_MAGIC",
            self.provider_source,
        )
        self.assertIn(
            "KVAF_UDMA_TEARDOWN_RUNTIME_CAPABILITY",
            self.provider_source,
        )
        self.assertIn(
            "kvaf_udma_source_public_abi_matches(",
            self.provider_source,
        )
        self.assertIn(
            "result->reserved[3] =",
            self.provider_source,
        )

    def test_runtime_lifecycle_wraps_every_urma_resource(self) -> None:
        abi_gate = self.provider_source.index(
            "kvaf_udma_source_public_abi_matches("
        )
        runtime_init = self.provider_source.index(
            "ret = kvaf_udma_source_runtime_init()"
        )
        create = self.provider_source.index(
            "ret = kvaf_perftest_create_duplex_ctx_checked("
        )
        live_zero = self.provider_source.index(
            "!kvaf_udma_source_has_no_live_resources()",
            create,
        )
        runtime_uninit = self.provider_source.index(
            "kvaf_udma_source_runtime_uninit()", live_zero
        )
        tombstone = self.provider_source.index(
            "g_kvaf_last_closed_session_id = g_kvaf_session_id",
            runtime_uninit,
        )
        self.assertLess(abi_gate, runtime_init)
        self.assertLess(runtime_init, create)
        self.assertLess(create, live_zero)
        self.assertLess(live_zero, runtime_uninit)
        self.assertLess(runtime_uninit, tombstone)
        self.assertIn(
            "kvaf_perftest_confirm_runtime_ready(",
            self.provider_source,
        )

    def test_legacy_perftest_cannot_uninit_owned_runtime(self) -> None:
        include_offset = self.perftest_source.index(
            "#include KVAF_PERFTEST_RESOURCES_SOURCE"
        )
        before = self.perftest_source[:include_offset]
        after = self.perftest_source[include_offset:]
        self.assertIn(
            "#define urma_uninit kvaf_legacy_urma_uninit_noop",
            before,
        )
        self.assertIn("#undef urma_uninit", after)
        self.assertIn(
            "#define urma_get_device_by_name "
            "kvaf_owned_urma_get_device_by_name",
            before,
        )

    def test_owned_device_lookup_uses_the_pinned_public_urma_api(self) -> None:
        api = (
            ROOT.parent
            / "urma_perftest"
            / "urma"
            / "code"
            / "include"
            / "urma_api.h"
        ).read_text(encoding="utf-8")
        implementation = (
            ROOT.parent
            / "urma_perftest"
            / "urma"
            / "code"
            / "lib"
            / "urma"
            / "core"
            / "urma_main.c"
        ).read_text(encoding="utf-8")
        api_include = self.perftest_source.index('#include "urma_api.h"')
        owned_lookup = self.perftest_source.index(
            "static urma_device_t *kvaf_owned_urma_get_device_by_name"
        )

        self.assertLess(api_include, owned_lookup)
        self.assertIn(
            "urma_device_t *urma_get_device_by_name(char *dev_name);", api
        )
        self.assertIn(
            "urma_device_t *urma_get_device_by_name(char *dev_name)",
            implementation,
        )
        self.assertNotIn(
            "extern urma_device_t *urma_get_device_by_name",
            self.perftest_source,
        )

    def test_synchronized_creation_failure_keeps_handles_for_close(self) -> None:
        create_start = self.perftest_source.index(
            "int kvaf_perftest_create_duplex_ctx_checked("
        )
        create_end = self.perftest_source.index(
            "int kvaf_perftest_confirm_provider_ready(", create_start
        )
        create = self.perftest_source[create_start:create_end]
        self.assertIn("kvaf_init_device_checked", create)
        self.assertIn("kvaf_create_duplex_queues_public", create)
        self.assertIn("kvaf_register_memory_checked", create)
        self.assertNotIn("status = init_device(", create)
        self.assertNotIn("status = register_mem(", create)

        queue_start = self.perftest_source.index(
            "static int kvaf_create_duplex_queues_public("
        )
        queue_end = self.perftest_source.index(
            "static int kvaf_register_memory_checked(", queue_start
        )
        queue = self.perftest_source[queue_start:queue_end]
        self.assertNotIn("destroy_jfc", queue)
        self.assertNotIn("urma_delete_", queue)

        self.assertIn(
            "if (ret != KVAF_PERFTEST_CHECKED_CREATE_CLEANUP_REQUIRED) {\n"
            "            g_kvaf_cleanup_poisoned = true;",
            self.provider_source,
        )

    def test_close_requires_all_private_live_counts_to_reach_zero(
        self,
    ) -> None:
        helper_start = self.provider_source.index(
            "static bool kvaf_udma_source_has_no_live_resources"
        )
        close_start = self.provider_source.index(
            "static int kvaf_close_unlocked(uint64_t session_id)\n{"
        )
        helper = self.provider_source[helper_start:close_start]
        for resource in ("segment", "token", "jetty", "jfc", "context"):
            self.assertIn(
                f"kvaf_udma_source_live_{resource}_count() == 0U",
                helper,
            )
        for counter in (
            "kvaf_source_ummu_live_tid_count() == 0U",
            "kvaf_source_ummu_pending_operation_count() == 0U",
            "kvaf_source_ummu_retained_mapping_count() == 0U",
        ):
            self.assertIn(counter, helper)
        tombstone = self.provider_source.index(
            "g_kvaf_last_closed_session_id = g_kvaf_session_id",
            close_start,
        )
        zero_check = self.provider_source.index(
            "!kvaf_udma_source_has_no_live_resources()", close_start
        )
        self.assertLess(zero_check, tombstone)

    def test_lost_close_result_is_idempotent_only_in_same_instance(
        self,
    ) -> None:
        close_start = self.provider_source.index(
            "static int kvaf_close_unlocked(uint64_t session_id)\n{"
        )
        close_end = self.provider_source.index(
            "int kvaf_urma_provider_close_v2", close_start
        )
        close = self.provider_source[close_start:close_end]
        self.assertIn("g_kvaf_last_closed_session_id == session_id", close)
        self.assertIn("? KVAF_URMA_PROVIDER_CLOSE_OK", close)
        self.assertIn(": KVAF_URMA_PROVIDER_CLOSE_UNSAFE", close)
        self.assertIn("g_kvaf_session_id != session_id", close)

    def test_orphan_rollback_is_drained_before_public_tokens(self) -> None:
        orphan_phase = self.perftest_source.index(
            "case KVAF_PERFTEST_TEARDOWN_ORPHAN_RESOURCES:"
        )
        orphan_drain = self.perftest_source.index(
            "kvaf_udma_source_retry_orphan_cleanup(", orphan_phase
        )
        public_segments = self.perftest_source.index(
            "case KVAF_PERFTEST_TEARDOWN_LOCAL_SEGMENTS:", orphan_drain
        )
        public_tokens = self.perftest_source.index(
            "case KVAF_PERFTEST_TEARDOWN_TOKEN_IDS:", public_segments
        )
        self.assertLess(orphan_phase, orphan_drain)
        self.assertLess(orphan_drain, public_segments)
        self.assertLess(public_segments, public_tokens)
        retry = self.perftest_source.index(
            "return kvaf_teardown_retryable(state, status);", orphan_drain
        )
        advance = self.perftest_source.index(
            "KVAF_PERFTEST_TEARDOWN_LOCAL_SEGMENTS", orphan_drain
        )
        self.assertLess(retry, advance)


if __name__ == "__main__":
    unittest.main()
