from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVIDER = ROOT / "access_operator" / "a5" / "provider"
ABI_HEADER = PROVIDER / "include" / "kvaf_urma_provider_abi.h"
PROVIDER_SOURCE = PROVIDER / "urma_perftest_provider.c"
EXPORT_MAP = PROVIDER / "provider_exports.map"
CONTRACT_PROBE = PROVIDER / "provider_aicpu_contract_probe.c"


def function_body(source: str, name: str) -> str:
    match = re.search(
        rf"\b(?:static\s+)?(?:bool|int|void|uint64_t)\b[^\n]*"
        rf"\b{re.escape(name)}\s*\([^)]*\)\s*\{{",
        source,
        flags=re.MULTILINE,
    )
    if match is None:
        raise AssertionError(f"could not find definition of {name}")
    start = source.index("{", match.start())
    depth = 0
    for offset in range(start, len(source)):
        character = source[offset]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[start : offset + 1]
    raise AssertionError(f"unterminated definition of {name}")


class AicpuEpochOwnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.abi = ABI_HEADER.read_text(encoding="utf-8")
        cls.provider = PROVIDER_SOURCE.read_text(encoding="utf-8")
        cls.exports = EXPORT_MAP.read_text(encoding="utf-8")
        cls.contract_probe = CONTRACT_PROBE.read_text(encoding="utf-8")

    def test_epoch_owner_abi_is_versioned_fixed_and_cache_separated(
        self,
    ) -> None:
        compiler = shutil.which("c++") or shutil.which("clang++")
        if compiler is None:
            self.skipTest("no C++ compiler is installed")
        source = textwrap.dedent(
            """
            #include <cstddef>
            #include <cstdint>
            #include "kvaf_urma_provider_abi.h"

            static_assert(KVAF_URMA_EPOCH_OWNER_MAGIC == 0x4b45504fU);
            static_assert(KVAF_URMA_EPOCH_OWNER_ABI_VERSION == 1U);
            static_assert(
                KVAF_URMA_EPOCH_OWNER_SHUTDOWN_SEQUENCE == 1ULL);
            static_assert(KVAF_URMA_EPOCH_OWNER_COMMAND_NONE == 0U);
            static_assert(
                KVAF_URMA_EPOCH_OWNER_COMMAND_SHUTDOWN_QUIESCED == 1U);
            static_assert(KVAF_URMA_EPOCH_OWNER_STATE_EMPTY == 0U);
            static_assert(KVAF_URMA_EPOCH_OWNER_STATE_OPENING == 1U);
            static_assert(KVAF_URMA_EPOCH_OWNER_STATE_READY == 2U);
            static_assert(KVAF_URMA_EPOCH_OWNER_STATE_CLOSING == 3U);
            static_assert(
                KVAF_URMA_EPOCH_OWNER_STATE_CLOSE_RETRYING == 4U);
            static_assert(KVAF_URMA_EPOCH_OWNER_STATE_CLOSED == 5U);
            static_assert(
                KVAF_URMA_EPOCH_OWNER_STATE_OPEN_FAILED_SAFE == 6U);
            static_assert(
                KVAF_URMA_EPOCH_OWNER_STATE_UNSAFE_WAITING == 7U);
            static_assert(sizeof(kvaf_urma_epoch_owner_command_v1) == 64);
            static_assert(alignof(kvaf_urma_epoch_owner_command_v1) == 64);
            static_assert(sizeof(kvaf_urma_epoch_owner_status_v1) == 64);
            static_assert(alignof(kvaf_urma_epoch_owner_status_v1) == 64);
            static_assert(
                sizeof(kvaf_urma_aicpu_epoch_owner_args_v1) == 384);
            static_assert(
                alignof(kvaf_urma_aicpu_epoch_owner_args_v1) == 64);
            static_assert(offsetof(
                kvaf_urma_epoch_owner_command_v1, opcode) == 12);
            static_assert(offsetof(
                kvaf_urma_epoch_owner_command_v1, epoch) == 16);
            static_assert(offsetof(
                kvaf_urma_epoch_owner_command_v1, final_sq_head) == 40);
            static_assert(offsetof(
                kvaf_urma_epoch_owner_status_v1, state) == 12);
            static_assert(offsetof(
                kvaf_urma_aicpu_epoch_owner_args_v1, epoch) == 320);
            static_assert(offsetof(
                kvaf_urma_aicpu_epoch_owner_args_v1,
                command_address) == 328);
            static_assert(offsetof(
                kvaf_urma_aicpu_epoch_owner_args_v1,
                status_address) == 336);
            static_assert(offsetof(
                kvaf_urma_aicpu_epoch_owner_args_v1,
                open_result_address) == 344);
            static_assert(offsetof(
                kvaf_urma_aicpu_epoch_owner_args_v1, magic) == 368);
            static_assert(offsetof(
                kvaf_urma_aicpu_epoch_owner_args_v1, abi_version) == 372);
            static_assert(offsetof(
                kvaf_urma_aicpu_epoch_owner_args_v1, struct_size) == 376);
            static_assert(offsetof(
                kvaf_urma_aicpu_epoch_owner_args_v1, flags) == 380);

            int main() {
              auto* entry = &aicpu_kvaf_urma_epoch_owner_v1;
              return entry == nullptr;
            }
            """
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            unit = Path(temporary_dir) / "epoch_owner_abi.cpp"
            unit.write_text(source, encoding="utf-8")
            result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-fsyntax-only",
                    "-I",
                    str(PROVIDER / "include"),
                    str(unit),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_epoch_owner_entry_is_declared_defined_and_exported(self) -> None:
        symbol = "aicpu_kvaf_urma_epoch_owner_v1"
        self.assertIn(symbol, self.abi)
        self.assertIn(symbol, self.provider)
        self.assertIn(symbol, self.exports)
        self.assertIn(f'load_entry(\n        library, "{symbol}")', self.contract_probe)
        self.assertIn("epoch_owner_entry(NULL)", self.contract_probe)

    def test_epoch_owner_header_compiles_as_c11(self) -> None:
        compiler = shutil.which("cc") or shutil.which("clang")
        if compiler is None:
            self.skipTest("no C compiler is installed")
        source = textwrap.dedent(
            """
            #include "kvaf_urma_provider_abi.h"

            int main(void) {
              int (*entry)(void *) = &aicpu_kvaf_urma_epoch_owner_v1;
              struct kvaf_urma_epoch_owner_command_v1 command = {0};
              struct kvaf_urma_epoch_owner_status_v1 status = {0};
              struct kvaf_urma_aicpu_epoch_owner_args_v1 args = {0};
              return entry == 0 || command.opcode != 0 ||
                  status.state != 0 || args.magic != 0;
            }
            """
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            unit = Path(temporary_dir) / "epoch_owner_abi.c"
            unit.write_text(source, encoding="utf-8")
            result = subprocess.run(
                [
                    compiler,
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-fsyntax-only",
                    "-I",
                    str(PROVIDER / "include"),
                    str(unit),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_one_invocation_performs_open_wait_and_close(self) -> None:
        body = function_body(
            self.provider, "aicpu_kvaf_urma_epoch_owner_v1"
        )
        open_call = body.index("kvaf_urma_provider_open_v2(")
        ready = body.index("KVAF_URMA_EPOCH_OWNER_STATE_READY", open_call)
        command_poll = body.index("__ATOMIC_ACQUIRE", ready)
        shutdown = body.index(
            "KVAF_URMA_EPOCH_OWNER_COMMAND_SHUTDOWN_QUIESCED",
            command_poll,
        )
        close = body.index(
            "kvaf_epoch_owner_close_until_complete", shutdown
        )
        self.assertLess(open_call, ready)
        self.assertLess(ready, command_poll)
        self.assertLess(command_poll, shutdown)
        self.assertLess(shutdown, close)

    def test_shutdown_validates_epoch_session_sequence_and_four_counters(
        self,
    ) -> None:
        body = function_body(
            self.provider, "aicpu_kvaf_urma_epoch_owner_v1"
        )
        shutdown = body.index(
            "KVAF_URMA_EPOCH_OWNER_COMMAND_SHUTDOWN_QUIESCED"
        )
        close = body.index("kvaf_epoch_owner_close_until_complete", shutdown)
        guarded = body[shutdown:close]
        for required in (
            "command->epoch == args->epoch",
            "command->session_id == args->config.session_id",
            "KVAF_URMA_EPOCH_OWNER_SHUTDOWN_SEQUENCE",
            "command->final_sq_head == command->final_sq_tail",
            "command->final_sq_tail == command->final_cq_tail",
            "command->final_cq_tail == command->final_wqe_count",
            "kvaf_quiesce_aiv_queue",
            "KVAF_URMA_EPOCH_OWNER_RESULT_QUEUE_NOT_QUIESCED",
        ):
            self.assertIn(required, guarded)

    def test_lease_expiry_never_closes_or_returns_without_queue_proof(
        self,
    ) -> None:
        body = function_body(
            self.provider, "aicpu_kvaf_urma_epoch_owner_v1"
        )
        ready = body.index("KVAF_URMA_EPOCH_OWNER_STATE_READY")
        wait_loop = body.index("for (;;)", ready)
        lease = body.index("lease_nanoseconds != 0", wait_loop)
        sleep = body.index("kvaf_owner_probe_sleep", lease)
        lease_path = body[lease:sleep]
        self.assertIn(
            "KVAF_URMA_EPOCH_OWNER_STATE_UNSAFE_WAITING", lease_path
        )
        self.assertIn(
            "KVAF_URMA_EPOCH_OWNER_RESULT_LEASE_EXPIRED", lease_path
        )
        self.assertNotIn("kvaf_urma_provider_close_v2", lease_path)
        self.assertNotIn("kvaf_epoch_owner_close_until_complete", lease_path)
        self.assertNotRegex(lease_path, r"\breturn\b")

    def test_checked_close_retries_in_the_same_invocation(self) -> None:
        body = function_body(
            self.provider, "kvaf_epoch_owner_close_until_complete"
        )
        self.assertIn("for (;;)", body)
        self.assertIn("kvaf_urma_provider_close_v2(", body)
        self.assertIn("KVAF_URMA_PROVIDER_CLOSE_RETRYABLE", body)
        self.assertIn("KVAF_URMA_EPOCH_OWNER_STATE_CLOSE_RETRYING", body)
        self.assertIn("KVAF_URMA_EPOCH_OWNER_RESULT_CLOSE_RETRY_LIMIT", body)
        self.assertIn("kvaf_owner_probe_sleep", body)
        self.assertIn("kvaf_epoch_owner_hold_unsafe", body)
        close_ok = body.index("KVAF_URMA_PROVIDER_CLOSE_OK")
        safe_return = body.index("return 0;", close_ok)
        retryable = body.index("KVAF_URMA_PROVIDER_CLOSE_RETRYABLE")
        self.assertLess(safe_return, retryable)
        self.assertNotIn("return 0;", body[retryable:])

    def test_epoch_claim_excludes_public_and_legacy_mutators(self) -> None:
        self.assertIn("static _Thread_local bool", self.provider)
        self.assertIn("g_kvaf_epoch_owner_authorized", self.provider)
        public_open = function_body(
            self.provider, "kvaf_urma_provider_open_v2"
        )
        public_close = function_body(
            self.provider, "kvaf_urma_provider_close_v2"
        )
        legacy_close = function_body(
            self.provider, "aicpu_kvaf_urma_close_v2"
        )
        for body in (public_open, public_close, legacy_close):
            self.assertIn("g_kvaf_epoch_claim_active", body)
            self.assertIn("kvaf_epoch_owner_is_authorized_unlocked", body)
        self.assertIn("KVAF_URMA_PROVIDER_ALREADY_OPEN", public_open)
        self.assertIn("KVAF_URMA_PROVIDER_CLOSE_UNSAFE", public_close)

    def test_epoch_owner_claim_wraps_open_and_checked_close(self) -> None:
        owner = function_body(
            self.provider, "aicpu_kvaf_urma_epoch_owner_v1"
        )
        claim = owner.index("kvaf_epoch_owner_acquire_claim")
        open_call = owner.index("kvaf_urma_provider_open_v2(")
        self.assertLess(claim, open_call)
        self.assertIn("KVAF_EPOCH_CLAIM_CONFLICT", owner[claim:open_call])
        self.assertIn(
            "KVAF_URMA_EPOCH_OWNER_STATE_OPEN_FAILED_SAFE",
            owner[claim:open_call],
        )

        close = function_body(
            self.provider, "kvaf_epoch_owner_close_until_complete"
        )
        close_ok = close.index("KVAF_URMA_PROVIDER_CLOSE_OK")
        release = close.index("kvaf_epoch_owner_release_claim", close_ok)
        terminal = close.index("kvaf_epoch_owner_publish", release)
        self.assertLess(close_ok, release)
        self.assertLess(release, terminal)

    def test_safe_failed_open_proof_is_bound_to_launch_session(self) -> None:
        helper = function_body(
            self.provider,
            "kvaf_epoch_owner_make_safe_failed_open_result",
        )
        self.assertIn("kvaf_init_result", helper)
        self.assertIn(
            "result->session_id = args->config.session_id", helper
        )

        owner = function_body(
            self.provider, "aicpu_kvaf_urma_epoch_owner_v1"
        )
        conflict = owner.index("KVAF_EPOCH_CLAIM_CONFLICT")
        opening = owner.index(
            "KVAF_URMA_EPOCH_OWNER_STATE_OPENING", conflict
        )
        conflict_path = owner[conflict:opening]
        self.assertIn(
            "kvaf_epoch_owner_make_safe_failed_open_result",
            conflict_path,
        )
        self.assertIn(
            "KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID", conflict_path
        )
        self.assertIn(
            "KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE", conflict_path
        )

        failure = owner.index("if (open_status != 0)")
        success_validation = owner.index(
            "\n    {\n        const bool result_valid", failure
        )
        failure_path = owner[failure:success_validation]
        release = failure_path.index("kvaf_epoch_owner_release_claim")
        bind = failure_path.index(
            "open_result->session_id = args->config.session_id", release
        )
        terminal = failure_path.index(
            "KVAF_URMA_EPOCH_OWNER_STATE_OPEN_FAILED_SAFE", bind
        )
        self.assertLess(release, bind)
        self.assertLess(bind, terminal)

    def test_clock_failure_before_claim_has_complete_safe_proof(self) -> None:
        owner = function_body(
            self.provider, "aicpu_kvaf_urma_epoch_owner_v1"
        )
        clock = owner.index("CLOCK_MONOTONIC")
        claim = owner.index("kvaf_epoch_owner_acquire_claim")
        clock_path = owner[clock:claim]
        self.assertIn(
            "kvaf_epoch_owner_make_safe_failed_open_result", clock_path
        )
        self.assertIn(
            "KVAF_URMA_EPOCH_OWNER_OPEN_RESULT_VALID", clock_path
        )
        self.assertIn(
            "KVAF_URMA_EPOCH_OWNER_TEARDOWN_COMPLETE", clock_path
        )

    def test_epoch_owner_reads_provider_globals_only_under_snapshot_lock(
        self,
    ) -> None:
        owner = function_body(
            self.provider, "aicpu_kvaf_urma_epoch_owner_v1"
        )
        self.assertIn("kvaf_epoch_owner_snapshot_local_state", owner)
        for global_name in (
            "g_kvaf_config_initialized",
            "g_kvaf_connection_open",
            "g_kvaf_context_open",
            "g_kvaf_runtime_acquired",
            "g_kvaf_cleanup_poisoned",
            "g_kvaf_session_id",
        ):
            self.assertNotIn(global_name, owner)

        snapshot = function_body(
            self.provider, "kvaf_epoch_owner_snapshot_local_state"
        )
        lock = snapshot.index("pthread_mutex_lock")
        first_global = min(
            snapshot.index("g_kvaf_epoch_claim_epoch"),
            snapshot.index("g_kvaf_session_id"),
        )
        unlock = snapshot.index("pthread_mutex_unlock", first_global)
        self.assertLess(lock, first_global)
        self.assertLess(first_global, unlock)

    def test_unsafe_terminal_holds_the_process_local_owner(self) -> None:
        body = function_body(self.provider, "kvaf_epoch_owner_hold_unsafe")
        self.assertIn("KVAF_URMA_EPOCH_OWNER_STATE_UNSAFE_WAITING", body)
        self.assertIn("KVAF_URMA_EPOCH_OWNER_OWNERSHIP_UNSAFE", body)
        self.assertIn("open_result_valid", body)
        self.assertIn("open_result_flag", body)
        self.assertIn("for (;;)", body)
        self.assertNotRegex(body, r"\breturn\b")

    def test_open_failure_only_retries_explicit_same_session_cleanup(
        self,
    ) -> None:
        body = function_body(
            self.provider, "aicpu_kvaf_urma_epoch_owner_v1"
        )
        failure = body.index("if (open_status != 0)")
        success_validation = body.index(
            "\n    {\n        const bool result_valid", failure
        )
        failure_path = body[failure:success_validation]
        for required in (
            "failure_result_contradicts_ownership",
            "cleanup_flags != 0 && !result_session_matches",
            "cleanup_retryable && !cleanup_uncertain",
            "cleanup_retryable || local_state_present",
            "result_valid",
        ):
            self.assertIn(required, failure_path)
        self.assertNotIn(
            "g_kvaf_session_id == args->config.session_id",
            failure_path,
        )

    def test_epoch_args_are_device_only_aligned_and_non_overlapping(
        self,
    ) -> None:
        body = function_body(
            self.provider, "kvaf_epoch_owner_args_are_valid"
        )
        for required in (
            "KVAF_URMA_PROVIDER_EXPECT_DEVICE_PRIVATE_ABI",
            "KVAF_URMA_PROVIDER_EXPECT_HOST_PRIVATE_ABI",
            "KVAF_URMA_EPOCH_OWNER_CACHE_LINE_BYTES",
            "kvaf_epoch_owner_ranges_overlap",
            "KVAF_URMA_EPOCH_OWNER_MAX_LEASE_MILLISECONDS",
            "KVAF_URMA_EPOCH_OWNER_MAX_CLOSE_RETRY_LIMIT",
            "args->magic != KVAF_URMA_EPOCH_OWNER_MAGIC",
            "args->abi_version != KVAF_URMA_EPOCH_OWNER_ABI_VERSION",
            "args->struct_size != sizeof(*args)",
            "args->flags != 0",
        ):
            self.assertIn(required, body)


if __name__ == "__main__":
    unittest.main()
