#!/usr/bin/env python3

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class DirectUrmaSourceSyntaxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.smoke_source = (
            ROOT
            / "access_operator"
            / "a5"
            / "smoke"
            / "aiv_simt_direct_urma_h2d_smoke.cpp"
        ).read_text(encoding="utf-8")
        cls.resources_source = (
            ROOT
            / "access_operator"
            / "a5"
            / "src"
            / "direct_urma_resources.cpp"
        ).read_text(encoding="utf-8")
        cls.mapping_source = (
            ROOT
            / "access_operator"
            / "a5"
            / "src"
            / "host_ddr_mapping.cpp"
        ).read_text(encoding="utf-8")
        cls.runtime_kernel_header = (
            ROOT
            / "tests"
            / "fake_acl"
            / "runtime"
            / "rt_external_kernel.h"
        ).read_text(encoding="utf-8")
        cls.aiv_kernel_source = (
            ROOT
            / "access_operator"
            / "a5"
            / "src"
            / "aiv_simt_urma_h2d_kernel.cpp"
        ).read_text(encoding="utf-8")

    def test_host_control_plane_sources_compile(self) -> None:
        compiler = shutil.which("c++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("no C++ compiler is installed")
        command = [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsyntax-only",
            "-I",
            str(ROOT / "tests" / "fake_acl"),
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "access_operator" / "include"),
            "-I",
            str(ROOT / "access_operator" / "a5" / "include"),
            "-I",
            str(ROOT / "access_operator" / "a5" / "provider" / "include"),
            str(
                ROOT
                / "access_operator"
                / "a5"
                / "src"
                / "direct_urma_resources.cpp"
            ),
            str(
                ROOT
                / "access_operator"
                / "a5"
                / "smoke"
                / "aiv_simt_direct_urma_h2d_smoke.cpp"
            ),
        ]
        result = subprocess.run(
            command, check=False, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fake_cann_header_does_not_export_private_kernel_type(self) -> None:
        self.assertNotIn("rtKernelType_t", self.runtime_kernel_header)
        self.assertNotIn(
            "KERNEL_TYPE_AICPU_KFC", self.runtime_kernel_header
        )

    def test_transport_safety_marker_precedes_runtime_detach(self) -> None:
        provisional = self.smoke_source.index(
            'pre_runtime_cleanup.reason = '
            '"transport_safe_runtime_cleanup_pending"'
        )
        marker = self.smoke_source.index(
            "PersistTerminalMarkerUntilDurable(\n"
            '      options, pre_runtime_cleanup, "before_runtime_cleanup")',
            provisional,
        )
        reset = self.smoke_source.index("aclrtResetDevice", marker)
        pass_replace = self.smoke_source.index(
            "PersistTerminalMarkerUntilDurable(\n"
            '      options, result, "after_runtime_cleanup")',
            reset,
        )
        self.assertLess(provisional, marker)
        self.assertLess(marker, reset)
        self.assertLess(reset, pass_replace)

    def test_safe_terminal_marker_retries_until_directory_fsync_succeeds(
        self,
    ) -> None:
        retry_start = self.smoke_source.index(
            "void PersistTerminalMarkerUntilDurable"
        )
        retry_end = self.smoke_source.index(
            "void PublishDeviceSafeAbortWakeEdgeBestEffort", retry_start
        )
        retry_body = self.smoke_source[retry_start:retry_end]
        self.assertIn("IgnoreLifecycleSignals();", retry_body)
        self.assertIn("for (;;)", retry_body)
        self.assertIn("WriteTerminalMarker(options, result)", retry_body)
        self.assertIn(
            "sleep_for(kTerminalMarkerRetryInterval)", retry_body
        )
        self.assertIn("worker remains live and will retry", retry_body)

    def test_device_safe_abort_edge_follows_durable_full_proof(self) -> None:
        main_start = self.smoke_source.index("int main(int argc")
        marker = self.smoke_source.index(
            "PersistTerminalMarkerUntilDurable(\n"
            '      options, pre_runtime_cleanup, "before_runtime_cleanup")',
            main_start,
        )
        wake_edge = self.smoke_source.index(
            "PublishDeviceSafeAbortWakeEdgeBestEffort(",
            marker,
        )
        reset = self.smoke_source.index("aclrtResetDevice", wake_edge)
        guarded_publish = self.smoke_source[marker:wake_edge]
        self.assertIn(
            "result.outcome == EndpointOutcome::kFailedSafe",
            guarded_publish,
        )
        self.assertLess(marker, wake_edge)
        self.assertLess(wake_edge, reset)

    def test_reopen_rejects_every_retained_resource(self) -> None:
        self.assertIn("bool HasRetainedState() const", self.resources_source)
        open_start = self.resources_source.index(
            "Status DirectUrmaResources::Open"
        )
        close_start = self.resources_source.index(
            "Status DirectUrmaResources::Close", open_start
        )
        open_body = self.resources_source[open_start:close_start]
        self.assertIn("impl_->HasRetainedState()", open_body)

    def test_runtime_and_hal_loading_do_not_search_arbitrary_sonames(self) -> None:
        self.assertIn("RTLD_NOLOAD", self.resources_source)
        self.assertNotIn('"libruntime.so"', self.resources_source)
        self.assertIn(
            '"/usr/local/Ascend/driver/lib64/driver/libascend_hal.so"',
            self.mapping_source,
        )

    def test_data_and_shutdown_use_the_resources_owned_fifo_stream(self) -> None:
        self.assertIn(
            "resources->EnqueueBatch(",
            self.smoke_source,
        )
        self.assertIn(
            "resources->SynchronizeDataPlane()",
            self.smoke_source,
        )
        self.assertNotIn(
            "LaunchAivSimtUrmaH2D(\n        stream", self.smoke_source
        )
        enqueue = self.resources_source.index(
            "Status DirectUrmaResources::EnqueueBatch"
        )
        close = self.resources_source.index(
            "Status DirectUrmaResources::Close", enqueue
        )
        enqueue_body = self.resources_source[enqueue:close]
        self.assertIn("impl_->data_plane_mutex", enqueue_body)
        self.assertIn("impl_->data_plane_frozen", enqueue_body)
        self.assertIn("impl_->control_stream", enqueue_body)
        close_body = self.resources_source[close:]
        self.assertIn("impl_->data_plane_frozen = true", close_body)
        self.assertIn(
            "LaunchAivEpochOwnerShutdown(\n"
            "            impl_->control_stream",
            close_body,
        )

    def test_inactive_descriptor_suffix_is_fully_unchanged(self) -> None:
        self.assertIn(
            "completed_descriptors(\n      descriptor_capacity)",
            self.smoke_source,
        )
        self.assertIn(
            "allocations->descriptors,\n"
            "                          descriptor_capacity",
            self.smoke_source,
        )
        self.assertIn(
            "std::memcmp(\n"
            "          &completed_descriptors[request], "
            "&empty_descriptors[request],",
            self.smoke_source,
        )

    def test_exception_unwind_retains_combined_ownership(self) -> None:
        self.assertIn("class RetainEndpointOnException", self.smoke_source)
        self.assertIn("std::uncaught_exceptions()", self.smoke_source)
        self.assertIn(
            "TryRecoverRetainedAfterException", self.smoke_source
        )
        self.assertIn("std::set_terminate(FailClosedTerminate)", self.smoke_source)
        self.assertIn(
            "KVAF_A5_DIRECT_TERMINATE_WITH_OWNERSHIP",
            self.smoke_source,
        )

    def test_host_exception_never_releases_ddr_without_device_close_proof(
        self,
    ) -> None:
        recovery_start = self.smoke_source.index(
            "bool TryRecoverRetainedAfterException"
        )
        recovery_end = self.smoke_source.index(
            "}  // namespace", recovery_start
        )
        recovery_body = self.smoke_source[recovery_start:recovery_end]
        proof = recovery_body.index(
            "ReadExactMarker(device_done, provider_closed_pass.c_str())"
        )
        safe_abort = recovery_body.index(
            "ValidDeviceSafeAbortProof(options)", proof
        )
        fail_closed = recovery_body.index(
            "if (!device_teardown_safe)", safe_abort
        )
        mapping_close = recovery_body.index(
            "CloseMappingWithRetry", fail_closed
        )
        self.assertIn(
            'JoinPath(options.exchange_dir, "device.done")',
            recovery_body,
        )
        self.assertIn(
            'ControlMarker(options, "provider_closed", "PASS")',
            recovery_body,
        )
        self.assertIn("return false;", recovery_body[fail_closed:mapping_close])
        self.assertLess(proof, mapping_close)
        self.assertLess(safe_abort, mapping_close)
        self.assertLess(fail_closed, mapping_close)

    def test_host_wait_accepts_late_nonce_bound_evidence(self) -> None:
        self.assertIn("PeerMarkerWait WaitForPeerEvidence", self.smoke_source)
        wait_start = self.smoke_source.index(
            "PeerMarkerWait WaitForPeerEvidence"
        )
        wait_end = self.smoke_source.index(
            "[[noreturn]] void QuarantineUnsafeTeardown", wait_start
        )
        wait_body = self.smoke_source[wait_start:wait_end]
        self.assertIn("for (;;)", wait_body)
        self.assertIn("late evidence will be", wait_body)
        self.assertNotIn("kTimedOut", wait_body)
        self.assertIn("ValidDeviceSafeAbortProof", wait_body)

    def test_close_ready_publish_failure_retries_before_provider_close(self) -> None:
        publish_loop = self.smoke_source.index(
            "while (!AtomicWriteText(close_ready, close_ready_marker))"
        )
        provider_close = self.smoke_source.index(
            "bool provider_local_cleanup_complete = false;", publish_loop
        )
        retry_body = self.smoke_source[publish_loop:provider_close]
        self.assertIn("ReadExactMarker", retry_body)
        self.assertIn("sleep_for", retry_body)
        self.assertNotIn("CloseProviderWithRetry", retry_body)

    def test_checked_retryable_close_never_ages_into_unsafe(self) -> None:
        retry_start = self.smoke_source.index(
            "bool CloseProviderWithRetry"
        )
        retry_end = self.smoke_source.index(
            "DirectUrmaOpenSpec MakeHostOpenSpec", retry_start
        )
        retry_body = self.smoke_source[retry_start:retry_end]
        self.assertIn("for (std::uint64_t attempt = 1;; ++attempt)", retry_body)
        self.assertIn("transport_teardown_retryable()", retry_body)
        self.assertIn("kSlowRetryInterval", retry_body)
        self.assertIn("active-session lock", retry_body)
        self.assertNotIn("kMaxCheckedCloseAttempts", retry_body)
        self.assertNotIn("exhausted", retry_body)

    def test_aicpu_stream_sync_gets_one_bounded_retry(self) -> None:
        self.assertIn("first_sync_result", self.resources_source)
        self.assertIn("final_sync_result", self.resources_source)
        self.assertIn("failed twice", self.resources_source)

    def test_safe_open_failure_reports_provider_and_owner_diagnostics(self) -> None:
        for field in (
            '"provider_result_status="',
            '" provider_status="',
            '" owner_result_code="',
            "std::to_string(impl_->result.status)",
            "std::to_string(impl_->result.provider_status)",
            "std::to_string(ready_status.result_code)",
            '" management_stage="',
            '" management_raw_errno="',
            '" management_effective_errno="',
            '" management_operation_status="',
            '" management_auxiliary_status="',
            "FormatManagementDiagnostic(impl_->result)",
        ):
            self.assertIn(field, self.resources_source)

    def test_ready_observation_cannot_race_provider_open_cleanup(self) -> None:
        open_start = self.resources_source.index(
            "Status DirectUrmaResources::Open"
        )
        close_start = self.resources_source.index(
            "Status DirectUrmaResources::Close", open_start
        )
        open_body = self.resources_source[open_start:close_start]
        self.assertIn(
            "kEpochOwnerReadyAivPollBudget = 1U", self.resources_source
        )
        self.assertIn(
            "kEpochOwnerShutdownAivPollBudget = 20000000U",
            self.resources_source,
        )
        self.assertIn("ready_still_pending", open_body)
        self.assertIn(
            "KVAF_URMA_EPOCH_OWNER_STATE_EMPTY", open_body
        )
        self.assertIn(
            "KVAF_URMA_EPOCH_OWNER_STATE_OPENING", open_body
        )
        self.assertIn(
            "KVAF_URMA_EPOCH_OWNER_STATE_CLOSING", open_body
        )
        self.assertIn(
            "KVAF_URMA_EPOCH_OWNER_STATE_CLOSE_RETRYING", open_body
        )
        self.assertIn("ready_result_identity_valid", open_body)
        self.assertIn("ready_owner_identity_valid", open_body)
        self.assertIn("ready_cleanup_result_valid", open_body)
        self.assertIn("ready_cleanup_status_fields_valid", open_body)
        self.assertIn("ready_cleanup_state_is_monotonic", open_body)
        self.assertIn("ready_observation_can_progress", open_body)
        self.assertIn(
            "::usleep(kEpochOwnerReadyRetryMicroseconds)", open_body
        )
        self.assertNotIn("ready_deadline", open_body)

    def test_ready_observation_reports_rate_limited_host_progress(self) -> None:
        open_start = self.resources_source.index(
            "Status DirectUrmaResources::Open"
        )
        close_start = self.resources_source.index(
            "Status DirectUrmaResources::Close", open_start
        )
        open_body = self.resources_source[open_start:close_start]
        for expected in (
            '"KVAF_A5_AICPU_OWNER_PROGRESS phase=open polls=%llu "',
            '"observer_elapsed_ms=%lld aiv_identity_valid=%u "',
            '"owner_identity_valid=%u open_result_valid=%u "',
            "progress_now >= next_progress_log",
            "progress_now + std::chrono::seconds(1)",
        ):
            self.assertIn(expected, open_body)

    def test_aiv_ready_wait_does_not_make_cleanup_progress_terminal(
        self,
    ) -> None:
        terminal_start = self.aiv_kernel_source.index(
            "__aicore__ inline bool EpochOwnerStateIsTerminal"
        )
        terminal_end = self.aiv_kernel_source.index(
            "KvafAivEpochOwnerWaitReadyKernel", terminal_start
        )
        terminal_body = self.aiv_kernel_source[terminal_start:terminal_end]
        self.assertIn("KVAF_URMA_EPOCH_OWNER_STATE_CLOSED", terminal_body)
        self.assertIn(
            "KVAF_URMA_EPOCH_OWNER_STATE_OPEN_FAILED_SAFE", terminal_body
        )
        return_expression = terminal_body.split("return", 1)[1]
        self.assertNotIn(
            "KVAF_URMA_EPOCH_OWNER_STATE_CLOSING", return_expression
        )
        self.assertNotIn(
            "KVAF_URMA_EPOCH_OWNER_STATE_CLOSE_RETRYING",
            return_expression,
        )

        wait_start = self.aiv_kernel_source.index(
            "KvafAivEpochOwnerWaitReadyKernel"
        )
        wait_end = self.aiv_kernel_source.index(
            "KvafAivEpochOwnerShutdownKernel", wait_start
        )
        wait_body = self.aiv_kernel_source[wait_start:wait_end]
        self.assertIn("last_observed_state", wait_body)
        self.assertIn("last_observed_sequence", wait_body)
        self.assertIn(
            "state == KVAF_URMA_EPOCH_OWNER_STATE_CLOSING", wait_body
        )
        self.assertIn(
            "state == KVAF_URMA_EPOCH_OWNER_STATE_CLOSE_RETRYING",
            wait_body,
        )

    def test_smoke_payload_is_bounded_before_resource_acquisition(self) -> None:
        self.assertIn(
            "kSmokeMaxBatchBytes = 64ULL * 1024ULL * 1024ULL",
            self.smoke_source,
        )

    def test_batch_defaults_and_legacy_total_bytes_are_unambiguous(self) -> None:
        for expected in (
            "kDefaultTransferCount = 4096",
            "kDefaultDescriptorCapacity = 4096",
            "kDefaultBytesPerTransfer = 1024",
            'key == "--transfer-count"',
            'key == "--descriptor-capacity"',
            'key == "--bytes-per-transfer"',
            'key == "--batch-iterations"',
            'key == "--bytes"',
            "options->transfer_count = 1;",
            "options->descriptor_capacity = 1;",
            "options->bytes_per_transfer = legacy_bytes;",
            "cannot be mixed with",
        ):
            self.assertIn(expected, self.smoke_source)

    def test_urma_management_is_default_and_tcp_is_explicit_fallback(self) -> None:
        for expected in (
            "DirectUrmaManagementMode::kUrma",
            'key == "--management-mode"',
            'value == "ub"',
            'value == "tcp"',
            'key == "--management-jetty-id"',
            'key == "--control-server-address"',
            'key == "--control-bind-address"',
            "options->management_mode == DirectUrmaManagementMode::kTcp",
            "NumericIpFamily(\n        options->control_server_address)",
            "FormatUrmaEid(options.host_eid)",
            "FormatUrmaEid(options.device_eid)",
            "spec.control_server_address = options.control_server_address",
            "spec.control_bind_address = options.control_bind_address",
            "spec.control_bind_address.clear()",
            "spec.management_jetty_id = options.management_jetty_id",
        ):
            self.assertIn(expected, self.smoke_source)
        for expected in (
            "KVAF_URMA_PROVIDER_URMA_MANAGEMENT",
            "KVAF_URMA_PROVIDER_EXPLICIT_TCP_MANAGEMENT",
            "config.management_peer_address",
            "config.management_local_address",
            "config.management_jetty_id",
            "ParseUrmaEid(spec.local_eid",
            "ParseUrmaEid(spec.peer_eid",
            "FormatUrmaEid(local_eid)",
            "FormatUrmaEid(peer_eid)",
        ):
            self.assertIn(expected, self.resources_source)
        self.assertIn(
            "config.control_server_address", self.resources_source
        )
        self.assertIn("config.control_bind_address", self.resources_source)
        self.assertNotIn("config.local_eid", self.resources_source)
        self.assertNotIn("config.peer_eid", self.resources_source)

    def test_large_batch_uses_one_simt_plan_and_bounded_queue_waves(self) -> None:
        self.assertIn(
            "request += kAivUrmaSimtLanes", self.aiv_kernel_source
        )
        self.assertEqual(
            self.aiv_kernel_source.count(
                "asc_vf_call<BuildH2DDescriptorsSimt>"
            ),
            1,
        )
        for expected in (
            "wave_capacity",
            "wave_submitted < wave_capacity",
            "++completion->reserved[0]",
            "PollDirectCq(",
            "kDirectCounterOverflow",
            "ready_count > kMaxQueueCounter - direct_channel->sq_head",
        ):
            self.assertIn(expected, self.aiv_kernel_source)

    def test_batch_iterations_reuse_one_open_and_validate_every_round(self) -> None:
        transfer_start = self.smoke_source.index(
            "DeviceTransferOutcome RunDeviceTransfer"
        )
        transfer_end = self.smoke_source.index(
            "EndpointResult RunHost", transfer_start
        )
        transfer = self.smoke_source[transfer_start:transfer_end]
        self.assertIn(
            "iteration < options.batch_iterations", transfer
        )
        self.assertIn("params.generation", transfer)
        self.assertIn("completion.generation", transfer)
        self.assertIn("submission_budget", transfer)
        self.assertNotIn("resources->Open", transfer)
        self.assertIn("canary_output", transfer)

    def test_zero_wqe_lifecycle_is_explicit(self) -> None:
        self.assertIn(
            "if (options.transfer_count == 0)", self.smoke_source
        )
        self.assertIn(
            "KVAF_A5_DIRECT_ZERO_WQE_OK", self.smoke_source
        )


if __name__ == "__main__":
    unittest.main()
