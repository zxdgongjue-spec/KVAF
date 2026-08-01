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

    def test_quiesced_stream_destroy_is_retried_and_retained(self) -> None:
        self.assertIn(
            "quiesced AIV transfer stream destroy failed; retrying once",
            self.smoke_source,
        )
        self.assertIn(
            "RetainEndpointOwnership(nullptr, nullptr, nullptr",
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

    def test_aicpu_stream_sync_gets_one_bounded_retry(self) -> None:
        self.assertIn("first_sync_result", self.resources_source)
        self.assertIn("final_sync_result", self.resources_source)
        self.assertIn("failed twice", self.resources_source)

    def test_smoke_payload_is_bounded_before_resource_acquisition(self) -> None:
        self.assertIn(
            "kSmokeMaxTransferBytes = 64ULL * 1024ULL * 1024ULL",
            self.smoke_source,
        )


if __name__ == "__main__":
    unittest.main()
