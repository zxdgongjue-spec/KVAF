#!/usr/bin/env python3

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
FAKE = ROOT / "tests" / "fake_acl"


class DirectUrmaResourcesBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = shutil.which("c++") or shutil.which("g++")
        if cls.compiler is None:
            raise unittest.SkipTest("no C++ compiler is installed")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.build = Path(cls.temporary.name)
        cls.library = cls.build / "libascendcl.so"
        cls.harness = cls.build / "direct-urma-resources-harness"

        include_args = [
            "-I", str(FAKE),
            "-I", str(ROOT / "include"),
            "-I", str(ROOT / "access_operator" / "include"),
            "-I", str(ROOT / "access_operator" / "a5" / "include"),
            "-I", str(
                ROOT / "access_operator" / "a5" / "provider" / "include"
            ),
        ]
        if sys.platform == "darwin":
            shared_args = [
                "-dynamiclib",
                "-Wl,-install_name,@rpath/libascendcl.so",
            ]
            rpath = "-Wl,-rpath,@loader_path"
        else:
            shared_args = ["-shared", "-Wl,-soname,libascendcl.so"]
            rpath = "-Wl,-rpath,$ORIGIN"

        library_build = subprocess.run(
            [
                cls.compiler,
                "-std=c++17",
                "-fPIC",
                "-Wall",
                "-Wextra",
                "-Werror",
                *shared_args,
                *include_args,
                str(FAKE / "fake_ascendcl.cpp"),
                "-o", str(cls.library),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if library_build.returncode != 0:
            raise RuntimeError(library_build.stderr)

        harness_build = subprocess.run(
            [
                cls.compiler,
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                *include_args,
                str(
                    ROOT / "access_operator" / "a5" / "src" /
                    "direct_urma_resources.cpp"
                ),
                str(FAKE / "direct_urma_resources_harness.cpp"),
                "-L", str(cls.build),
                "-lascendcl",
                rpath,
                "-ldl",
                "-o", str(cls.harness),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if harness_build.returncode != 0:
            raise RuntimeError(harness_build.stderr)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "temporary"):
            cls.temporary.cleanup()

    def run_scenario(self, scenario: str) -> None:
        result = subprocess.run(
            [str(self.harness), scenario, str(self.library)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_open_sync_transient_is_retried(self) -> None:
        self.run_scenario("sync_retry")

    def test_aicpu_launch_uses_kfc_kernel_type_abi(self) -> None:
        self.run_scenario("kernel_type")

    def test_double_sync_failure_retains_every_allocation(self) -> None:
        self.run_scenario("sync_twice")

    def test_open_launch_failure_retains_ambiguous_backing(self) -> None:
        self.run_scenario("open_launch_failure")

    def test_foreign_already_open_is_never_closed(self) -> None:
        self.run_scenario("foreign_already_open")

    def test_open_result_copy_failure_closes_exact_session(self) -> None:
        self.run_scenario("open_d2h_failure")

    def test_retryable_failed_open_preserves_session_for_close(self) -> None:
        self.run_scenario("open_cleanup_retryable")

    def test_host_retryable_failed_open_preserves_session_for_close(self) -> None:
        self.run_scenario("host_open_cleanup_retryable")

    def test_partial_runtime_capability_is_rejected_and_closed(self) -> None:
        self.run_scenario("runtime_capability_partial")

    def test_missing_runtime_public_abi_evidence_is_rejected(self) -> None:
        self.run_scenario("runtime_public_abi_missing")

    def test_lost_close_result_retries_same_session(self) -> None:
        self.run_scenario("close_d2h_retry")

    def test_provider_retry_relaunches_only_after_result_reread(self) -> None:
        self.run_scenario("close_d2h_provider_retryable")

    def test_prelaunch_copy_failure_is_safely_retryable(self) -> None:
        self.run_scenario("close_prelaunch_d2h_retry")

    def test_non_drained_queue_is_unsafe_without_launch(self) -> None:
        self.run_scenario("close_queue_not_drained")

    def test_close_launch_failure_is_ownership_unknown(self) -> None:
        self.run_scenario("close_launch_failure")

    def test_explicit_unsafe_close_is_not_retried(self) -> None:
        self.run_scenario("close_unsafe")

    def test_host_explicit_unsafe_close_is_not_retried(self) -> None:
        self.run_scenario("host_close_unsafe")

    def test_local_free_failure_does_not_repeat_provider_close(self) -> None:
        self.run_scenario("free_retry")

    def test_stream_destroy_failure_is_retryable_after_transport_close(self) -> None:
        self.run_scenario("destroy_stream_retry")


if __name__ == "__main__":
    unittest.main()
