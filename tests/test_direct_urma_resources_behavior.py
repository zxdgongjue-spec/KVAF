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
            [str(self.harness), scenario],
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

    def test_foreign_already_open_is_never_closed(self) -> None:
        self.run_scenario("foreign_already_open")

    def test_open_result_copy_failure_closes_exact_session(self) -> None:
        self.run_scenario("open_d2h_failure")

    def test_lost_close_result_retries_same_session(self) -> None:
        self.run_scenario("close_d2h_retry")

    def test_local_free_failure_does_not_repeat_provider_close(self) -> None:
        self.run_scenario("free_retry")

    def test_stream_destroy_failure_is_retryable_after_transport_close(self) -> None:
        self.run_scenario("destroy_stream_retry")


if __name__ == "__main__":
    unittest.main()
