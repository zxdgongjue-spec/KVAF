"""Check real Runtime CMake capability flags without needing CANN or device libraries."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CMAKE = os.environ.get("KVAF_TEST_CMAKE_COMMAND", "cmake")
FEATURES = ("KVAF_ENABLE_AIV_BATCH_GET", "KVAF_ENABLE_AIV_BATCH_PUT",
            "KVAF_ENABLE_AICPU_URMA", "KVAF_ENABLE_AICPU_CPU_PUSH")


class RuntimeBuildFlagsTests(unittest.TestCase):
    def test_runtime_compilation_matches_enabled_providers(self):
        # Stub only dependency targets; configure the production runtime/CMakeLists.txt.
        # Empty INTERFACE dependencies intentionally export no feature macros, as the
        # real access operators' PRIVATE definitions do not reach NpuRuntime either.
        project = '''cmake_minimum_required(VERSION 3.18)
project(kvaf_runtime_flags LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(KVAF_BUILD_REMOTE_CPU_SERVICE OFF)
set(ASCEND_CANN_PACKAGE_PATH "${CMAKE_CURRENT_BINARY_DIR}/unused-cann")
foreach(dependency IN ITEMS kvaf_common kvaf_meta_manager kvaf_access_operator
        kvaf_default_access_operator kvaf_ubmem_operator kvaf_execute_batch_get_host
        kvaf_aicpu_access_operator kvaf_umdk_host kvaf_execute_batch_put_host)
  add_library(${dependency} INTERFACE)
endforeach()
add_subdirectory("${KVAF_RUNTIME_SOURCE_DIR}" runtime)
'''
        configurations = ((True, True, True, True), (True, True, False, False),
                          (True, False, False, False),
                          (False, False, True, False), (False, False, True, True))
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "CMakeLists.txt").write_text(project)
            for index, enabled in enumerate(configurations):
                with self.subTest(features=dict(zip(FEATURES, enabled))):
                    build = source / f"build-{index}"
                    result = subprocess.run(
                        [CMAKE, "-G", "Unix Makefiles", "-S", str(source), "-B", str(build),
                         f"-DKVAF_RUNTIME_SOURCE_DIR={ROOT / 'runtime'}"] +
                        [f"-D{name}={'ON' if value else 'OFF'}"
                         for name, value in zip(FEATURES, enabled)],
                        capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    commands = json.loads((build / "compile_commands.json").read_text())
                    entries = [entry for entry in commands
                               if Path(entry["file"]).name == "npu_runtime.cc"]
                    self.assertEqual(len(entries), 1)
                    arguments = entries[0].get("arguments") or shlex.split(entries[0]["command"])
                    definitions = {arg[2:] for arg in arguments if arg.startswith("-D")}
                    for name, value in zip(FEATURES, enabled):
                        self.assertEqual(f"{name}=1" in definitions, value,
                                         f"{name} missing or leaked in NpuRuntime compilation: "
                                         f"{sorted(definitions)}")


if __name__ == "__main__":
    unittest.main()
