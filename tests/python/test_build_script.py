"""Exercise build.sh with simulated compilation and real temporary installation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCATTER = "KvafAicpuCpuPushScatterA12V24"
JSON_PATH = "opp/vendors/cust/op_impl/aicpu/config/kvaf_aicpu_urma.json"
TAR_PATH = "opp/vendors/cust/op_impl/aicpu/kernel/kvaf-aicpu-urma.tar.gz"

# Only compilation is simulated. The real build entry point selects its
# options, checks dependencies and invokes the real installer on a temp CANN.
FAKE_CMAKE = r'''
import io
import json
import os
from pathlib import Path
import sys
import tarfile

args = sys.argv[1:]
with open(os.environ["KVAF_TEST_COMMANDS"], "a") as log:
    log.write(json.dumps(args) + "\n")
if "--build" not in args:
    build = Path(args[args.index("-B") + 1])
    build.mkdir(parents=True, exist_ok=True)
    cache = build / "test-cache.json"
    options = json.loads(cache.read_text()) if cache.exists() else {}
    options.update(arg[2:].split("=", 1) for arg in args if arg.startswith("-D"))
    cache.write_text(json.dumps(options))
else:
    if os.environ.get("KVAF_TEST_FAIL_BUILD"):
        sys.exit(42)
    build = Path(args[args.index("--build") + 1])
    options = json.loads((build / "test-cache.json").read_text())
    if options["KVAF_BUILD_PYTHON"] == "ON":
        (build / "python/kvaf").mkdir(parents=True, exist_ok=True)
        (build / "python/kvaf/_kvaf.so").write_bytes(b"simulated Host binding")
    if options["KVAF_ENABLE_AICPU_URMA"] == "ON":
        artifacts = build / "access_operator/src/aicpu_urma"
        artifacts.mkdir(parents=True, exist_ok=True)
        entries = {name: {} for name in (
            "KvafAicpuBatchPut", "KvafAicpuGatherRead", "KvafAicpuUmdkControl")}
        if options["KVAF_ENABLE_AICPU_CPU_PUSH"] == "ON" and not os.environ.get("KVAF_TEST_OMIT_SCATTER"):
            entries["KvafAicpuCpuPushScatterA12V24"] = {}
        (artifacts / "kernel.json").write_text(json.dumps(entries))
        with tarfile.open(artifacts / "kvaf-aicpu-urma.tar.gz", "w:gz") as archive:
            info = tarfile.TarInfo("aicpu_kernels_device/libkvaf_aicpu_urma.so")
            info.size = 4
            archive.addfile(info, io.BytesIO(b"test"))
'''

FAKE_PYTHON = r'''
import os
from pathlib import Path
import sys

if sys.argv[1:] != ["-m", "pybind11", "--cmakedir"]:
    sys.exit("unexpected Python invocation")
Path(os.environ["KVAF_TEST_PYTHON_USED"]).touch()
print(os.environ["KVAF_TEST_PYBIND_DIR"])
'''


class BuildScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="kvaf build ")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.cann = self.directory / "cann"
        (self.cann / "conf").mkdir(parents=True)
        (self.cann / "conf/ascend_package_load.ini").write_text("# other packages\n")
        (self.cann / "include/acl").mkdir(parents=True)
        (self.cann / "include/acl/acl.h").touch()
        self.umdk = self.directory / "umdk"
        (self.umdk / "include").mkdir(parents=True)
        (self.umdk / "include/urma_api.h").touch()
        self.build = self.directory / "build"
        self.bin = self.directory / "bin"
        self.bin.mkdir()
        self.write_tool("cmake", FAKE_CMAKE)
        self.write_tool("python tool", FAKE_PYTHON)
        self.commands = self.directory / "commands.jsonl"
        self.python_used = self.directory / "python-used"
        self.env = dict(os.environ)
        for key in list(self.env):
            if key.startswith(("KVAF_", "ASCEND_")):
                self.env.pop(key)
        self.env.update(
            PATH=str(self.bin) + os.pathsep + self.env.get("PATH", ""),
            PYTHON=str(self.bin / "python tool"),
            ASCEND_HOME_PATH=str(self.cann),
            ASCEND_CANN_PACKAGE_PATH=str(self.cann),
            KVAF_UMDK_ROOT=str(self.umdk),
            KVAF_TEST_COMMANDS=str(self.commands),
            KVAF_TEST_PYTHON_USED=str(self.python_used),
            KVAF_TEST_PYBIND_DIR=str(self.directory / "pybind cmake"),
        )

    def write_tool(self, name, source):
        path = self.bin / name
        path.write_text("#!" + sys.executable + "\n" + textwrap.dedent(source))
        path.chmod(0o755)

    def run_build(self, mode="npu", *args, **environment):
        return subprocess.run(
            ["bash", str(ROOT / "build.sh"), mode, "--build-dir", str(self.build), "--jobs", "2", *args],
            env=dict(self.env, **environment), capture_output=True, text=True,
        )

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def options(self):
        return json.loads((self.build / "test-cache.json").read_text())

    def test_default_npu_builds_all_and_installs_one_package(self):
        result = self.run_build(KVAF_AICPU_CXX="/custom compiler/cxx")
        self.assert_success(result)
        options = self.options()
        for key in ("KVAF_ENABLE_AIV_BATCH_GET", "KVAF_ENABLE_AIV_BATCH_PUT",
                    "KVAF_ENABLE_AICPU_URMA", "KVAF_ENABLE_AICPU_CPU_PUSH", "KVAF_BUILD_PYTHON"):
            self.assertEqual(options[key], "ON")
        self.assertEqual(options["KVAF_BUILD_REMOTE_CPU_SERVICE"], "OFF")
        self.assertEqual(options["KVAF_AICPU_CXX"], "/custom compiler/cxx")
        self.assertEqual(options["Python_EXECUTABLE"], self.env["PYTHON"])
        self.assertEqual(options["Python3_EXECUTABLE"], self.env["PYTHON"])
        self.assertEqual(options["pybind11_DIR"], self.env["KVAF_TEST_PYBIND_DIR"])
        self.assertEqual(len(self.commands.read_text().splitlines()), 2)  # One configure/build.
        self.assertEqual(result.stdout.count("NPU Python package:"), 1)
        self.assertTrue((self.build / "python/kvaf/_kvaf.so").is_file())
        self.assertIn(SCATTER, json.loads((self.cann / JSON_PATH).read_text()))
        self.assertIn("CHECK_OK", result.stdout)
        self.assertEqual((self.cann / TAR_PATH).read_bytes(),
                         (self.build / "access_operator/src/aicpu_urma/kvaf-aicpu-urma.tar.gz").read_bytes())

    def test_trim_clears_cached_aicpu_flags_and_leaves_installed_package(self):
        self.assert_success(self.run_build())
        installed = (self.cann / TAR_PATH).read_bytes()
        registration = (self.cann / "conf/ascend_package_load.ini").read_bytes()
        # AIV-only must not inspect or download UMDK, even with stale AICPU artifacts.
        self.assert_success(self.run_build("npu", "--without-aicpu", KVAF_UMDK_ROOT=str(self.directory)))
        for key in ("KVAF_ENABLE_AICPU_URMA", "KVAF_ENABLE_AICPU_CPU_PUSH"):
            self.assertEqual(self.options()[key], "OFF")
        self.assertEqual(self.options()["KVAF_ENABLE_AIV_BATCH_GET"], "ON")
        self.assertEqual((self.cann / TAR_PATH).read_bytes(), installed)
        self.assertEqual((self.cann / "conf/ascend_package_load.ini").read_bytes(), registration)
        self.assert_success(self.run_build())
        self.assertEqual(self.options()["KVAF_ENABLE_AICPU_CPU_PUSH"], "ON")

    def test_aicpu_only_auto_installs_gather_and_optional_scatter(self):
        self.assert_success(self.run_build("aicpu-urma"))
        self.assertEqual(self.options()["KVAF_ENABLE_AIV_BATCH_GET"], "OFF")
        self.assertEqual(self.options()["KVAF_ENABLE_AIV_BATCH_PUT"], "OFF")
        self.assertNotIn(SCATTER, json.loads((self.cann / JSON_PATH).read_text()))
        self.assert_success(self.run_build("aicpu-urma", "--cpu-push"))
        self.assertIn(SCATTER, json.loads((self.cann / JSON_PATH).read_text()))

    def test_without_python_still_installs_aicpu(self):
        self.assert_success(self.run_build("npu", "--without-python"))
        self.assertEqual(self.options()["KVAF_BUILD_PYTHON"], "OFF")
        self.assertFalse(self.python_used.exists())
        self.assertFalse((self.build / "python/kvaf").exists())
        self.assertTrue((self.cann / JSON_PATH).is_file())
        self.assert_success(self.run_build("npu", "--without-python", "--without-aicpu",
                                           KVAF_UMDK_ROOT=str(self.directory)))

    def test_legacy_install_and_management_without_build_tools(self):
        self.assert_success(self.run_build("npu", "--install-kernel"))
        before = self.commands.read_text()
        (self.cann / "include/acl/acl.h").unlink()
        self.assert_success(self.run_build("npu", "--check-kernel", PYTHON="/absent/python"))
        self.assertEqual(self.commands.read_text(), before)
        self.assert_success(self.run_build("npu", "--uninstall-kernel"))
        self.assertFalse((self.cann / JSON_PATH).exists())
        self.assertFalse((self.cann / TAR_PATH).exists())
        self.assertEqual(self.commands.read_text(), before)

    def test_build_failure_does_not_replace_installed_package(self):
        self.assert_success(self.run_build())
        installed = (self.cann / TAR_PATH).read_bytes()
        result = self.run_build(KVAF_TEST_FAIL_BUILD="1")
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
        self.assertEqual((self.cann / TAR_PATH).read_bytes(), installed)
        self.assertNotIn("installed;", result.stdout)

    def test_full_build_rejects_package_without_scatter(self):
        result = self.run_build(KVAF_TEST_OMIT_SCATTER="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Scatter missing", result.stderr)
        self.assertFalse((self.cann / "opp").exists())

    def test_conflicting_options_fail_before_build(self):
        cases = [("cpu", "--without-aicpu"), ("aicpu-urma", "--without-aicpu")]
        cases += [("npu", "--without-aicpu", action) for action in
                  ("--install-kernel", "--check-kernel", "--uninstall-kernel")]
        for args in cases:
            with self.subTest(args=args):
                result = self.run_build(*args)
                self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.commands.exists())


if __name__ == "__main__":
    unittest.main()
