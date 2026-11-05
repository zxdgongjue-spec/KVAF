"""Host-only regression for the public AICPU CLI and package installation."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "access_operator/demo/aicpu_batch_get/python"))
import test_batch_get as functional
import benchmark_gather_read as benchmark

EIDS = "00112233445566778899aabbccddeeff,102132435465768798a9bacbdcedfe0f"
SCATTER = "KvafAicpuCpuPushScatterA12V24"


class PublicCliTests(unittest.TestCase):
    def test_both_algorithms_accept_candidate_lists_without_json(self):
        with patch.dict(os.environ, {}, clear=True):
            args = functional.parse_args(["--npu-eids", EIDS, "--modes", "gather_read", "cpu_push_scatter"])
            self.assertEqual(args.npu_eids, EIDS.split(","))
            for mode in ("gather_read", "cpu_push_scatter"):
                args = benchmark.parse_args(["--npu-eids", EIDS], mode=mode)
                self.assertEqual(args.npu_eids, EIDS.split(","))
                self.assertFalse(hasattr(args, "kernel_json"))

    def test_auto_discovery_and_existing_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(functional.parse_args([]).npu_eids, [])
            self.assertEqual(benchmark.parse_args([]).npu_eids, [])
        with patch.dict(os.environ, {"KVAF_NPU_EIDS": EIDS}, clear=True):
            self.assertEqual(functional.parse_args([]).npu_eids, EIDS.split(","))
            self.assertEqual(benchmark.parse_args([]).npu_eids, EIDS.split(","))

    def test_retired_options_are_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            for parse in (functional.parse_args, benchmark.parse_args):
                for argv in (["--kernel-json", "/tmp/kernel.json"], ["--source-mode", "direct"],
                             ["--urma-device", "udma0"], ["--eid-index", "0"], ["--npu-eid", EIDS[:32]]):
                    with self.subTest(parse=parse, argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        parse(argv)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                functional.parse_args(["--modes", "gather_read_direct"])


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "cann"
        (self.root / "conf").mkdir(parents=True)
        (self.root / "conf/ascend_package_load.ini").write_text("# loader\n")
        self.artifacts = Path(self.temp.name) / "artifacts"
        self.artifacts.mkdir()
        self.entries = {name: {} for name in ("KvafAicpuBatchPut", "KvafAicpuGatherRead", "KvafAicpuUmdkControl")}
        self.write_json()
        with tarfile.open(self.artifacts / "kvaf-aicpu-urma.tar.gz", "w:gz") as archive:
            info = tarfile.TarInfo("aicpu_kernels_device/libkvaf_aicpu_urma.so")
            info.size = 4
            archive.addfile(info, io.BytesIO(b"test"))

    def write_json(self):
        (self.artifacts / "kernel.json").write_text(json.dumps(self.entries))

    def installer(self, *args):
        return subprocess.run(["bash", str(ROOT / "access_operator/src/aicpu_urma/package/install_kernel.sh"), *map(str, args)],
                              capture_output=True, text=True)

    def resolve_root(self, **environment):
        env = dict(os.environ)
        env.pop("ASCEND_HOME_PATH", None)
        env.pop("ASCEND_CANN_PACKAGE_PATH", None)
        env.update(environment)
        return subprocess.run(["bash", "-c", 'source "$1"; kvaf_find_cann_root', "bash", str(ROOT / "scripts/cann_root.sh")],
                              env=env, capture_output=True, text=True)

    def test_root_selection_matches_runtime_contract(self):
        alias = Path(self.temp.name) / "latest"
        alias.symlink_to(self.root, target_is_directory=True)
        result = self.resolve_root(ASCEND_HOME_PATH=str(alias), ASCEND_CANN_PACKAGE_PATH=str(self.root))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.strip()), self.root.resolve())
        other = Path(self.temp.name) / "other"
        other.mkdir()
        result = self.resolve_root(ASCEND_HOME_PATH=str(self.root), ASCEND_CANN_PACKAGE_PATH=str(other))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("roots differ", result.stderr)
        self.assertNotEqual(self.resolve_root(ASCEND_HOME_PATH=str(other / "missing"),
                                            ASCEND_CANN_PACKAGE_PATH=str(self.root)).returncode, 0)

    def test_install_and_scatter_requirement(self):
        result = self.installer("--require-scatter", "--install", self.artifacts, self.root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Scatter missing", result.stderr)
        self.assertFalse((self.root / "opp").exists())
        result = self.installer("--install", self.artifacts, self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.installer("--require-scatter", "--check", self.root)
        self.assertNotEqual(result.returncode, 0)
        self.entries[SCATTER] = {}
        self.write_json()
        result = self.installer("--require-scatter", "--install", self.artifacts, self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.installer("--require-scatter", "--check", self.root, self.artifacts)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.entries["different-build"] = {}
        self.write_json()
        result = self.installer("--check", self.root, self.artifacts)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("differs from build artifact", result.stderr)


if __name__ == "__main__":
    unittest.main()
