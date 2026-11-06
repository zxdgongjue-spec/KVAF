"""Exercise production binding helpers with changing storage and stream state."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class Tensor:
    def __init__(self, pointer, count=8192, rows=8194):
        self.pointer, self.count, self.rows = pointer, count, rows
        self.calls = []

    def data_ptr(self):
        self.calls.append("pointer")
        return self.pointer

    def numel(self):
        self.calls.append("count")
        return self.count

    def size(self, dimension):
        self.calls.append("rows")
        assert dimension == 0
        return self.rows

    def __getattr__(self, name):
        raise AssertionError(f"trusted path accessed unnecessary metadata: {name}")


class GatherBindingFastpathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import pybind11
        except ImportError:
            raise unittest.SkipTest("pybind11 is required for the native binding-helper test")
        compiler = shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            raise unittest.SkipTest("a C++ compiler is required")
        cls.directory = tempfile.TemporaryDirectory(prefix="kvaf-gather-binding-")
        cls.addClassCleanup(cls.directory.cleanup)
        repo = Path(__file__).resolve().parents[2]
        output = Path(cls.directory.name) / ("_kvaf_gather_metadata_test" + sysconfig.get_config_var("EXT_SUFFIX"))
        command = [compiler, "-std=c++17", "-O1", "-shared", "-fPIC", "-fvisibility=hidden",
                   "-ffunction-sections", "-fdata-sections", f"-I{pybind11.get_include()}",
                   f"-I{sysconfig.get_paths()['include']}"]
        command += [f"-I{repo / p}" for p in ("tests/fake_acl", "bindings/include", "common/include",
                                             "runtime/include", "meta_manager/include", "access_operator/include")]
        command += (["-undefined", "dynamic_lookup", "-Wl,-dead_strip"] if sys.platform == "darwin"
                    else ["-Wl,--gc-sections"])
        command += [str(repo / "tests/gather_binding_fastpath_module.cc"), "-o", str(output)]
        compiled = subprocess.run(command, capture_output=True, text=True)
        if compiled.returncode:
            raise RuntimeError(compiled.stdout + compiled.stderr)
        spec = importlib.util.spec_from_file_location("_kvaf_gather_metadata_test", output)
        cls.native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.native)

    def test_live_pointers_capacities_and_explicit_stream(self):
        harness = self.native.Harness()
        tensors = [Tensor(0x1000 + 0x100 * i) for i in range(7)]
        stream = SimpleNamespace(npu_stream=0x1234)
        for iteration in range(3):
            tensors[0].count = 16384 + iteration
            tensors[2].count = 8192 + iteration
            tensors[6].rows = 8194 + iteration
            for tensor in tensors:
                tensor.pointer += 64
                tensor.calls.clear()
            stream.npu_stream += 1
            result = harness.request(*tensors, stream)
            self.assertEqual(tuple(result["pointers"]), tuple(t.pointer for t in tensors))
            self.assertEqual(tuple(result["bytes"]), ((16384 + iteration) * 8, 16384 + iteration,
                (8192 + iteration) * 4, (8192 + iteration) * 4, 4,
                (8192 + iteration) * 4, (8194 + iteration) * 656))
            self.assertEqual(result["stream"], stream.npu_stream)
            self.assertEqual(sum(t.calls.count("pointer") for t in tensors), 7)
            self.assertEqual(sum(t.calls.count("count") for t in tensors), 2)
            self.assertEqual(sum(t.calls.count("rows") for t in tensors), 1)

    def test_current_stream_callable_keeps_resolving_current_handle(self):
        harness = self.native.Harness()
        tensors = [Tensor(0x1000 + i * 0x100) for i in range(7)]
        state = {"stream": SimpleNamespace(npu_stream=42), "calls": 0}
        def current_stream():
            state["calls"] += 1
            return state["stream"]
        with patch.dict(sys.modules, {"torch": SimpleNamespace(npu=SimpleNamespace(current_stream=current_stream))}):
            for handle in (42, 87, 123):
                state["stream"] = SimpleNamespace(npu_stream=handle)
                self.assertEqual(harness.request(*tensors, None)["stream"], handle)
            self.assertEqual(state["calls"], 3)
            self.assertEqual(harness.request(*tensors, SimpleNamespace(npu_stream=11))["stream"], 11)
            self.assertEqual(state["calls"], 3)

    def test_checked_path_preserves_layout_and_attribute_errors(self):
        tensor = SimpleNamespace(data_ptr=lambda: 123, numel=lambda: 12, element_size=lambda: 4,
                                 is_contiguous=lambda: True, device="npu:0", shape=(3, 4))
        self.assertEqual(tuple(self.native.checked(tensor)), (123, 48, 2, 3))
        tensor.is_contiguous = lambda: False
        with self.assertRaises(ValueError):
            self.native.checked(tensor)
        with self.assertRaises(TypeError):
            self.native.checked(object())


if __name__ == "__main__":
    unittest.main()
