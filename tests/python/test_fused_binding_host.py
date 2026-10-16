#!/usr/bin/env python3
"""CPU-only actual-binding lifecycle tests; this is not NPU correctness coverage.

Run: python3 tests/python/test_fused_binding_host.py
Requires a C++17 compiler, Python headers and pybind11 headers (from pybind11,
an installed torch, or PYBIND11_INCLUDE_DIR). No CANN, peer or device is used;
torch is not imported or executed by these tests.
"""
from __future__ import annotations

import gc
import importlib
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from types import SimpleNamespace
import unittest
import weakref


ROOT = Path(__file__).resolve().parents[2]


def compile_fixture(destination: Path) -> Path:
    include = os.environ.get("PYBIND11_INCLUDE_DIR")
    if not include:
        try:
            import pybind11
        except ImportError as error:
            torch_spec = importlib.util.find_spec("torch")
            if torch_spec is None or torch_spec.origin is None:
                raise RuntimeError(
                    "host binding tests require pybind11 headers (pybind11, "
                    "torch, or PYBIND11_INCLUDE_DIR)"
                ) from error
            include = str(Path(torch_spec.origin).parent / "include")
        else:
            include = pybind11.get_include()
    if not (Path(include) / "pybind11/pybind11.h").is_file():
        raise RuntimeError(f"pybind11 headers missing from {include}")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not shutil.which(compiler[0]):
        raise RuntimeError("host binding tests require a C++17 compiler")
    output = destination / ("_kvaf_fused_binding_host" + sysconfig.get_config_var("EXT_SUFFIX"))
    command = compiler + ["-std=c++17", "-shared", "-fPIC", "-O0", "-g0"]
    if sys.platform == "darwin":
        command += ["-undefined", "dynamic_lookup"]
    includes = [include, sysconfig.get_path("include"),
                ROOT / "tests/fused_context_stubs", ROOT / "bindings/include",
                ROOT / "common/include", ROOT / "runtime/include",
                ROOT / "meta_manager/include", ROOT / "access_operator/include"]
    command += [f"-I{path}" for path in includes]
    command += [str(ROOT / path) for path in (
        "bindings/src/npu_python_runtime.cc", "bindings/src/acl_python_utils.cc",
        "tests/python/fused_binding_host_fixture.cc")]
    command += ["-o", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output


class Tensor:
    def __init__(self, device="npu:0", shape=(4, 656), contiguous=True):
        self.device, self.shape, self.contiguous = device, shape, contiguous

    def data_ptr(self):
        return 0x12340000

    def numel(self):
        count = 1
        for value in self.shape:
            count *= value
        return count

    def element_size(self):
        return 1

    def is_contiguous(self):
        return self.contiguous


class Stream:
    def __init__(self, pointer=0x34560000, device="npu:0"):
        self.npu_stream, self.device = pointer, device


class FusedBindingHostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="kvaf-fused-binding-")
        try:
            compile_fixture(Path(cls.temporary.name))
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"real binding compilation failed:\n{error.stdout}\n{error.stderr}") from error
        sys.path.insert(0, cls.temporary.name)
        cls.module = importlib.import_module("_kvaf_fused_binding_host")

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(cls.temporary.name)
        cls.temporary.cleanup()

    def setUp(self):
        gc.collect()
        self.module.reset()
        self.module._before_release = None

    def runtime(self, device=0):
        runtime = self.module.NpuRuntime("fixture.o", 656, 128, device_id=device,
                                         npu_core_count=4, jettys_per_core=2)
        runtime.initialize(4096, 256)
        return runtime

    def test_put_workers_are_optional_and_do_not_change_get_lease(self):
        # The trailing argument preserves the previous nine positional args.
        legacy = self.module.NpuRuntime("fixture.o", 656, 128, 0, 30,
                                         "put.o", 2, True, "thp")
        self.assertEqual(legacy.npu_core_count, 30)
        self.assertEqual(legacy.batch_put_npu_core_count, 30)
        legacy.shutdown()

        independent = self.module.NpuRuntime("fixture.o", 656, 128,
                                              npu_core_count=30,
                                              batch_put_npu_core_count=1)
        independent.initialize(4096, 256)
        self.assertEqual(independent.batch_put_npu_core_count, 1)
        context = independent.acquire_fused_batch_get_context(Tensor(), Stream())
        self.assertEqual(context["worker_count"], 30)
        with self.assertRaises(AttributeError):
            independent.batch_put_npu_core_count = 2
        independent.release_fused_batch_get_context(context)
        independent.shutdown()

    def test_cq_diagnostic_snapshot_exports_late_success_without_releasing_lease(self):
        runtime = self.runtime()
        context = runtime.acquire_fused_batch_get_context(Tensor(), Stream())
        before = self.module.stats()
        records = runtime.batch_get_cq_diagnostics()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual((record["core"], record["jetty"], record["last_error"]), (29, 1, 0))
        self.assertEqual(record["reason"], "late_success")
        self.assertEqual(record["raw_cq_word"], 4)
        self.assertEqual((record["observed_owner"], record["expected_owner"]), (1, 1))
        self.assertEqual((record["polls"], record["slow_elapsed_ticks"]), (20003, 1234))
        self.assertTrue(record["late_success"] and record["fast_budget_exceeded"] and record["bypass_probe"])
        self.assertEqual(record["trace_scope"], "most_recent_slow_or_error_wait")
        self.assertEqual(self.module.stats(), before)
        runtime.release_fused_batch_get_context(context)
        runtime.shutdown()

    def test_exports_unsigned_context_and_releases_only_after_fence(self):
        runtime = self.runtime()
        output, stream = Tensor(), Stream()
        output_ref, stream_ref = weakref.ref(output), weakref.ref(stream)
        context = runtime.acquire_fused_batch_get_context(output, stream)
        self.assertEqual(context, dict(
            lease_id=context["lease_id"], device_channel_entity=0xF000000000000001,
            device_queue_state=0x2000, device_core_status=0x3000,
            output_address=output.data_ptr(), output_bytes=2624, worker_count=4,
            jettys_per_core=2, token_bytes=656, tokens_per_block=128))
        del output, stream
        gc.collect()
        self.assertIsNotNone(output_ref())
        self.assertIsNotNone(stream_ref())
        self.module._before_release = lambda: (
            self.assertIsNotNone(output_ref()), self.assertIsNotNone(stream_ref()))
        runtime.release_fused_batch_get_context(context)
        self.assertIsNone(output_ref())
        self.assertIsNone(stream_ref())
        self.assertEqual(self.module.stats()["fences"], 1)
        self.assertEqual(self.module.stats()["deregistrations"], 1)
        self.assertEqual(self.module.stats()["last_stream"], 0x34560000)
        runtime.shutdown()

    def test_active_lease_rejects_shutdown_duplicate_profile_and_wrong_release(self):
        runtime = self.runtime()
        context = runtime.acquire_fused_batch_get_context(Tensor(), Stream())
        with self.assertRaisesRegex(ValueError, "before shutdown"):
            runtime.shutdown()
        with self.assertRaisesRegex(ValueError, "already owns"):
            runtime.acquire_fused_batch_get_context(Tensor(), Stream())
        with self.assertRaisesRegex(ValueError, "active lease"):
            runtime.start_batch_get_kernel_profile()
        for field in context:
            corrupt = dict(context)
            corrupt[field] += 1
            with self.assertRaisesRegex(ValueError, "does not match"):
                runtime.release_fused_batch_get_context(corrupt)
        with self.assertRaisesRegex(ValueError, "does not match"):
            runtime.release_fused_batch_get_context({})
        with self.assertRaisesRegex(ValueError, "stream selected"):
            runtime.release_fused_batch_get_context(context, Stream(0x87650000))
        self.assertEqual(self.module.stats()["shutdowns"], 0)
        self.assertEqual(self.module.stats()["release_calls"], 0)
        runtime.release_fused_batch_get_context(context)
        with self.assertRaisesRegex(ValueError, "no active"):
            runtime.release_fused_batch_get_context(context)
        next_context = runtime.acquire_fused_batch_get_context(Tensor(), Stream())
        self.assertNotEqual(next_context["lease_id"], context["lease_id"])
        with self.assertRaisesRegex(ValueError, "does not match"):
            runtime.release_fused_batch_get_context(context)
        runtime.release_fused_batch_get_context(next_context)
        runtime.shutdown()

    def test_default_stream_is_captured_on_runtime_device_not_resolved_at_release(self):
        runtime = self.runtime(1)
        old_torch = sys.modules.get("torch")
        requested = []
        current = [Stream(device="npu:1")]
        def current_stream(device):
            requested.append(device)
            return current[0]
        sys.modules["torch"] = SimpleNamespace(npu=SimpleNamespace(current_stream=current_stream))
        try:
            context = runtime.acquire_fused_batch_get_context(Tensor("npu:1"))
            current[0] = Stream(0x77770000, "npu:1")
            runtime.release_fused_batch_get_context(context)
            self.assertEqual(requested, [1])
            self.assertEqual(self.module.stats()["last_stream"], 0x34560000)
            self.assertEqual(self.module.stats()["selected_device"], 1)
            runtime.shutdown()
        finally:
            if old_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = old_torch

    def test_invalid_device_layout_stream_and_failed_acquire_do_not_retain(self):
        runtime = self.runtime()
        for output, stream in ((Tensor("cpu"), Stream()), (Tensor("npu:1"), Stream()),
                               (Tensor(shape=(4, 655)), Stream()),
                               (Tensor(contiguous=False), Stream()),
                               (Tensor(), Stream(device="npu:1")), (Tensor(), Stream(0))):
            with self.assertRaises(ValueError):
                runtime.acquire_fused_batch_get_context(output, stream)
        self.assertEqual(self.module.stats()["acquires"], 0)
        output = Tensor()
        reference = weakref.ref(output)
        self.module.fail_acquire(True)
        with self.assertRaisesRegex(RuntimeError, "registration failure"):
            runtime.acquire_fused_batch_get_context(output, Stream())
        del output
        gc.collect()
        self.assertIsNone(reference())
        runtime.shutdown()

    def test_uninitialized_and_foreign_context_cannot_acquire_or_release(self):
        runtime = self.module.NpuRuntime("fixture.o", 656, 128)
        with self.assertRaisesRegex(RuntimeError, "not been initialized"):
            runtime.acquire_fused_batch_get_context(Tensor(), Stream())
        runtime.initialize(4096, 256)
        other = self.runtime()
        context = runtime.acquire_fused_batch_get_context(Tensor(), Stream())
        other_context = other.acquire_fused_batch_get_context(Tensor(), Stream())
        with self.assertRaisesRegex(ValueError, "does not match"):
            other.release_fused_batch_get_context(context)
        self.assertEqual(self.module.stats()["release_calls"], 0)
        runtime.release_fused_batch_get_context(context)
        other.release_fused_batch_get_context(other_context)
        runtime.shutdown()
        other.shutdown()

    def test_device_selection_failure_keeps_lease_recoverable(self):
        runtime = self.runtime()
        context = runtime.acquire_fused_batch_get_context(Tensor(), Stream())
        self.module.fail_set_device(True)
        with self.assertRaisesRegex(RuntimeError, "SetDevice"):
            runtime.release_fused_batch_get_context(context)
        self.assertEqual(self.module.stats()["release_calls"], 0)
        with self.assertRaises(ValueError):
            runtime.shutdown()
        self.module.fail_set_device(False)
        runtime.release_fused_batch_get_context(context)
        runtime.shutdown()

    def test_failed_completion_and_gc_quarantine_references_and_native_owner(self):
        # Isolate intentional process-lifetime retention and assert its warning.
        script = r'''
import gc, weakref
from test_fused_binding_host import Tensor, Stream
import _kvaf_fused_binding_host as fixture
for failed in (False, True):
    fixture.reset()
    runtime = fixture.NpuRuntime("fixture.o", 656, 128)
    runtime.initialize(4096, 256)
    output, stream = Tensor(), Stream()
    output_ref, stream_ref = weakref.ref(output), weakref.ref(stream)
    context = runtime.acquire_fused_batch_get_context(output, stream)
    del output, stream
    if failed:
        fixture.fail_release(True)
        try:
            runtime.release_fused_batch_get_context(context)
        except RuntimeError as error:
            assert "completion failure" in str(error)
        else:
            raise AssertionError("failed completion was swallowed")
        fixture.fail_release(False)
        try:
            runtime.release_fused_batch_get_context(context)
        except RuntimeError as error:
            assert "quarantined" in str(error)
        else:
            raise AssertionError("uncertain DMA quarantine was bypassed")
    try:
        runtime.shutdown()
    except ValueError:
        pass
    else:
        raise AssertionError("active lease shutdown was accepted")
    del runtime
    gc.collect()
    assert output_ref() is not None and stream_ref() is not None
    assert fixture.stats()["deregistrations"] == 0
    assert fixture.stats()["destructions"] == 0
    assert fixture.stats()["shutdowns"] == 0
'''
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join((self.temporary.name, str(Path(__file__).parent)))
        result = subprocess.run([sys.executable, "-c", script], env=env,
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stderr.count("FUSED_PYTHON_LEASE_RETAINED_ON_DESTRUCTION"), 2)
        self.assertIn("teardown_not_safe=1", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
