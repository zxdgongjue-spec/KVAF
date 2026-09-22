#!/usr/bin/env python3
"""Hardware-independent checks for the AIV output allocation switch."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location(
    "npu_perf_allocator_test", Path(__file__).with_name("test_npu_runtime_perf.py")
)
perf = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"kvaf": types.SimpleNamespace()}):
    spec.loader.exec_module(perf)


class AllocatorTest(unittest.TestCase):
    def test_default_torch_shape_and_device(self):
        torch = types.SimpleNamespace(empty=Mock(), uint8="uint8")
        output = perf.make_output(torch, 2048, 4096, 2, "torch")
        torch.empty.assert_called_once_with(
            (2048, 4096), dtype="uint8", device="npu:2"
        )
        self.assertIs(output, torch.empty.return_value)

    def test_normal_allocation_bypasses_torch(self):
        torch = types.SimpleNamespace(empty=Mock())
        allocate = Mock()
        with patch.object(perf, "kvaf", types.SimpleNamespace(NormalHbmBuffer=allocate)):
            output = perf.make_output(torch, 2048, 4096, 2, "acl-normal")
        allocate.assert_called_once_with(2048, 4096, 2)
        torch.empty.assert_not_called()
        self.assertIs(output, allocate.return_value)

    def test_no_silent_fallback_without_new_binding(self):
        with self.assertRaisesRegex(RuntimeError, "rebuild"):
            perf.make_output(None, 2048, 4096, 0, "acl-normal")

    def test_unknown_allocator_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "torch or acl-normal"):
            perf.make_output(None, 2048, 4096, 0, "unknown")


if __name__ == "__main__":
    unittest.main()
