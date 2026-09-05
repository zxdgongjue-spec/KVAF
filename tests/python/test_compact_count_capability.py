"""Execute the public package's native capability forwarding without CANN."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class CompactCountCapabilityTests(unittest.TestCase):
    def _load(self, *, npu: bool, capability: int | None):
        name = "kvaf_compact_capability_test"
        native = types.ModuleType(f"{name}._kvaf")
        for attribute in (
            "AccessTable", "Runtime", "MockRuntime", "create_mock_runtime",
            "runtime", "meta_manager", "access_operator",
        ):
            setattr(native, attribute, object())
        if npu:
            for attribute in ("NpuRuntime", "AclGraph", "DeviceAccessTable"):
                setattr(native, attribute, object())
        if capability is not None:
            native.AIV_BATCH_GET_COMPACT_COUNT_ABI = capability
        package_path = Path(__file__).resolve().parents[2] / "python/kvaf/__init__.py"
        spec = importlib.util.spec_from_file_location(
            name, package_path, submodule_search_locations=[str(package_path.parent)]
        )
        assert spec is not None and spec.loader is not None
        package = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {name: package, f"{name}._kvaf": native}):
            spec.loader.exec_module(package)
        return package

    def test_cpu_build_does_not_claim_device_support(self):
        self.assertEqual(self._load(npu=False, capability=None).AIV_BATCH_GET_COMPACT_COUNT_ABI, 0)

    def test_old_npu_binary_does_not_claim_compact_count_support(self):
        self.assertEqual(self._load(npu=True, capability=None).AIV_BATCH_GET_COMPACT_COUNT_ABI, 0)

    def test_current_native_capability_is_reexported(self):
        package = self._load(npu=True, capability=1)
        self.assertEqual(package.AIV_BATCH_GET_COMPACT_COUNT_ABI, 1)
        self.assertIn("AIV_BATCH_GET_COMPACT_COUNT_ABI", package.__all__)

    def test_future_capability_is_not_silently_claimed_as_version_one(self):
        self.assertEqual(self._load(npu=True, capability=2).AIV_BATCH_GET_COMPACT_COUNT_ABI, 2)


if __name__ == "__main__":
    unittest.main()
