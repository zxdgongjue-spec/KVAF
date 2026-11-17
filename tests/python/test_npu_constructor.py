"""Exercise the production keyword-only constructor with a device-free probe."""
import unittest
from _kvaf_constructor_test import NpuRuntime


class ConstructorTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(NpuRuntime(656, 128).values,
                         (656, 128, 0, "urma", None, None, None, None, [], ""))

    def test_independent_defaults_and_resource_budgets(self):
        put, get = {"executor": "aiv"}, {"executor": "aicpu", "algorithm": "gather"}
        puts, gets = [(put, {"cores": 2})], [(get, {"cores": 5})]
        runtime = NpuRuntime(batch_get_token_bytes=656, batch_get_tokens_per_block=128,
            device_id=7, transport_mode="urma+ubmem", default_put=put, default_get=get,
            put_resources=puts, get_resources=gets, npu_eids=["eid"], peer_ub_device="udma0")
        self.assertEqual(runtime.values, (656, 128, 7, "urma+ubmem", put, get, puts, gets, ["eid"], "udma0"))

    def test_options_are_keyword_only(self):
        with self.assertRaises(TypeError):
            NpuRuntime(656, 128, 7)

    def test_retired_constructor_options_rejected(self):
        for name in ("aicpu_batch_get_mode", "per_call_execution", "transport",
                     "npu_core_count", "jettys_per_core", "batch_get_max_inflight_total",
                     "batch_get_kernel_object_path", "batch_put_kernel_object_path",
                     "adaptive_ub_device", "adaptive_ub_miss_count", "adaptive_peer_ub_devices",
                     "enabled_put_configs", "enabled_get_configs"):
            with self.subTest(name=name), self.assertRaises(TypeError):
                NpuRuntime(656, 128, **{name: []})


if __name__ == "__main__":
    unittest.main()
