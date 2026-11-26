"""Call the production constructor registration using a device-free probe."""
import unittest

from _kvaf_constructor_test import NpuRuntime


class ConstructorTests(unittest.TestCase):
    def test_defaults_and_layout(self):
        self.assertEqual(NpuRuntime().values,
                         (None, None, 0, "urma", None, None, None, None, [], "", None, None))
        self.assertEqual(NpuRuntime(656, 128).values[:4], (656, 128, 0, "urma"))

    def test_direction_policy_and_resources_are_independent(self):
        put, get, put_budget, get_budget = object(), object(), object(), object()
        runtime = NpuRuntime(656, 2, device_id=7, transport_mode="urma+ubmem",
                             default_put=put, default_get=get,
                             put_resources=put_budget, get_resources=get_budget,
                             npu_eids=["eid"], peer_ub_device="udma0")
        self.assertEqual(runtime.values[:10],
                         (656, 2, 7, "urma+ubmem", put, get, put_budget, get_budget, ["eid"], "udma0"))

    def test_rejects_retired_constructor_and_positional_policy(self):
        with self.assertRaises(TypeError):
            NpuRuntime("get.o", 656, 2)
        with self.assertRaises(TypeError):
            NpuRuntime(656, 2, 7)
        with self.assertRaises(TypeError):
            NpuRuntime(656, 2, aiv_batch_get_mode="gather_read")


if __name__ == "__main__":
    unittest.main()
