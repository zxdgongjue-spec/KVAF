"""Opt-in real AIV/remote CPU integration; never substitutes a mock for hardware."""
from __future__ import annotations

import os
import unittest

import kvaf


@unittest.skipUnless(os.environ.get("KVAF_RUN_NPU_TESTS") == "1", "set KVAF_RUN_NPU_TESTS=1 on the NPU host")
class NpuPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import torch_npu  # noqa: F401
        cls.torch = torch
        cls.device_id = int(os.environ.get("DEVICE_ID", "0"))
        torch.npu.set_device(cls.device_id)

    def test_two_regions_shared_core_layers_readback_and_destroy(self):
        torch = self.torch
        runtime = kvaf.NpuRuntime(
            os.environ["KVAF_BATCH_GET_KERNEL"], batch_get_token_bytes=656,
            batch_get_tokens_per_block=128, device_id=self.device_id,
            batch_put_kernel_object_path=os.environ["KVAF_BATCH_PUT_KERNEL"],
        )
        runtime.link_to(os.environ["KVAF_CPU_SERVICE_ADDR"])
        runtime.initialize_pools()
        stream = torch.npu.current_stream()
        device = f"npu:{self.device_id}"
        pools = []
        tables = []
        prepared = []
        graph = None
        try:
            for token, layer_id in [(1024, 1), (656, 5)]:
                pool = runtime.create_pool(f"token_{token}", token * 128, 4, token, 128)
                layer = runtime.bind_layer(layer_id, pool)
                payload = torch.arange(token * 128, dtype=torch.int64).remainder(251).to(torch.uint8)
                source = payload.to(device)
                runtime.batch_put([71], source, [0], [token * 128], stream=stream, layer=layer)
                table = runtime.prepare_access_table([71], layer=layer)
                tables.append(table)
                device_table = runtime.access_table_view(table, device="npu")
                for offset in [0, 127 * token, token]:
                    indices = torch.tensor([0], dtype=torch.int32, device=device)
                    offsets = torch.tensor([offset], dtype=torch.int32, device=device)
                    count = torch.tensor([1], dtype=torch.int32, device=device)
                    rows = torch.tensor([0], dtype=torch.int32, device=device)
                    output = torch.empty((1, token), dtype=torch.uint8, device=device)
                    runtime.batch_get(device_table, indices, offsets, count, rows, output,
                                      stream=stream, synchronize=True)
                    self.assertTrue(torch.equal(output.cpu().flatten(), payload[offset:offset+token]))
                prepared.append((device_table, indices, offsets, count, rows, output,
                                 payload[token:2*token]))
                pools.append((pool, layer))
            self.assertNotEqual(runtime.pool_info(pools[0][0])["base"], runtime.pool_info(pools[1][0])["base"])
            # Warm both output registrations on the capture stream first.
            capture_stream = torch.npu.Stream(device=self.device_id)
            stream.synchronize()
            for table, indices, offsets, count, rows, output, _ in prepared:
                runtime.batch_get(table, indices, offsets, count, rows, output,
                                  stream=capture_stream, synchronize=True)
            graph = kvaf.AclGraph()
            graph.capture_begin(capture_stream)
            try:
                for table, indices, offsets, count, rows, output, _ in prepared:
                    runtime.batch_get(table, indices, offsets, count, rows, output,
                                      stream=capture_stream, synchronize=False)
            finally:
                graph.capture_end()
            for _ in range(3):
                with torch.npu.stream(capture_stream):
                    for *_, output, expected in prepared:
                        output.fill_(165)  # Detect no-op replay or stale warmed output.
                graph.replay(capture_stream)
                graph.synchronize(capture_stream)
                for *_, output, expected in prepared:
                    self.assertTrue(torch.equal(output.cpu().flatten(), expected))
            graph.reset()
            graph = None
            for table in tables:
                runtime.release_access_table(table)
            tables.clear()
            first_pool, first_layer = pools[0]
            runtime.batch_free([71], layer=first_layer)
            runtime.unbind_layer(first_layer)
            runtime.destroy_pool(first_pool)
            second_pool, second_layer = pools[1]
            self.assertEqual(runtime.pool_info(second_pool)["committed_slots"], 1)
            runtime.batch_free([71], layer=second_layer)
            runtime.unbind_layer(second_layer)
            runtime.destroy_pool(second_pool)
        finally:
            stream.synchronize()
            if graph is not None:
                graph.synchronize(capture_stream)
                graph.reset()
            for table in tables:
                runtime.release_access_table(table)
            runtime.shutdown()


if __name__ == "__main__":
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: real NPU and remote CPU service are required")
        raise SystemExit(77)
    unittest.main(verbosity=2)
