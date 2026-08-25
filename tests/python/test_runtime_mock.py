#!/usr/bin/env python3
"""Direct Python API tests for the synchronous KVAF Runtime.

No C++ driver process is involved.  Each test creates ``kvaf.Runtime`` and
uses the same lifecycle that a Python caller will use in its own scenarios.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import kvaf

from backend import TorchBackend, detect_torch_backend, test_payload


class RuntimeMockTests(unittest.TestCase):
    backend = TorchBackend("python", None, "not detected")

    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = detect_torch_backend()
        print(f"Mock input backend={cls.backend.name}: {cls.backend.reason}")

    def setUp(self) -> None:
        self.kvaf = kvaf.Runtime()

    def tearDown(self) -> None:
        if self.kvaf.initialized:
            self.kvaf.shutdown()

    def payload(self, values: list[int]) -> bytes:
        return test_payload(values, self.backend)

    def batch_put(
        self, keys: list[int], blocks: list[bytes], *, layer_id: int = 0
    ) -> None:
        """Submit variable-size blocks through the public packed-input API."""
        packed = bytearray()
        input_offsets: list[int] = []
        lengths: list[int] = []
        for block in blocks:
            input_offsets.append(len(packed))
            lengths.append(len(block))
            packed.extend(block)
        self.kvaf.batch_put(
            keys, packed, input_offsets, lengths, layer_id=layer_id
        )

    def test_put_prepare_get_release_and_slot_reuse(self) -> None:
        self.kvaf.initialize(ddr_bytes=256, slot_size=64)
        self.batch_put(
            [101, 102],
            [self.payload([1, 2, 3]), self.payload([160, 161, 162, 163])],
        )
        table = self.kvaf.prepare_access_table([101, 102])

        view = self.kvaf.access_table_view(table)
        self.assertEqual(view["addresses"][1] - view["addresses"][0], 64)
        self.assertEqual(list(view["lengths"]), [3, 4])
        self.assertEqual(list(view["valid"]), [1, 1])
        first_address = view["addresses"][0]

        # Requests are scattered to caller-owned output rows in reverse order.
        output = bytearray(16)
        self.kvaf.batch_get(
            table,
            indices=[1, 0],
            offsets=[0, 0],
            output_indices=[0, 1],
            output=output,
            output_block_bytes=8,
        )
        self.assertEqual(output, bytes.fromhex("a0a1a2a3000000000102030000000000"))

        # The handle pins its keys until the caller releases it.
        with self.assertRaises(ValueError):
            self.kvaf.batch_free([101])
        self.kvaf.release_access_table(table)
        self.kvaf.batch_free([101, 102])

        # The MetaManager free queue is FIFO. Consume the two never-used slots
        # first, then verify that a freed address becomes available again.
        self.batch_put(
            [103, 104, 105],
            [self.payload([255]), self.payload([238]), self.payload([221])],
        )
        recycled = self.kvaf.prepare_access_table([105])
        self.assertEqual(
            self.kvaf.access_table_view(recycled)["addresses"][0],
            first_address,
        )
        self.kvaf.release_access_table(recycled)

    def test_failed_put_is_aborted_and_key_is_reusable(self) -> None:
        self.kvaf.initialize(ddr_bytes=128, slot_size=64)
        self.kvaf._fail_next_put()
        with self.assertRaises(RuntimeError):
            self.batch_put([7], [self.payload([170])])

        # Runtime must invoke AbortPut, otherwise this would be kAlreadyExists.
        self.batch_put([7], [self.payload([187])])
        table = self.kvaf.prepare_access_table([7])
        output = bytearray(4)
        self.kvaf.batch_get(table, [0], [0], [0], output, output_block_bytes=4)
        self.assertEqual(output, bytes.fromhex("bb000000"))
        self.kvaf.release_access_table(table)

    def test_initial_slot_order_can_be_shuffled_deterministically(self) -> None:
        slot_size = 64
        slot_count = 8
        self.kvaf.initialize(
            ddr_bytes=slot_count * slot_size,
            slot_size=slot_size,
            slot_allocation_order="shuffled",
            slot_shuffle_seed=1234,
        )
        self.assertEqual(
            self.kvaf.remote_region["slot_allocation_order"], "shuffled"
        )
        self.assertEqual(self.kvaf.remote_region["slot_shuffle_seed"], 1234)

        self.batch_put(
            list(range(slot_count)),
            [self.payload([index]) for index in range(slot_count)],
        )
        table = self.kvaf.prepare_access_table(list(range(slot_count)))
        try:
            addresses = list(self.kvaf.access_table_view(table)["addresses"])
            first_address = min(addresses)
            self.assertEqual(
                sorted(addresses),
                [first_address + slot_size * index for index in range(slot_count)],
            )
            self.assertNotEqual(
                addresses,
                [first_address + slot_size * index for index in range(slot_count)],
            )
        finally:
            self.kvaf.release_access_table(table)

        with self.assertRaises(ValueError):
            invalid_runtime = kvaf.Runtime()
            invalid_runtime.initialize(
                ddr_bytes=slot_count * slot_size,
                slot_size=slot_size,
                slot_allocation_order="random",
            )

    def test_layer_partition_scopes_keys_and_addresses(self) -> None:
        """One Runtime keeps each layer in a separate contiguous DDR range."""
        slot_size = 64
        slots_per_layer = 4
        layer_count = 78
        self.kvaf.initialize(
            ddr_bytes=slot_size * slots_per_layer * layer_count,
            slot_size=slot_size,
            layer_count=layer_count,
        )
        self.assertEqual(self.kvaf.remote_region["layer_count"], layer_count)
        self.assertEqual(
            self.kvaf.remote_region["slots_per_layer"], slots_per_layer
        )

        # Local key 7 is valid independently in each layer. A layer-local
        # BatchPut must allocate from that layer's fixed DDR subregion.
        self.batch_put(
            layer_id=0,
            keys=[7, 8],
            blocks=[self.payload([0x10]), self.payload([0x11])],
        )
        self.batch_put(
            layer_id=1,
            keys=[7, 8],
            blocks=[self.payload([0x20]), self.payload([0x21])],
        )
        self.batch_put(
            layer_id=77,
            keys=[9],
            blocks=[self.payload([0x30])],
        )

        layer0 = self.kvaf.prepare_access_table([7, 8], layer_id=0)
        layer1 = self.kvaf.prepare_access_table([7, 8], layer_id=1)
        layer77 = self.kvaf.prepare_access_table([9], layer_id=77)
        try:
            view0 = self.kvaf.access_table_view(layer0)
            view1 = self.kvaf.access_table_view(layer1)
            view77 = self.kvaf.access_table_view(layer77)
            self.assertEqual(
                view0["addresses"][1] - view0["addresses"][0], slot_size
            )
            self.assertEqual(
                view1["addresses"][1] - view1["addresses"][0], slot_size
            )
            self.assertEqual(
                view1["addresses"][0] - view0["addresses"][0],
                slots_per_layer * slot_size,
            )
            self.assertEqual(
                view77["addresses"][0] - view1["addresses"][0],
                (77 - 1) * slots_per_layer * slot_size,
            )

            output = bytearray(4)
            self.kvaf.batch_get(
                layer1,
                indices=[0],
                offsets=[0],
                output_indices=[0],
                output=output,
                output_block_bytes=4,
            )
            self.assertEqual(output, bytes.fromhex("20000000"))
        finally:
            self.kvaf.release_access_table(layer0)
            self.kvaf.release_access_table(layer1)
            self.kvaf.release_access_table(layer77)

        # The same local key is not visible through a different layer id.
        with self.assertRaises(ValueError):
            self.kvaf.prepare_access_table([7], layer_id=99)
        with self.assertRaises(KeyError):
            self.kvaf.prepare_access_table([9], layer_id=0)

        self.kvaf.batch_free([7, 8], layer_id=0)
        self.kvaf.batch_free([7, 8], layer_id=1)
        self.kvaf.batch_free([9], layer_id=77)

    def test_layer_count_must_divide_slot_count(self) -> None:
        with self.assertRaises(ValueError):
            self.kvaf.initialize(ddr_bytes=4 * 64, slot_size=64, layer_count=3)

    def test_get_failure_preserves_lifecycle_and_invalid_index_is_rejected(self) -> None:
        self.kvaf.initialize(ddr_bytes=128, slot_size=64)
        self.batch_put([9], [self.payload([10, 11])])
        table = self.kvaf.prepare_access_table([9])
        self.kvaf._fail_next_get()
        with self.assertRaises(RuntimeError):
            self.kvaf.batch_get(table, [0], [0], [0], bytearray(4), output_block_bytes=4)
        with self.assertRaises(ValueError):
            self.kvaf.batch_get(table, [1], [0], [0], bytearray(4), output_block_bytes=4)
        # A data-plane failure does not leak the stable-read lock.
        self.kvaf.release_access_table(table)
        self.kvaf.batch_free([9])

    def test_validation_and_shutdown(self) -> None:
        with self.assertRaises(ValueError):
            self.kvaf.initialize(ddr_bytes=10, slot_size=4)
        self.kvaf.initialize(ddr_bytes=64, slot_size=32)
        with self.assertRaises(KeyError):
            self.kvaf.prepare_access_table([999])
        self.batch_put([1], [self.payload([1])])
        table = self.kvaf.prepare_access_table([1])
        self.kvaf.shutdown()
        with self.assertRaises(RuntimeError):
            self.kvaf.batch_get(table, [0], [0], [0], bytearray(4), output_block_bytes=4)

    def test_get_uses_index_and_656_byte_row_offset(self) -> None:
        entry_bytes = 128 * 656
        self.kvaf.initialize(ddr_bytes=entry_bytes, slot_size=entry_bytes)
        block = bytearray(entry_bytes)
        block[:656] = bytes([0x11]) * 656
        block[656:1312] = bytes([0x22]) * 656
        self.batch_put([17], [block])
        table = self.kvaf.prepare_access_table([17])
        output = bytearray(656)
        self.kvaf.batch_get(
            table, indices=[0], offsets=[656], output_indices=[0],
            output=output, output_block_bytes=656,
        )
        self.assertEqual(output, bytes([0x22]) * 656)
        with self.assertRaises(ValueError):
            self.kvaf.batch_get(
                table, indices=[0], offsets=[entry_bytes], output_indices=[0],
                output=bytearray(656), output_block_bytes=656,
            )
        self.kvaf.release_access_table(table)

    def test_put_rejects_out_of_range_variable_block(self) -> None:
        self.kvaf.initialize(ddr_bytes=64, slot_size=32)
        with self.assertRaises(ValueError):
            self.kvaf.batch_put(
                keys=[1],
                input=bytearray([1, 2, 3]),
                input_offsets=[2],
                lengths=[2],
            )


if __name__ == "__main__":
    unittest.main()
