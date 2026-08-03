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

    def test_put_prepare_get_release_and_slot_reuse(self) -> None:
        self.kvaf.initialize(ddr_bytes=256, slot_size=64)
        self.kvaf.batch_put(
            keys=[101, 102],
            blocks=[self.payload([1, 2, 3]), self.payload([160, 161, 162, 163])],
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
        self.kvaf.batch_put(
            keys=[103, 104, 105],
            blocks=[self.payload([255]), self.payload([238]), self.payload([221])],
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
            self.kvaf.batch_put(keys=[7], blocks=[self.payload([170])])

        # Runtime must invoke AbortPut, otherwise this would be kAlreadyExists.
        self.kvaf.batch_put(keys=[7], blocks=[self.payload([187])])
        table = self.kvaf.prepare_access_table([7])
        output = bytearray(4)
        self.kvaf.batch_get(table, [0], [0], output, output_block_bytes=4)
        self.assertEqual(output, bytes.fromhex("bb000000"))
        self.kvaf.release_access_table(table)

    def test_get_failure_preserves_lifecycle_and_invalid_index_is_rejected(self) -> None:
        self.kvaf.initialize(ddr_bytes=128, slot_size=64)
        self.kvaf.batch_put(keys=[9], blocks=[self.payload([10, 11])])
        table = self.kvaf.prepare_access_table([9])
        self.kvaf._fail_next_get()
        with self.assertRaises(RuntimeError):
            self.kvaf.batch_get(table, [0], [0], bytearray(4), output_block_bytes=4)
        with self.assertRaises(ValueError):
            self.kvaf.batch_get(table, [1], [0], bytearray(4), output_block_bytes=4)
        # A data-plane failure does not leak the stable-read lock.
        self.kvaf.release_access_table(table)
        self.kvaf.batch_free([9])

    def test_validation_and_shutdown(self) -> None:
        with self.assertRaises(ValueError):
            self.kvaf.initialize(ddr_bytes=10, slot_size=4)
        self.kvaf.initialize(ddr_bytes=64, slot_size=32)
        with self.assertRaises(KeyError):
            self.kvaf.prepare_access_table([999])
        self.kvaf.batch_put(keys=[1], blocks=[self.payload([1])])
        table = self.kvaf.prepare_access_table([1])
        self.kvaf.shutdown()
        with self.assertRaises(RuntimeError):
            self.kvaf.batch_get(table, [0], [0], bytearray(4), output_block_bytes=4)


if __name__ == "__main__":
    unittest.main()
