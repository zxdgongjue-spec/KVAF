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

    def batch_put(self, keys: list[str], blocks: list[bytes]) -> None:
        """Submit variable-size blocks through the public packed-input API."""
        packed = bytearray()
        input_offsets: list[int] = []
        lengths: list[int] = []
        for block in blocks:
            input_offsets.append(len(packed))
            lengths.append(len(block))
            packed.extend(block)
        self.kvaf.batch_put(keys, packed, input_offsets, lengths)

    def test_put_prepare_get_release_and_slot_reuse(self) -> None:
        self.kvaf.initialize(
            ddr_bytes=768,
            short_slot_size=64,
            long_slot_size=128,
            slot_count_per_class=4,
        )
        self.batch_put(
            ["key-101", "key-102"],
            [self.payload([1, 2, 3]), self.payload([160, 161, 162, 163])],
        )
        table = self.kvaf.prepare_access_table(["key-101", "key-102"])

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
            self.kvaf.batch_free(["key-101"])
        self.kvaf.release_access_table(table)
        self.kvaf.batch_free(["key-101", "key-102"])

        # The MetaManager free queue is FIFO. Consume the two never-used slots
        # first, then verify that a freed address becomes available again.
        self.batch_put(
            ["key-103", "key-104", "key-105"],
            [self.payload([255]), self.payload([238]), self.payload([221])],
        )
        recycled = self.kvaf.prepare_access_table(["key-105"])
        self.assertEqual(
            self.kvaf.access_table_view(recycled)["addresses"][0],
            first_address,
        )
        self.kvaf.release_access_table(recycled)

    def test_failed_put_is_aborted_and_key_is_reusable(self) -> None:
        self.kvaf.initialize(
            ddr_bytes=384,
            short_slot_size=64,
            long_slot_size=128,
            slot_count_per_class=2,
        )
        self.kvaf._fail_next_put()
        with self.assertRaises(RuntimeError):
            self.batch_put(["key-7"], [self.payload([170])])

        # Runtime must invoke AbortPut, otherwise this would be kAlreadyExists.
        self.batch_put(["key-7"], [self.payload([187])])
        table = self.kvaf.prepare_access_table(["key-7"])
        output = bytearray(4)
        self.kvaf.batch_get(table, [0], [0], output, output_block_bytes=4)
        self.assertEqual(output, bytes.fromhex("bb000000"))
        self.kvaf.release_access_table(table)

    def test_get_failure_preserves_lifecycle_and_invalid_index_is_rejected(self) -> None:
        self.kvaf.initialize(
            ddr_bytes=384,
            short_slot_size=64,
            long_slot_size=128,
            slot_count_per_class=2,
        )
        self.batch_put(["key-9"], [self.payload([10, 11])])
        table = self.kvaf.prepare_access_table(["key-9"])
        self.kvaf._fail_next_get()
        with self.assertRaises(RuntimeError):
            self.kvaf.batch_get(table, [0], [0], bytearray(4), output_block_bytes=4)
        with self.assertRaises(ValueError):
            self.kvaf.batch_get(table, [1], [0], bytearray(4), output_block_bytes=4)
        # A data-plane failure does not leak the stable-read lock.
        self.kvaf.release_access_table(table)
        self.kvaf.batch_free(["key-9"])

    def test_validation_and_shutdown(self) -> None:
        with self.assertRaises(ValueError):
            self.kvaf.initialize(
                ddr_bytes=256,
                short_slot_size=64,
                long_slot_size=128,
                slot_count_per_class=1,
            )
        self.kvaf.initialize(
            ddr_bytes=384,
            short_slot_size=64,
            long_slot_size=128,
            slot_count_per_class=2,
        )
        # KeyId is a string at the Python boundary; integer keys must not be
        # silently converted because that can hide caller-side key bugs.
        with self.assertRaises(TypeError):
            self.kvaf.batch_put(
                keys=[1],
                input=bytearray([1]),
                input_offsets=[0],
                lengths=[1],
            )
        with self.assertRaises(TypeError):
            self.kvaf.prepare_access_table([1])
        with self.assertRaises(KeyError):
            self.kvaf.prepare_access_table(["key-999"])
        self.batch_put(["key-1"], [self.payload([1])])
        table = self.kvaf.prepare_access_table(["key-1"])
        self.kvaf.shutdown()
        with self.assertRaises(RuntimeError):
            self.kvaf.batch_get(table, [0], [0], bytearray(4), output_block_bytes=4)

    def test_put_rejects_out_of_range_variable_block(self) -> None:
        self.kvaf.initialize(
            ddr_bytes=192,
            short_slot_size=32,
            long_slot_size=64,
            slot_count_per_class=2,
        )
        with self.assertRaises(ValueError):
            self.kvaf.batch_put(
                keys=["key-1"],
                input=bytearray([1, 2, 3]),
                input_offsets=[2],
                lengths=[2],
            )


if __name__ == "__main__":
    unittest.main()
