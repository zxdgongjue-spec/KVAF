"""Real native binding tests for one Core, independent pool regions and layers."""
from __future__ import annotations

import random
import unittest

import kvaf


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.runtime = kvaf.Runtime()
        self.runtime.initialize_pools()
        self.tables = []

    def tearDown(self):
        for runtime, table in self.tables:
            try:
                runtime.release_access_table(table)
            except (ValueError, KeyError):
                pass
        if self.runtime.initialized:
            self.runtime.shutdown()

    def pool(self, name="a", count=8, token=8, blocks=2, padding=0):
        return self.runtime.create_pool(name, token * blocks + padding, count, token, blocks)

    def table(self, keys, layer):
        table = self.runtime.prepare_access_table(keys, layer=layer)
        self.tables.append((self.runtime, table))
        return table

    def put(self, layer, keys, payloads):
        source = b"".join(payloads)
        lengths = [len(p) for p in payloads]
        offsets, offset = [], 0
        for size in lengths:
            offsets.append(offset)
            offset += size
        self.runtime.batch_put(keys, source, offsets, lengths, layer=layer)

    def get(self, table, token, offsets, indices=None, rows=None, count=None):
        n = len(offsets)
        output = bytearray(n * token)
        self.runtime.batch_get(table, list(range(n)) if indices is None else indices,
                               offsets, n if count is None else count,
                               list(range(n)) if rows is None else rows, output, token)
        return bytes(output)

    def test_actual_1024_656_shapes_and_scatter(self):
        for token, name, layer_id in [(1024, "a", 1), (656, "b", 5)]:
            pool = self.pool(name, count=4, token=token, blocks=128)
            layer = self.runtime.bind_layer(layer_id, pool)
            blocks = [bytes((seed + i * 7) % 256 for i in range(128 * token)) for seed in (19, 57)]
            self.put(layer, [2**63 + 19, 4], blocks)
            table = self.table([2**63 + 19, 4], layer)
            view = self.runtime.access_table_view(table)
            self.assertEqual(view["token_bytes"], token)
            self.assertEqual(view["addresses"][1] - view["addresses"][0], token * 128)
            result = self.get(table, token, [127 * token, token], rows=[1, 0])
            self.assertEqual(result, blocks[1][token:2*token] + blocks[0][127*token:])
            self.assertEqual(self.runtime.pool_info(pool)["committed_slots"], 2)

    def test_same_key_layers_namespaces_and_regions(self):
        a, b = self.pool("a"), self.pool("b")
        layers = [self.runtime.bind_layer(1, a, namespace_id=0),
                  self.runtime.bind_layer(2, a, namespace_id=0),
                  self.runtime.bind_layer(1, a, namespace_id=1),
                  self.runtime.bind_layer(1, b, namespace_id=2)]
        addresses = set()
        for i, layer in enumerate(layers):
            payload = bytes([i+1]) * 16
            self.put(layer, [7], [payload])
            table = self.table([7], layer)
            addresses.add(self.runtime.access_table_view(table)["addresses"][0])
            self.assertEqual(self.get(table, 8, [0]), payload[:8])
        self.assertEqual(len(addresses), 4)
        self.assertNotEqual(self.runtime.pool_info(a)["base"], self.runtime.pool_info(b)["base"])

    def test_capacity_isolation_and_reuse(self):
        a, b = self.pool("a", 1), self.pool("b", 1)
        l1, l2 = self.runtime.bind_layer(1, a), self.runtime.bind_layer(2, b)
        self.put(l1, [9], [b"a" * 16])
        with self.assertRaises(RuntimeError):
            self.put(l1, [10], [b"b" * 16])
        self.put(l2, [9], [b"c" * 16])
        table = self.table([9], l1)
        address = self.runtime.access_table_view(table)["addresses"][0]
        self.runtime.release_access_table(table)
        self.runtime.batch_free([9], layer=l1)
        self.put(l1, [10], [b"z" * 16])
        replacement = self.table([10], l1)
        self.assertEqual(self.runtime.access_table_view(replacement)["addresses"][0], address)

    def test_pin_and_batch_atomicity(self):
        pool = self.pool()
        layer = self.runtime.bind_layer(1, pool)
        self.put(layer, [1, 2], [b"a" * 16, b"b" * 16])
        table = self.table([1, 1], layer)
        with self.assertRaises(ValueError):
            self.runtime.batch_free([2, 1], layer=layer)
        with self.assertRaises(KeyError):
            self.runtime.prepare_access_table([2, 77], layer=layer)
        self.runtime.batch_free([2], layer=layer)  # failed prepare did not leave a pin
        with self.assertRaises(ValueError):
            self.runtime.shutdown()
        self.runtime.release_access_table(table)
        self.runtime.batch_free([1], layer=layer)
        self.assertEqual(self.runtime.pool_info(pool)["free_slots"], 8)

    def test_invalid_spec_and_block_input(self):
        for args in [("",16,2,8,2), ("x",0,2,8,2), ("x",16,0,8,2),
                     ("x",15,2,8,2), ("x",16,2,0,2), ("x",2**64-1,2,8,2)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.runtime.create_pool(*args)
        pool = self.pool()
        layer = self.runtime.bind_layer(1, pool)
        with self.assertRaises(ValueError):
            self.pool()
        for keys, source, offsets, lengths in [([1,1],b"x"*32,[0,16],[16,16]),
                                               ([1],b"x"*16,[1],[16]),
                                               ([1],b"x"*15,[0],[15])]:
            with self.subTest(keys=keys, lengths=lengths), self.assertRaises(ValueError):
                self.runtime.batch_put(keys, source, offsets, lengths, layer=layer)
        self.assertEqual(self.runtime.pool_info(pool)["free_slots"], 8)

    def test_stale_foreign_handles_and_independent_destroy(self):
        pool = self.pool()
        layer = self.runtime.bind_layer(1, pool)
        other = kvaf.Runtime()
        other.initialize_pools()
        try:
            with self.assertRaises(KeyError):
                other.bind_layer(1, pool)
            self.put(layer, [1], [b"x"*16])
            table = self.table([1], layer)
            with self.assertRaises(KeyError):
                other.access_table_view(table)
            with self.assertRaises(ValueError):
                self.runtime.unbind_layer(layer)
            self.runtime.release_access_table(table)
            self.runtime.batch_free([1], layer=layer)
            self.runtime.unbind_layer(layer)
            with self.assertRaises(KeyError):
                self.runtime.prepare_access_table([1], layer=layer)
            fresh = self.runtime.bind_layer(1, pool)
            self.assertNotEqual(fresh.id, layer.id)
            with self.assertRaises(ValueError):
                self.runtime.destroy_pool(pool)
            self.runtime.unbind_layer(fresh)
            self.runtime.destroy_pool(pool)
            with self.assertRaises(KeyError):
                self.runtime.pool_info(pool)
            self.assertNotEqual(self.pool().id, pool.id)
        finally:
            other.shutdown()

    def test_padded_slots_and_inactive_suffix(self):
        pool = self.pool(padding=16)
        layer = self.runtime.bind_layer(1, pool)
        self.put(layer, [3,4], [b"a"*16,b"b"*16])
        table = self.table([3,4], layer)
        view = self.runtime.access_table_view(table)
        self.assertEqual(view["addresses"][1]-view["addresses"][0],32)
        self.assertEqual(list(view["lengths"]),[16,16])
        self.assertEqual(self.get(table,8,[8,-999],indices=[0,-999],rows=[0,-999],count=1),b"a"*8+b"\0"*8)
        with self.assertRaises(ValueError):
            self.get(table,4,[0],indices=[0])

    def test_randomized_repeated_free_and_lookup(self):
        rng = random.Random(178)
        pool = self.pool(count=32)
        layers = [self.runtime.bind_layer(i,pool) for i in range(4)]
        live = {}
        for step in range(150):
            layer_index, key = rng.randrange(4), rng.randrange(16)
            layer = layers[layer_index]
            identity = layer_index, key
            if identity in live:
                self.runtime.batch_free([key],layer=layer)
                del live[identity]
            elif len(live)<32:
                payload=bytes([step%256])*16
                self.put(layer,[key],[payload])
                live[identity]=payload
                table=self.table([key],layer)
                self.assertEqual(self.get(table,8,[8]),payload[8:])
                self.runtime.release_access_table(table)
            info=self.runtime.pool_info(pool)
            self.assertEqual(info["committed_slots"],len(live))
            self.assertEqual(info["free_slots"]+info["committed_slots"],32)

    def test_restart_invalidates_handles(self):
        pool = self.pool()
        layer=self.runtime.bind_layer(1,pool)
        self.put(layer,[1],[b"x"*16])
        self.runtime.shutdown()
        self.runtime.initialize_pools()
        with self.assertRaises(KeyError):
            self.runtime.get_layer(1)
        with self.assertRaises(KeyError):
            self.runtime.pool_info(pool)
        with self.assertRaises(KeyError):
            self.runtime.prepare_access_table([1],layer=layer)

    def test_mode_and_missing_layer_are_explicit(self):
        with self.assertRaises(ValueError):
            self.runtime.initialize_pools()
        with self.assertRaises(ValueError):
            self.runtime.initialize(16, 16)
        with self.assertRaises(ValueError):
            _ = self.runtime.remote_region
        with self.assertRaises(KeyError):
            self.runtime.get_layer(999)
        with self.assertRaises(KeyError):
            self.runtime.batch_put([1], b"x"*16, [0], [16])
        with self.assertRaises(KeyError):
            self.runtime.prepare_access_table([1])
        with self.assertRaises(KeyError):
            self.runtime.batch_free([1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
