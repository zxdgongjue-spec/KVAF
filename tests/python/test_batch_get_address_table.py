#!/usr/bin/env python3
"""Host-only checks for address-table preparation and measurement boundaries."""

import ast
import contextlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_npu_batch_get_perf as perf


class Tensor:
    """Only the small, one-dimensional Tensor surface used by the helpers."""

    def __init__(self, values, dtype="int64"):
        self.values = list(values)
        self.dtype = dtype

    def __getitem__(self, key):
        if not isinstance(key, slice):
            return self.values[key]
        return Tensor(self.values[key], self.dtype)

    def to(self, dtype):
        return Tensor(self.values, dtype)

    def index_select(self, dim, lookup):
        if dim != 0 or lookup.dtype != "int64":
            raise AssertionError("lookup must use int64 on dimension zero")
        if any(index < 0 or index >= len(self.values) for index in lookup.values):
            raise IndexError("invalid lookup index")
        return Tensor([self.values[index] for index in lookup.values], self.dtype)

    def __add__(self, other):
        if len(self.values) != len(other.values):
            raise AssertionError("mismatched address and offset lengths")
        return Tensor([left + right for left, right in zip(self.values, other.values)])


TORCH = SimpleNamespace(
    int64="int64", equal=lambda left, right: left.values == right.values
)


class AddressTableTests(unittest.TestCase):
    def setUp(self):
        self.addresses = Tensor([2**48 + 16, 2**48 + 256, 2**48 + 4096])
        self.valid = Tensor([1, 0, 1], "uint8")
        # The suffix must never be looked up, even though it is out of range.
        self.indices = Tensor([2, 0, 2, 1, 999, -1], "int32")
        self.compact = Tensor([0, 1, 2, 3, 0, 0], "int32")
        self.offsets = Tensor([4, 8, 12, 16, 99999, -1], "int32")

    def prepare(self, mode, count=4):
        return perf.prepare_address_table(
            TORCH, self.addresses, self.valid, self.indices,
            count, self.compact, mode,
        )

    def validate(self, addresses, valid, indices, count=4):
        perf.validate_resolved_addresses(
            TORCH, self.addresses, self.valid, self.indices,
            addresses, valid, indices, self.offsets, count,
        )

    def test_original_preserves_objects_without_lookup(self):
        sentinels = [object() for _ in range(4)]
        result = perf.prepare_address_table(
            object(), *sentinels[:3], 4, sentinels[3], "original"
        )
        for actual, original in zip(result[:3], sentinels):
            self.assertIs(actual, original)
        self.assertEqual(result[3], ())

    def test_prewarm_retains_metadata_but_submits_original_table(self):
        addresses, valid, indices, keepalive = self.prepare("prewarm")
        self.assertIs(addresses, self.addresses)
        self.assertIs(valid, self.valid)
        self.assertIs(indices, self.indices)
        self.assertEqual(keepalive[0].values, [2, 0, 2, 1])
        self.assertEqual(keepalive[1].values, [
            self.addresses.values[index] for index in [2, 0, 2, 1]
        ])
        self.assertEqual(keepalive[2].values, [1, 1, 1, 0])
        self.validate(addresses, valid, indices)

    def test_compact_preserves_repeated_reads_valid_bits_and_64_bit_addresses(self):
        addresses, valid, indices, keepalive = self.prepare("compact")
        self.assertIs(indices, self.compact)
        self.assertIs(addresses, keepalive[1])
        self.assertIs(valid, keepalive[2])
        self.assertEqual(addresses.dtype, "int64")
        self.assertEqual(addresses.values, [
            2**48 + 4096, 2**48 + 16, 2**48 + 4096, 2**48 + 256,
        ])
        self.assertEqual(valid.dtype, "uint8")
        self.assertEqual(valid.values, [1, 1, 1, 0])
        self.assertEqual(self.indices.values, [2, 0, 2, 1, 999, -1])
        self.validate(addresses, valid, indices)

    def test_zero_active_retains_nonempty_original_table_in_every_mode(self):
        for mode in ("original", "prewarm", "compact"):
            with self.subTest(mode=mode):
                addresses, valid, indices, keepalive = self.prepare(mode, 0)
                self.assertIs(addresses, self.addresses)
                self.assertIs(valid, self.valid)
                self.assertIs(indices, self.indices)
                self.assertEqual(keepalive, ())
                self.validate(addresses, valid, indices, 0)

    def test_preflight_detects_wrong_block_address(self):
        addresses, valid, indices, _ = self.prepare("compact")
        addresses.values[2] += 4096
        with self.assertRaisesRegex(AssertionError, "READ address or valid bit"):
            self.validate(addresses, valid, indices)

    def test_preflight_detects_changed_valid_bit(self):
        addresses, valid, indices, _ = self.prepare("compact")
        valid.values[3] = 1
        with self.assertRaisesRegex(AssertionError, "READ address or valid bit"):
            self.validate(addresses, valid, indices)


class MeasurementTests(unittest.TestCase):
    def test_actual_measurement_loop_sync_boundaries_and_accounting(self):
        source = Path(perf.__file__).read_text()
        loops = [
            node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.For)
            and ast.unparse(node.iter) == "range(args.iterations)"
        ]
        self.assertEqual(len(loops), 1, "find the production measurement loop")
        code = compile(ast.Module(body=loops, type_ignores=[]), str(perf.__file__), "exec")

        for scope in ("batch-get", "resolve-and-batch-get"):
            with self.subTest(scope=scope):
                events = []
                clock = SimpleNamespace(now=0, pending=0)
                plan = (object(), object(), object())
                batch_args = (object(), object())
                retained_metadata = (object(),)

                def create_plan():
                    events.append("plan")
                    clock.now += 11
                    clock.pending += 7  # Request generation must stay excluded.
                    return plan

                def prepare(*actual):
                    self.assertEqual(actual, plan)
                    events.append("resolve")
                    clock.now += 2  # Host-side preparation.
                    clock.pending += 5  # Queued device address lookup.
                    return batch_args, retained_metadata

                def synchronize():
                    events.append("sync")
                    clock.now += clock.pending
                    clock.pending = 0

                def now():
                    events.append("time")
                    return clock.now

                def batch_get(*actual):
                    self.assertEqual(actual, batch_args)
                    self.assertIs(namespace["resolved_table"], retained_metadata)
                    events.append("submit")
                    clock.now += 3
                    clock.pending += 13

                namespace = dict(
                    args=SimpleNamespace(iterations=2, timing_scope=scope, kernel_profile=False),
                    create_request_plan=create_plan, prepare_call=prepare,
                    torch=SimpleNamespace(npu=SimpleNamespace(synchronize=synchronize)),
                    runtime=SimpleNamespace(batch_get=batch_get),
                    time=SimpleNamespace(perf_counter=now),
                    submit_samples=[], synchronize_samples=[], batch_get_samples=[],
                    kernel_profile_samples=[],
                )
                exec(code, namespace)
                if scope == "batch-get":
                    order = ["plan", "resolve", "sync", "time", "submit", "time", "sync", "time"]
                    wait_ms, total_ms = 13000, 16000
                else:
                    order = ["plan", "sync", "time", "resolve", "time", "submit", "time", "sync", "time"]
                    wait_ms, total_ms = 18000, 23000
                self.assertEqual(events, order * 2)
                self.assertEqual(namespace["submit_samples"], [3000, 3000])
                self.assertEqual(namespace["synchronize_samples"], [wait_ms] * 2)
                self.assertEqual(namespace["batch_get_samples"], [total_ms] * 2)
                self.assertEqual(clock.pending, 0)

    def test_cli_choices_defaults_and_result_labels(self):
        with patch.dict(os.environ, {}, clear=True):
            default = perf.parse_args([])
            self.assertEqual((default.address_table_mode, default.timing_scope),
                             ("original", "batch-get"))
            for mode in ("original", "prewarm", "compact"):
                for scope in ("batch-get", "resolve-and-batch-get"):
                    with self.subTest(mode=mode, scope=scope):
                        args = perf.parse_args([
                            "--address-table-mode", mode, "--timing-scope", scope,
                        ])
                        self.assertIsNone(perf.validate_args(args))
                        self.assertEqual((args.address_table_mode, args.timing_scope), (mode, scope))
                        output = io.StringIO()
                        with contextlib.redirect_stdout(output):
                            perf.print_samples(
                                "batch_get", [1.0], 0, 8, 4, 16,
                                address_table_mode=args.address_table_mode,
                                timing_scope=args.timing_scope,
                            )
                        fields = dict(field.split("=", 1) for field in output.getvalue().split()[1:])
                        self.assertEqual(fields["address_table_mode"], mode)
                        self.assertEqual(fields["timing_scope"], scope)
            for option in ("--address-table-mode", "--timing-scope"):
                with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        perf.parse_args([option, "invalid"])


if __name__ == "__main__":
    unittest.main()
