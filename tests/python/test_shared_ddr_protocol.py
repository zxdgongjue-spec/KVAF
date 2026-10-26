"""Host-only guards for the two-NPU experiment's IPC and address validation."""
import unittest

from test_npu_shared_ddr import receive, validate_descriptor, validate_table


class SharedDdrProtocolTests(unittest.TestCase):
    def setUp(self):
        self.region = dict(server_instance_id=7, region_id=1, base=4096,
                           length=1024, slot_size=256, candidate_id=0, session_id=1)
        self.table = dict(addresses=[4096, 4352, 4608, 4864], lengths=[256] * 4, valid=[1] * 4)

    def test_same_allocation_requires_distinct_sessions(self):
        validate_descriptor(self.region, dict(self.region, session_id=2))
        with self.assertRaises(AssertionError):
            validate_descriptor(self.region, self.region)

    def test_same_address_does_not_make_a_different_region_valid(self):
        for field in ("server_instance_id", "region_id", "length", "slot_size", "candidate_id", "base"):
            with self.subTest(field=field), self.assertRaises(AssertionError):
                validate_descriptor(self.region, dict(self.region, session_id=2,
                                                     **{field: self.region[field] + 1}))

    def test_rejects_missing_partial_or_wrong_slots(self):
        validate_table(self.region, self.table, 4, 256)
        cases = [dict(self.table, valid=[1, 1, 0, 1]),
                 dict(self.table, lengths=[256, 256, 255, 256]),
                 dict(self.table, addresses=[4096, 4352, 4608, 5120]),
                 dict(self.table, addresses=[4096, 4096, 4608, 4864])]
        for table in cases:
            with self.subTest(table=table), self.assertRaises(AssertionError):
                validate_table(self.region, table, 4, 256)
        with self.assertRaises(AssertionError):
            validate_table(dict(self.region, length=512), self.table, 4, 256)

    def test_control_timeout_failure_and_out_of_order_are_not_success(self):
        class Pipe:
            def __init__(self, message): self.message = message
            def poll(self, timeout): return self.message is not None
            def recv(self): return self.message
        self.assertEqual(receive(Pipe(("READ_OK", 3)), "READ_OK", 1), 3)
        with self.assertRaises(TimeoutError):
            receive(Pipe(None), "READ_OK", 1)
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            receive(Pipe(("ERROR", "worker failed")), "READ_OK", 1)
        with self.assertRaises(RuntimeError):
            receive(Pipe(("WRITE_DONE", 3)), "READ_OK", 1)


if __name__ == "__main__":
    unittest.main()
