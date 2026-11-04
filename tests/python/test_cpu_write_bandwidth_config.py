"""Wire framing, independent payload checks and bandwidth accounting without NPU."""
import contextlib
import ctypes
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_cpu_urma_write_bandwidth as bench


def scalar_mix(value):
    mask = 2**64 - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def response():
    data = bytearray(bench.RESPONSE_BYTES)
    bench.HEADER.pack_into(data, 0, bench.MAGIC, bench.VERSION, bench.COMMAND | 0x8000, len(data))
    bench.META.pack_into(data, 208, 0x120000000000, 4096, 131072, 4, 4, 65536, 13, 4, 128, 1, 1, 2)
    bench.SAMPLE.pack_into(data, 276, 1000, 100, 900, 4)
    bench.SAMPLE.pack_into(data, 308, 2000, 200, 1800, 4)
    return data


class WriteBandwidthTests(unittest.TestCase):
    def test_layout_and_metrics(self):
        self.assertEqual(bench.HEADER.size + 8 + bench.OPTIONS.size, 72)
        result = bench.decode_response(response(), 2)
        self.assertEqual(result["samples"][0]["bandwidth_GBps"], 4.096)
        summary = bench.summarize(result)
        self.assertEqual(summary["mean_us"], 1.5)
        self.assertEqual(summary["max_observed_GBps"], 4.096)
        self.assertAlmostEqual(summary["effective_GBps"], 8192 / 3000)
        self.assertAlmostEqual(summary["p50_GBps"], 4096 / 1500)

    def test_incomplete_failed_and_invalid_reports_are_rejected(self):
        for field, value in ((276, 0), (284, 5000), (300, 0)):
            data = response()
            struct.pack_into("<Q", data, field, value)
            with self.assertRaises(RuntimeError):
                bench.decode_response(data, 2)
        with self.assertRaises(RuntimeError):
            bench.decode_response(response(), 1)
        data = response()
        struct.pack_into("<i", data, 12, 3)
        data[16:24] = b"CQ error"
        with self.assertRaisesRegex(RuntimeError, "CQ error"):
            bench.decode_response(data, 2)

    def test_request_codec_and_partial_tcp_reads(self):
        args = bench.arguments(["--build", "absent", "--sizes", "4K", "--source-bytes", "128K"])
        class Connection:
            def __init__(self):
                self.data = response()
            def __enter__(self): return self
            def __exit__(self, *ignored): return False
            def settimeout(self, timeout): pass
            def sendall(self, packet): self.packet = packet
            def recv(self, size):
                count = min(size, 37)
                data, self.data = self.data[:count], self.data[count:]
                return data
        connection = Connection()
        with patch.object(bench.socket, "create_connection", return_value=connection):
            result = bench.run_remote(args, 0x1234000, (4096, 65536, 4, "contiguous"), 23, 1, 2)
        self.assertEqual(len(connection.packet), 72)
        self.assertEqual(bench.HEADER.unpack_from(connection.packet), (bench.MAGIC, 6, 15, 72))
        self.assertEqual(struct.unpack_from("<Q", connection.packet, 12)[0], 0x1234000)
        self.assertEqual(bench.OPTIONS.unpack_from(connection.packet, 20),
                         (4096, 131072, 23, 65536, 656, 4, 128, 1, 2, 0))
        self.assertEqual(result["count"], 2)

    def test_full_payload_and_random_fragment_mapping(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy unavailable")
        self.assertEqual(int(bench.mix64(np.array([0], dtype=np.uint64), np)[0]), 0xE220A8397B1DCDAF)
        for mode in ("contiguous", "random-sgl"):
            seed, size, pool, fragment, address = 23, 4096, 131072, 656, 0x120000000000
            payload = bytearray()
            for offset in range(0, size, 8):
                source = offset
                if mode == "random-sgl":
                    source = scalar_mix(seed + offset // fragment) % (pool // fragment) * fragment + offset % fragment
                payload.extend(struct.pack("<Q", scalar_mix(seed + source // 8)))
            def copy(destination, maximum, source, count, kind):
                self.assertEqual(kind, 2)
                self.assertEqual(maximum, count)
                ctypes.memmove(destination, bytes(payload[source-address:source-address+count]), count)
                return 0
            result = dict(payload_bytes=size, hbm_address=address)
            args = SimpleNamespace(source_bytes=pool, fragment_bytes=fragment)
            bench.verify_hbm(result, args, mode, seed, copy)
            payload[-1] ^= 1
            with self.assertRaisesRegex(AssertionError, "4088"):
                bench.verify_hbm(result, args, mode, seed, copy)

    def test_dry_run_is_hardware_free_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as folder:
            sentinel = Path(folder) / "samples.csv"
            sentinel.write_text("keep")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(bench.main(["--build", "absent", "--dry-run", "--output-dir", folder]), 0)
            self.assertEqual(len(json.loads(output.getvalue())["cases"]), 9)
            self.assertEqual(sentinel.read_text(), "keep")

    def test_limits_fail_before_allocating_hbm(self):
        for flags in (("--sizes", "512M"), ("--sizes", "65"), ("--message-sizes", "64G"),
                      ("--queue-counts", "8"), ("--fragment-bytes", "13"),
                      ("--iterations", "0"), ("--source-modes", "fake")):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.arguments(["--build", "absent", *flags])


if __name__ == "__main__":
    unittest.main()
