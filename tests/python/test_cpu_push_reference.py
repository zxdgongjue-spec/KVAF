"""Keep the production V24 baseline identical to the requested Git commit."""

import hashlib
from pathlib import Path
import unittest


class CpuPushReferenceTests(unittest.TestCase):
    def test_a12bb16_source_blobs(self):
        root = Path(__file__).resolve().parents[2] / "common/include/cpu_push/a12bb16"
        blobs = {
            "abi.h": "f71cf36671024e06c941ed6bb1a33b4a5d8c942b",
            "pipeline.h": "15fdb12ef027f71e3bb5835aef09bb377b78d01f",
        }
        for filename, expected in blobs.items():
            with self.subTest(filename=filename):
                # Git's Windows checkout may convert source newlines to CRLF.
                source = (root / filename).read_bytes().replace(b"\r\n", b"\n")
                header = b"blob " + str(len(source)).encode("ascii") + b"\0"
                self.assertEqual(hashlib.sha1(header + source).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
