#!/usr/bin/env python3
"""Python-owned functional tests for the synchronous KVAF Runtime.

Each test describes its own Runtime lifecycle using the driver protocol, so
new scenarios can be added by changing only Python commands and assertions.
"""

import argparse
import pathlib
import subprocess
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from backend import TorchBackend, detect_torch_backend, test_payload


class RuntimeDriver:
    def __init__(self, executable: str):
        self.process = subprocess.Popen(
            [executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def command(self, text: str) -> str:
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()
        result = self.process.stdout.readline().strip()
        if not result:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise AssertionError(f"driver produced no result for {text!r}: {stderr}")
        return result

    def close(self) -> None:
        if self.process.poll() is None:
            self.command("QUIT")
            self.process.wait(timeout=5)
        if self.process.returncode:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise AssertionError(f"driver exited with {self.process.returncode}: {stderr}")
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()
        if self.process.stderr:
            self.process.stderr.close()


class RuntimeMockTests(unittest.TestCase):
    driver_path = ""
    backend = TorchBackend("python", None, "not detected")

    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = detect_torch_backend()
        print(f"Mock input backend={cls.backend.name}: {cls.backend.reason}")

    def setUp(self) -> None:
        self.driver = RuntimeDriver(self.driver_path)

    def tearDown(self) -> None:
        self.driver.close()

    def assert_ok(self, command: str, suffix: str | None = None) -> str:
        result = self.driver.command(command)
        self.assertTrue(result.startswith("OK"), f"{command}: {result}")
        if suffix is not None:
            self.assertTrue(result.endswith(suffix), f"{command}: {result}")
        return result

    def assert_error(self, command: str, code: int) -> None:
        self.assertEqual(self.driver.command(command), f"ERR code={code}")

    def payload_hex(self, values: list[int]) -> str:
        return test_payload(values, self.backend).hex()

    def test_put_prepare_get_release_and_slot_reuse(self) -> None:
        self.assert_ok("INIT 256 64")
        first = self.payload_hex([1, 2, 3])
        second = self.payload_hex([160, 161, 162, 163])
        self.assert_ok(f"PUT 101 {first} 102 {second}")
        self.assert_ok("PREPARE decode 101 102")

        view = self.assert_ok("VIEW decode")
        entries = view.removeprefix("OK VIEW ").split(",")
        self.assertEqual(len(entries), 2)
        first_address, first_length, first_valid = entries[0].split(":")
        second_address, second_length, second_valid = entries[1].split(":")
        self.assertEqual(int(second_address) - int(first_address), 64)
        self.assertEqual((first_length, first_valid), ("3", "1"))
        self.assertEqual((second_length, second_valid), ("4", "1"))

        # Requests are scattered to caller-owned output rows in reverse order.
        result = self.assert_ok("GET decode 8 1:0:1 0:1:1")
        self.assertEqual(
            result,
            "OK GET a0a1a2a3000000000102030000000000",
        )
        # The handle pins its keys until the caller releases it.
        self.assert_error("FREE 101", 6)  # kBusy
        self.assert_ok("RELEASE decode")
        self.assert_ok("FREE 101 102")

        # The MetaManager free queue is FIFO. Consume the two never-used slots
        # first, then verify that a freed address becomes available again.
        self.assert_ok(
            "PUT 103 "
            f"{self.payload_hex([255])} 104 {self.payload_hex([238])} "
            f"105 {self.payload_hex([221])}"
        )
        self.assert_ok("PREPARE recycled 105")
        reused_view = self.assert_ok("VIEW recycled")
        self.assertEqual(int(reused_view.removeprefix("OK VIEW ").split(":")[0]), int(first_address))
        self.assert_ok("RELEASE recycled")

    def test_failed_put_is_aborted_and_key_is_reusable(self) -> None:
        self.assert_ok("INIT 128 64")
        self.assert_ok("FAIL_NEXT_PUT")
        self.assert_error(f"PUT 7 {self.payload_hex([170])}", 9)  # kInternal from the mock data plane
        # Runtime must invoke AbortPut, otherwise this second PUT would be kAlreadyExists.
        self.assert_ok(f"PUT 7 {self.payload_hex([187])}")
        self.assert_ok("PREPARE key7 7")
        self.assertEqual(self.assert_ok("GET key7 4 0:0:1"), "OK GET bb000000")
        self.assert_ok("RELEASE key7")

    def test_get_failure_preserves_lifecycle_and_invalid_index_is_rejected(self) -> None:
        self.assert_ok("INIT 128 64")
        self.assert_ok(f"PUT 9 {self.payload_hex([10, 11])}")
        self.assert_ok("PREPARE key9 9")
        self.assert_ok("FAIL_NEXT_GET")
        self.assert_error("GET key9 4 0:0:1", 9)
        self.assert_error("GET key9 4 1:0:1", 1)
        # A data-plane failure does not leak the stable-read lock.
        self.assert_ok("RELEASE key9")
        self.assert_ok("FREE 9")

    def test_validation_and_shutdown(self) -> None:
        self.assert_error("INIT 10 4", 1)
        self.assert_ok("INIT 64 32")
        self.assert_error("PREPARE absent 999", 4)
        self.assert_ok(f"PUT 1 {self.payload_hex([1])}")
        self.assert_ok("PREPARE one 1")
        self.assert_ok("SHUTDOWN")
        self.assert_error("GET one 4 0:0:1", 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    args = parser.parse_args()
    RuntimeMockTests.driver_path = args.driver
    unittest.main(argv=["test_runtime_mock.py"])


if __name__ == "__main__":
    main()
