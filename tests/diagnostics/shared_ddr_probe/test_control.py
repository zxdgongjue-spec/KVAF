"""Wire interoperability, fragmented replies and data-version checks; no NPU."""
import ctypes
import socket
import struct
import threading
import types
import unittest
import numpy as np
from run_probe import Message, payload, rpc, eid


class ControlTest(unittest.TestCase):
    def test_wire_offsets(self):
        packet = Message(4, base=0x0102030405060708, round=13)
        self.assertEqual(ctypes.sizeof(packet), 408)
        self.assertEqual(struct.unpack_from("<Q", bytes(packet), 16)[0], packet.base)
        self.assertEqual(Message.round.offset, 32)
        self.assertEqual(Message.eid.offset, 72)
        self.assertEqual(Message.error.offset, 216)

    def test_cpp_pattern_vectors(self):
        # Known scalar outputs checked independently of the vectorized generator.
        expected = []
        for index in range(4097):
            x = index ^ 0x9e3779b97f4a7c15
            x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & ((1 << 64) - 1)
            x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & ((1 << 64) - 1)
            expected.append((x ^ (x >> 31)) & 255)
        np.testing.assert_array_equal(payload(4097, 1), expected)
        self.assertFalse(np.array_equal(payload(4097, 1), payload(4097, 257)))

    def test_eid_forms(self):
        self.assertEqual(bytes(eid("00000000000000000000000000000001")), bytes(eid("::1")))
        with self.assertRaises(ValueError):
            eid("0" * 32)

    def exchange(self, status=0, mismatch=False):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        errors = []

        def serve():
            try:
                with listener.accept()[0] as connection:
                    data = bytearray()
                    while len(data) < 408:
                        chunk = connection.recv(408 - len(data))
                        if not chunk:
                            raise RuntimeError("truncated request")
                        data.extend(chunk)
                    request = Message.from_buffer_copy(data)
                    response = Message(request.op, status=status, round=3)
                    response.error = b"local access error"
                    if mismatch:
                        response.version = 2
                    raw = bytes(response)
                    for offset in range(0, len(raw), 7):
                        connection.sendall(raw[offset:offset + 7])
            except Exception as exc:
                errors.append(exc)
            finally:
                listener.close()

        thread = threading.Thread(target=serve)
        thread.start()
        args = types.SimpleNamespace(control=f"127.0.0.1:{listener.getsockname()[1]}", timeout=5)
        try:
            return rpc(args, Message(1))
        finally:
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertFalse(errors)

    def test_fragmented_reply(self):
        self.assertEqual(self.exchange().round, 3)

    def test_remote_failure(self):
        with self.assertRaisesRegex(RuntimeError, "local access error"):
            self.exchange(status=-1)

    def test_protocol_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "protocol mismatch"):
            self.exchange(mismatch=True)


if __name__ == "__main__":
    unittest.main()
