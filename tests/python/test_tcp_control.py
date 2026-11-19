"""Real TCP framing and the NPU notification lifecycle without NPU hardware."""
import contextlib
import importlib.util
import io
from pathlib import Path
import pickle
import socket
import sys
import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch

import bench_hbm_read_raw_addr as bench


# The transport itself has no native dependency. Load it directly so this test
# also runs before building the optional kvaf._kvaf extension.
_PATH = Path(__file__).resolve().parents[2] / "python" / "kvaf" / "tcp_control.py"
_SPEC = importlib.util.spec_from_file_location("kvaf.tcp_control", _PATH)
control = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(control)


def tcp_process_peer(args):
    """Spawn target with no NPU dependency; bootstrapping also uses TCP."""
    with control.TcpControlListener(args.control_listen, timeout=5) as listener:
        with listener.accept() as channel:
            message = channel.recv()
            channel.send(("connected", message[1]))
            request = channel.recv()
            channel.send(("ready", request[1]))


class TcpNotifications(unittest.TestCase):
    def pair(self, timeout=1):
        with control.TcpControlListener("127.0.0.1:0", timeout=timeout) as listener:
            client = control.TcpControlChannel.connect(listener.address, timeout=timeout)
            self.addCleanup(client.close)
            server = listener.accept()
            self.addCleanup(server.close)
        return client, server

    def frame(self, message):
        payload = pickle.dumps(message, protocol=4)
        return control._HEADER.pack(control._MAGIC, control._VERSION, len(payload)) + payload

    def test_bidirectional_messages_and_tcp_nodelay(self):
        client, server = self.pair()
        self.assertEqual(client._socket.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY), 1)
        messages = [("hello", {"receiver": {"control_address": "10.0.0.1:29000"}, "groups": [(2, 16)]}),
                    ("request", 0), ("reuse", {"sequence": 1, "mr_id": 2}), ("shutdown", "session")]
        for message in messages:
            client.send(message)
            self.assertEqual(server.recv(), message)
            server.send(("ack", message[1]))
            self.assertEqual(client.recv(), ("ack", message[1]))

    def test_fragmented_and_coalesced_frames(self):
        client, server = self.pair()
        client._socket.setblocking(True)
        messages = [("request", i) for i in range(3)]
        first = self.frame(messages[0])
        errors = []

        def write():
            try:
                for value in first:
                    client._socket.sendall(bytes([value]))
                client._socket.sendall(b"".join(self.frame(message) for message in messages[1:]))
            except BaseException as error:
                errors.append(error)

        writer = threading.Thread(target=write)
        writer.start()
        try:
            self.assertEqual([server.recv() for _ in messages], messages)
        finally:
            writer.join(2)
        self.assertFalse(writer.is_alive())
        self.assertEqual(errors, [])

    def test_invalid_headers_and_payloads_close_the_session(self):
        for header in ((b"NOPE", 1, 10), (b"KVNT", 0, 10), (b"KVNT", 1, 0),
                       (b"KVNT", 1, control._MAX_FRAME + 1)):
            with self.subTest(header=header):
                client, server = self.pair()
                client._socket.sendall(control._HEADER.pack(*header))
                with self.assertRaises(control.ProtocolError):
                    server.recv()
                self.assertEqual(server._socket.fileno(), -1)
        for payload in (b"invalid pickle", pickle.dumps(["request", 1]),
                        pickle.dumps(("request", 1)) + b"trailing"):
            with self.subTest(payload=payload):
                client, server = self.pair()
                client._socket.sendall(control._HEADER.pack(b"KVNT", 1, len(payload)) + payload)
                with self.assertRaises(control.ProtocolError):
                    server.recv()

    def test_disconnect_in_header_or_body_is_terminal(self):
        frame = self.frame(("request", 1))
        for size in (0, 3, control._HEADER.size + 2):
            with self.subTest(size=size):
                client, server = self.pair()
                if size:
                    client._socket.sendall(frame[:size])
                client.close()
                with self.assertRaises(EOFError):
                    server.recv()
                self.assertEqual(server._socket.fileno(), -1)

    def test_timeout_covers_partial_frame(self):
        client, server = self.pair(timeout=.08)
        client._socket.sendall(self.frame(("request", 1))[:3])
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            server.recv()
        self.assertLess(time.monotonic() - start, 1)
        self.assertEqual(server._socket.fileno(), -1)

    def test_arbitrary_pickle_globals_are_rejected(self):
        class Unexpected:
            def __reduce__(self):
                return eval, ("1 + 1",)

        client, server = self.pair()
        frame = self.frame(("ready", Unexpected()))
        client._socket.sendall(frame)
        with patch("builtins.eval") as evaluate:
            with self.assertRaisesRegex(control.ProtocolError, "unsupported notification object"):
                server.recv()
        evaluate.assert_not_called()

    def test_address_timeout_and_outgoing_message_validation(self):
        for address in ("missing-port", ":12", "127.0.0.1:-1", "127.0.0.1:65536", "127.0.0.1:no"):
            with self.subTest(address=address), self.assertRaises(ValueError):
                control.split_address(address)
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                control.TcpControlListener("127.0.0.1:0", timeout=timeout)
        client, server = self.pair()
        for message in (["request", 1], ("", None), ("request", b"x" * control._MAX_FRAME)):
            with self.subTest(kind=type(message)), self.assertRaises(control.ProtocolError):
                client.send(message)
        client.send(("request", 1))
        self.assertEqual(server.recv(), ("request", 1))

    def test_reader_configuration_is_validated_and_preserves_local_device(self):
        reader = bench.parse_args(["--role", "reader", "--device-a", "0", "--device-b", "0",
                                   "--control-connect", "127.0.0.1:29100",
                                   "--npu-eid-a-raw", "a", "--graph", "--source-count", "2",
                                   "--packet-group", "2:16", "--packet-group", "2:4", "--verify-source"])
        producer = bench.parse_args(["--role", "producer", "--device-b", "0", "--npu-eid-b-raw", "b",
                                     "--control-listen", "127.0.0.1:29100"])
        adopted = bench.configured_producer(producer, bench.benchmark_config(reader))
        self.assertEqual(bench.packet_layout(reader), bench.packet_layout(adopted))
        self.assertEqual(bench.round_counts(reader), bench.round_counts(adopted))
        self.assertEqual((adopted.device_b, adopted.npu_eid_b_raw), (0, "b"))
        self.assertTrue(adopted.graph)
        self.assertTrue(adopted.verify_source)
        for change in ({"device_b": 7}, {"iterations": "2"}, {"packet_groups": []}, {"graph": 1}, {"verify_source": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                bench.configured_producer(producer, {**bench.benchmark_config(reader), **change})
        for argv in (["--role", "reader", "--npu-eid-a-raw", "a"],
                     ["--role", "producer", "--npu-eid-b-raw", "b", "--control-connect", "127.0.0.1:29100"],
                     ["--role", "reader", "--npu-eid-a-raw", "a", "--control-connect", "127.0.0.1:29100",
                      "--control-listen", "127.0.0.1:29101"],
                     ["--role", "reader", "--npu-eid-a-raw", "a", "--control-connect", "127.0.0.1:29100",
                      "--control-timeout", "nan"],
                     ["--npu-eid-a-raw", "a", "--npu-eid-b-raw", "b", "--control-listen", "127.0.0.1:0"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.parse_args(argv)

    def test_default_local_role_spawns_peer_and_exchanges_bootstrap_over_tcp(self):
        # Reserve a test port briefly; the spawned producer itself binds/listens.
        with control.TcpControlListener("127.0.0.1:0", timeout=5) as probe:
            address = probe.address
        args = bench.parse_args(["--npu-eid-a-raw", "a", "--npu-eid-b-raw", "b",
                                 "--control-listen", address, "--control-timeout", "5"])
        package = types.ModuleType("kvaf")
        package.__path__ = []
        with patch.dict(sys.modules, {"kvaf": package, "kvaf.tcp_control": control}), \
                patch.object(bench, "producer_main", tcp_process_peer), contextlib.redirect_stdout(io.StringIO()):
            channel, worker = bench.open_reader_channel(args)
        try:
            channel.send(("hello", {"receiver": "sent-over-tcp"}))
            self.assertEqual(channel.recv(), ("connected", {"receiver": "sent-over-tcp"}))
            channel.send(("request", 42))
            self.assertEqual(channel.recv(), ("ready", 42))
            worker.join(5)
            self.assertEqual(worker.exitcode, 0)
        finally:
            channel.close()
            if worker.is_alive():
                worker.terminate()
                worker.join(5)

    def test_real_tcp_runs_reader_and_producer_with_multiple_mrs(self):
        self.run_benchmark_session()

    def test_failed_read_wait_never_sends_reuse_or_releases_source(self):
        self.run_benchmark_session(fail_wait=True)

    def test_failed_registration_with_confirmed_rollback_closes_reader(self):
        self.run_benchmark_session(startup_failure="clean")

    def test_failed_registration_with_uncertain_rollback_retains_both_peers(self):
        self.run_benchmark_session(startup_failure="pending")

    def test_failed_registration_without_matching_cleanup_receipt_retains_reader(self):
        for mode in ("wrong-session", "lost-ack"):
            with self.subTest(mode=mode):
                self.run_benchmark_session(startup_failure=mode)

    def test_consumer_mismatch_keeps_source_frozen_without_reuse(self):
        self.run_benchmark_session(fail_compare=True)

    def run_benchmark_session(self, fail_wait=False, startup_failure=None, fail_compare=False):
        # Same module names as the pybind classes exercise the restricted token
        # serialization. Only tensor/kernel behavior is mocked; every application
        # message, workload negotiation and batch receipt uses a real TCP socket.
        package = types.ModuleType("kvaf")
        package.__path__ = []
        native = types.ModuleType("kvaf._kvaf")
        for name in ("ReadyToken", "NpuPullSourceDesc"):
            setattr(native, name, type(name, (), {"__module__": "kvaf._kvaf"}))
        package._kvaf = native
        torch = MagicMock()
        torch.npu.stream.side_effect = lambda stream: contextlib.nullcontext()
        torch.tensor.side_effect = lambda values, **kwargs: tuple(values)
        torch.equal.return_value = not fail_compare
        events = []
        source, runtime = MagicMock(), MagicMock()
        runtime.initialized, runtime.ddr_initialized, runtime.pending_raw_addr = True, False, False
        runtime.listen_hbm.return_value = dict(version=1, control_address="127.0.0.1:29000",
                                              receiver_id=10, receiver_epoch=20)
        source.frozen = False
        counter = [0]
        reader_errors, producer_errors = [], []

        class Retained(Exception):
            pass

        def stream(**kwargs):
            owner = threading.current_thread().name

            def synchronize():
                events.append((owner, "sync"))
                if fail_wait and owner != "tcp-producer" and any(event[1] == "read" for event in events):
                    raise RuntimeError("uncertain NPU stream completion")

            return types.SimpleNamespace(synchronize=synchronize)

        torch.npu.Stream.side_effect = stream

        def ready(*unused):
            self.assertFalse(source.frozen)
            source.frozen = True
            value = native.ReadyToken()
            value.source = native.NpuPullSourceDesc()
            value.source.source_id, value.source.source_epoch = 7, 11
            value.sequence, value.mr_id = counter[0] + 1, counter[0] % 2 + 1
            counter[0] += 1
            events.append(("tcp-producer", "ready"))
            return value

        def release(token):
            self.assertTrue(source.frozen)
            source.frozen = False
            events.append(("tcp-producer", "release"))

        source.ready.side_effect = ready
        source.release_batch.side_effect = release
        if startup_failure:
            source.register.side_effect = RuntimeError("mock HBM context initialization failed")
            if startup_failure == "pending":
                source.shutdown.side_effect = RuntimeError("source teardown remains InDoubt")
        runtime.batch_get_raw_addr.side_effect = lambda *a, **k: events.append(("reader", "read")) or {"state": "submitted"}
        package.NpuPullHbmSource = lambda *a: source
        package.NpuRuntime = lambda **kwargs: runtime
        modules = {"kvaf": package, "kvaf._kvaf": native, "kvaf.tcp_control": control,
                   "torch": torch, "torch_npu": types.SimpleNamespace()}
        reader_argv = ["--role", "reader", "--npu-eid-a-raw", "a", "--device-a", "0",
                                        "--device-b", "0", "--packets", "2", "--packet-bytes", "8",
                                        "--source-space-bytes", "64", "--source-count", "2", "--warmup", "2",
                       "--iterations", "2", "--profile-iterations", "0", "--control-timeout", "2"]
        with control.TcpControlListener("127.0.0.1:0", timeout=2) as listener:
            reader_args = bench.parse_args(reader_argv + ["--control-connect", listener.address])
            torch.empty.return_value.numel.return_value = reader_args.payload_bytes + 2 * bench.GUARD_BYTES
            producer_args = bench.parse_args(["--role", "producer", "--npu-eid-b-raw", "b", "--device-b", "0",
                                              "--control-listen", listener.address, "--control-timeout", "2"])
            listening = threading.Event()

            def listen(*args, **kwargs):
                self.assertEqual(threading.current_thread().name, "tcp-producer")
                listening.set()
                return listener

            def produce():
                try:
                    bench.producer_main(producer_args)
                except BaseException as error:
                    producer_errors.append(error)

            send = control.TcpControlChannel.send

            def send_message(channel, message):
                if message[0] == "startup_failed":
                    if startup_failure == "wrong-session":
                        message = ("startup_failed", {**message[1], "session": "different-session"})
                    elif startup_failure == "lost-ack":
                        channel.close()
                        raise BrokenPipeError("startup rollback confirmation lost")
                return send(channel, message)

            with patch.dict(sys.modules, modules), patch.object(control, "TcpControlListener", side_effect=listen) as server, \
                    patch.object(control.TcpControlChannel, "send", new=send_message), \
                    patch.object(bench, "packet_payload", return_value=MagicMock()), \
                    patch.object(bench, "report_read_mismatch") as diagnostic, \
                    patch.object(bench, "retain", side_effect=Retained), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                worker = threading.Thread(target=produce, name="tcp-producer", daemon=True)
                worker.start()
                try:
                    self.assertTrue(listening.wait(2), "producer did not start its server")
                    source.register.assert_not_called()  # Wait for the reader's HELLO/receiver ticket first.
                    bench.reader(reader_args)
                except BaseException as error:
                    reader_errors.append(error)
                finally:
                    worker.join(5)
                self.assertFalse(worker.is_alive(), "TCP peer did not finish")
                server.assert_called_once_with(producer_args.control_listen, timeout=producer_args.control_timeout)
        if startup_failure:
            self.assertEqual(len(reader_errors), 1)
            self.assertEqual(len(producer_errors), 1)
            source.shutdown.assert_called_once()
            source.release_batch.assert_not_called()
            runtime.batch_get_raw_addr.assert_not_called()
            if startup_failure == "clean":
                self.assertIsInstance(reader_errors[0], bench.ProducerStartupFailed)
                runtime.shutdown.assert_called_once()
            else:
                self.assertIsInstance(reader_errors[0], Retained)
                runtime.shutdown.assert_not_called()
            if startup_failure == "pending":
                self.assertIsInstance(producer_errors[0], Retained)
            else:
                self.assertIsInstance(producer_errors[0], RuntimeError)
                self.assertIn("HBM context initialization failed", str(producer_errors[0]))
            return
        if fail_wait or fail_compare:
            self.assertEqual(len(reader_errors), 1)
            self.assertEqual(len(producer_errors), 1)
            self.assertIsInstance(reader_errors[0], Retained)
            self.assertIsInstance(producer_errors[0], Retained)
            source.release_batch.assert_not_called()
            source.shutdown.assert_not_called()
            runtime.shutdown.assert_not_called()
            self.assertTrue(source.frozen)
            if fail_compare:
                diagnostic.assert_called_once()
        else:
            self.assertEqual(reader_errors, [])
            self.assertEqual(producer_errors, [])
            self.assertEqual(source.release_batch.call_count, 4)
            self.assertEqual(runtime.batch_get_raw_addr.call_count, 4)
            source.shutdown.assert_called_once()
            runtime.shutdown.assert_called_once()
            self.assertFalse(source.frozen)
            # Each release follows a READ and an explicit reader stream wait.
            for index, event in enumerate(events):
                if event[1] == "release":
                    self.assertEqual(events[index - 1][1], "sync")
                    self.assertNotEqual(events[index - 1][0], "tcp-producer")
                    self.assertEqual(events[index - 2][1], "read")


if __name__ == "__main__":
    unittest.main()
