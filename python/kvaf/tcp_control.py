"""Small, framed application notifications for KVAF connectors.

This channel carries receiver tickets, ReadyTokens and application metadata,
never HBM payload. It is independent of KVAF's internal KVHR connection.
Use with trusted cluster peers: this protocol does not authenticate endpoints.
Broken sessions are terminal; reconnecting must not implicitly authorize reuse
of a frozen source or replay an uncertain READ.
"""
import importlib
import io
import math
import pickle
import select
import socket
import struct
import threading
import time


_HEADER = struct.Struct("!4sBI")
_MAGIC = b"KVNT"
_VERSION = 1
_MAX_FRAME = 4 * 1024 * 1024


class ProtocolError(ValueError):
    """The peer sent an invalid or incompatible notification frame."""


class _NotificationUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Preserve the binding's opaque, versioned token serialization without
        # allowing arbitrary Python globals to execute from a network frame.
        if module == "kvaf._kvaf" and name in ("ReadyToken", "NpuPullSourceDesc"):
            return getattr(importlib.import_module(module), name)
        raise ProtocolError(f"unsupported notification object: {module}.{name}")


def _decode(data):
    stream = io.BytesIO(data)
    try:
        value = _NotificationUnpickler(stream).load()
    except ProtocolError:
        raise
    except Exception as error:
        raise ProtocolError("invalid notification payload") from error
    if stream.read(1):
        raise ProtocolError("trailing notification payload")
    _validate_message(value)
    return value


def _validate_message(value):
    if (not isinstance(value, tuple) or len(value) != 2 or
            not isinstance(value[0], str) or not value[0] or len(value[0]) > 64):
        raise ProtocolError("notifications must be (kind, payload) pairs")


def _timeout(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout must be finite and positive")
    return value


def split_address(address):
    """Parse the IPv4 host:port notation also used by KVHR."""
    host, separator, port = address.rpartition(":")
    if not separator or not host or ":" in host:
        raise ValueError("TCP control address must be IPv4 host:port")
    try:
        port = int(port)
    except ValueError:
        raise ValueError("TCP control port must be an integer") from None
    if not 0 <= port <= 65535:
        raise ValueError("TCP control port must be in 0..65535")
    return host, port


class TcpControlChannel:
    """Full-duplex notifications, with one serialized sender and receiver.

    Each send/recv has an absolute timeout, including fragmented frames.
    send() only delivers a message to the socket; applications must separately
    acknowledge processing and verify NPU stream completion before source reuse.
    """
    def __init__(self, sock, *, timeout=180):
        self.timeout = _timeout(timeout)
        self._socket = sock
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setblocking(False)

    def _wait(self, writing, deadline):
        if self._socket.fileno() < 0:
            raise ConnectionError("TCP notification connection is closed")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("TCP notification timed out; retain unresolved resources")
        try:
            readers, writers, errors = select.select(
                [] if writing else [self._socket], [self._socket] if writing else [],
                [self._socket], remaining)
        except (OSError, ValueError) as error:
            raise ConnectionError("TCP notification connection failed") from error
        if errors:
            raise ConnectionError("TCP notification connection failed")
        if not readers and not writers:
            raise TimeoutError("TCP notification timed out; retain unresolved resources")

    def _read_exact(self, size, deadline):
        result = bytearray()
        while len(result) < size:
            self._wait(False, deadline)
            try:
                part = self._socket.recv(size - len(result))
            except (BlockingIOError, InterruptedError):
                continue
            if not part:
                raise EOFError("TCP notification disconnected; retain unresolved resources")
            result.extend(part)
        return bytes(result)

    def send(self, message):
        _validate_message(message)
        payload = pickle.dumps(message, protocol=4)
        if len(payload) > _MAX_FRAME:
            raise ProtocolError("notification exceeds 4 MiB")
        frame = memoryview(_HEADER.pack(_MAGIC, _VERSION, len(payload)) + payload)
        with self._send_lock:
            deadline = time.monotonic() + self.timeout
            try:
                while frame:
                    self._wait(True, deadline)
                    try:
                        sent = self._socket.send(frame)
                    except (BlockingIOError, InterruptedError):
                        continue
                    if not sent:
                        raise ConnectionError("TCP notification send failed")
                    frame = frame[sent:]
            except Exception:
                self.close()
                raise

    def recv(self):
        with self._recv_lock:
            deadline = time.monotonic() + self.timeout
            try:
                magic, version, size = _HEADER.unpack(self._read_exact(_HEADER.size, deadline))
                if magic != _MAGIC or version != _VERSION or not 0 < size <= _MAX_FRAME:
                    raise ProtocolError("invalid notification magic, version or frame length")
                return _decode(self._read_exact(size, deadline))
            except Exception:
                self.close()
                raise

    def close(self):
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    @classmethod
    def connect(cls, address, *, timeout=180):
        timeout = _timeout(timeout)
        target = split_address(address)
        if not target[1] or target[0] == "0.0.0.0":
            raise ValueError("connect requires a peer address and nonzero port")
        # Only initial establishment retries. Never reconnect an active session.
        deadline = time.monotonic() + timeout
        while True:
            try:
                sock = socket.create_connection(target, timeout=max(.001, deadline - time.monotonic()))
                try:
                    return cls(sock, timeout=timeout)
                except Exception:
                    sock.close()
                    raise
            except OSError as error:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"cannot connect TCP notifications to {address}") from error
                time.sleep(min(.1, remaining))


class TcpControlListener:
    """Accept one application session; address exposes an OS-selected port."""
    def __init__(self, address, *, timeout=180):
        self.timeout = _timeout(timeout)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Windows SO_REUSEADDR permits another live listener to steal a
            # port. POSIX uses it only to avoid TIME_WAIT after clean shutdown.
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind(split_address(address))
            self._socket.listen(1)
            self._socket.settimeout(self.timeout)
            host, port = self._socket.getsockname()
            self.address = f"{host}:{port}"
        except Exception:
            self.close()
            raise

    def accept(self):
        sock, _ = self._socket.accept()
        try:
            return TcpControlChannel(sock, timeout=self.timeout)
        except Exception:
            sock.close()
            raise

    def close(self):
        self._socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
