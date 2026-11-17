"""Per-iteration host barriers and common-window BatchGet measurements.

All workers run on the launcher's host and use its monotonic clock. These
windows include Python submission and device synchronization, not just kernel
execution. A host barrier cannot prove simultaneous READs on different NPUs.
"""

import csv
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import select
import socket
import statistics
import time


PROTOCOL_VERSION = 2
MAX_MESSAGE_BYTES = 4096
REPORT_FILES = ("concurrent_batch_get_samples.csv", "concurrent_batch_get_flows.csv",
                "concurrent_batch_get_summary.json")


def send_message(connection, **message):
    connection.sendall((json.dumps(message) + "\n").encode("ascii"))


def read_message(stream):
    line = stream.readline(MAX_MESSAGE_BYTES + 1)
    if not line.endswith(b"\n") or len(line) > MAX_MESSAGE_BYTES:
        raise RuntimeError("concurrent worker/launcher disconnected or sent an incomplete message")
    try:
        message = json.loads(line)
    except (ValueError, UnicodeError) as error:
        raise RuntimeError("invalid concurrent measurement message") from error
    if not isinstance(message, dict):
        raise RuntimeError("concurrent measurement message must be an object")
    return message


class ConcurrentMeasurement:
    """Silent no-op for standalone runs; persistent rendezvous for multi-flow runs."""

    def __init__(self, iterations, payload_bytes, *, execution_mode="eager", calls_per_round=1):
        self.iterations = iterations
        self.payload_bytes = payload_bytes
        self.execution_mode = execution_mode
        self.calls_per_round = calls_per_round
        self.connection = self.stream = None

    def __enter__(self):
        endpoint = os.environ.get("KVAF_BATCH_GET_START_GATE")
        if not endpoint:
            return self
        host, port = endpoint.rsplit(":", 1)
        timeout = int(os.environ.get("KVAF_REMOTE_CPU_PERF_TIMEOUT", "600"))
        try:
            self.connection = socket.create_connection((host, int(port)), timeout=timeout)
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.stream = self.connection.makefile("rb")
            send_message(self.connection, kind="hello", version=PROTOCOL_VERSION,
                         token=os.environ["KVAF_BATCH_GET_START_TOKEN"],
                         iterations=self.iterations, payload_bytes=self.payload_bytes,
                         execution_mode=self.execution_mode, calls_per_round=self.calls_per_round)
            if read_message(self.stream) != {"kind": "accepted"}:
                raise RuntimeError("concurrent launcher did not accept this worker")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def before(self, iteration):
        if self.connection is None:
            return
        send_message(self.connection, kind="ready", iteration=iteration)
        message = read_message(self.stream)
        if (message.get("kind") != "start" or message.get("iteration") != iteration
                or type(message.get("start_ns")) is not int):
            raise RuntimeError("invalid concurrent round release")
        while True:
            remaining = message["start_ns"] - time.monotonic_ns()
            if remaining <= 0:
                return
            time.sleep(remaining / 1_000_000_000)

    def after(self, iteration, begin_ns, end_ns, *, device_elapsed_ms=None):
        if self.connection is None:
            return
        send_message(self.connection, kind="done", iteration=iteration,
                     begin_ns=begin_ns, end_ns=end_ns, device_elapsed_ms=device_elapsed_ms)
        if read_message(self.stream) != {"kind": "complete", "iteration": iteration}:
            raise RuntimeError("concurrent launcher disconnected before round completion")

    def __exit__(self, *unused):
        if self.stream is not None:
            self.stream.close()
        if self.connection is not None:
            self.connection.close()
        self.stream = self.connection = None


def round_sample(iteration, scheduled_start_ns, measurements):
    """Aggregate bytes over the window spanning all this round's host intervals."""
    starts = [row["begin_ns"] for row in measurements]
    ends = [row["end_ns"] for row in measurements]
    for begin, end in zip(starts, ends):
        if type(begin) is not int or type(end) is not int or not scheduled_start_ns <= begin < end:
            raise RuntimeError("invalid concurrent measurement timestamps")
    first_start, last_start, first_end, last_end = min(starts), max(starts), min(ends), max(ends)
    window_ns = last_end - first_start
    payload = sum(row["payload_bytes"] for row in measurements)
    return dict(iteration=iteration, flow_count=len(measurements), payload_bytes=payload,
                scheduled_start_ns=scheduled_start_ns, first_start_ns=first_start,
                last_start_ns=last_start, last_end_ns=last_end, window_ns=window_ns,
                start_skew_ns=last_start - first_start,
                host_overlap_ns=max(0, first_end - last_start),
                aggregate_GBps=payload / window_ns)


def coordinate_rounds(listener, tokens, check_health, deadline, emit):
    """Release every round only after all workers are ready; fail as a group."""
    peers, samples, measurements = {}, [], []
    with ExitStack() as resources:
        while len(peers) < len(tokens):
            check_health()
            if time.monotonic() >= deadline:
                missing = [name for token, name in tokens.items() if token not in peers]
                raise TimeoutError(f"waiting for {', '.join(missing)} to become ready")
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            resources.enter_context(connection)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection.settimeout(min(1.0, max(0.01, deadline - time.monotonic())))
            stream = resources.enter_context(connection.makefile("rb"))
            message = read_message(stream)
            token = message.get("token")
            if (message.get("kind") != "hello" or message.get("version") != PROTOCOL_VERSION
                    or not isinstance(token, str) or token not in tokens or token in peers):
                raise RuntimeError("invalid or duplicate concurrent worker handshake; update both scripts")
            if (type(message.get("iterations")) is not int or message["iterations"] < 1
                    or type(message.get("payload_bytes")) is not int or message["payload_bytes"] < 0):
                raise RuntimeError("invalid concurrent iteration count or payload size")
            if (message.get("execution_mode") not in ("eager", "graph")
                    or type(message.get("calls_per_round")) is not int
                    or message["calls_per_round"] < 1
                    or (message["execution_mode"] == "eager" and message["calls_per_round"] != 1)):
                raise RuntimeError("invalid concurrent execution mode or calls per round")
            peers[token] = dict(connection=connection, stream=stream, name=tokens[token], **message)
            emit(f"[launcher] {tokens[token]} ready")

        counts = {peer["iterations"] for peer in peers.values()}
        if len(counts) != 1:
            raise ValueError("all flows must use the same --iterations for per-iteration synchronization")
        for peer in peers.values():
            send_message(peer["connection"], kind="accepted")
        group = "both flows" if len(peers) == 2 else f"all {len(peers)} flows"
        emit(f"[launcher] {group} ready; synchronized start scheduled for every iteration")

        def collect(kind, iteration):
            pending = {peer["connection"]: peer for peer in peers.values()}
            messages = {}
            while pending:
                check_health()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    names = ', '.join(peer["name"] for peer in pending.values())
                    raise TimeoutError(f"round {iteration}: waiting for {names} ({kind})")
                readable, _, _ = select.select(list(pending), [], [], min(0.05, remaining))
                for connection in readable:
                    peer = pending.pop(connection)
                    message = read_message(peer["stream"])
                    if message.get("kind") != kind or message.get("iteration") != iteration:
                        raise RuntimeError(f"{peer['name']}: expected {kind} for round {iteration}")
                    messages[peer["name"]] = message
            return messages

        for iteration in range(counts.pop()):
            collect("ready", iteration)
            # Allow every local process to receive the same future deadline.
            # Actual wakeup/submission skew is measured, not assumed to be zero.
            start_ns = time.monotonic_ns() + 5_000_000
            for peer in peers.values():
                send_message(peer["connection"], kind="start", iteration=iteration, start_ns=start_ns)
            done = collect("done", iteration)
            rows = []
            for token, name in tokens.items():
                peer = peers[token]
                device_ms = done[name].get("device_elapsed_ms")
                if peer["execution_mode"] == "graph" and (
                        type(device_ms) not in (int, float)
                        or not math.isfinite(device_ms) or device_ms <= 0):
                    raise RuntimeError(f"{name}: graph round requires a positive finite NPU Event duration")
                rows.append(dict(iteration=iteration, flow=name,
                                 execution_mode=peer["execution_mode"],
                                 calls_per_round=peer["calls_per_round"],
                                 payload_bytes=peer["payload_bytes"],
                                 begin_ns=done[name].get("begin_ns"), end_ns=done[name].get("end_ns"),
                                 device_elapsed_ms=device_ms))
            samples.append(round_sample(iteration, start_ns, rows))
            measurements.extend(rows)
            for peer in peers.values():
                send_message(peer["connection"], kind="complete", iteration=iteration)
    return samples, measurements


def report_results(samples, measurements, output_dir, emit):
    total_bytes = sum(row["payload_bytes"] for row in samples)
    total_ns = sum(row["window_ns"] for row in samples)
    summary = dict(synchronization="per_iteration", timing="host_call_to_device_synchronize",
                   samples=len(samples), flow_count=samples[0]["flow_count"],
                   total_payload_bytes=total_bytes, total_window_ns=total_ns,
                   aggregate_GBps=total_bytes / total_ns,
                   mean_window_us=total_ns / len(samples) / 1000,
                   p50_window_us=statistics.median(row["window_ns"] for row in samples) / 1000,
                   max_start_skew_us=max(row["start_skew_ns"] for row in samples) / 1000,
                   all_flow_host_overlap_samples=sum(row["host_overlap_ns"] > 0 for row in samples))
    modes = {row["execution_mode"] for row in measurements}
    summary["execution_mode"] = next(iter(modes)) if len(modes) == 1 else "mixed"
    if modes == {"graph"}:
        summary["timing"] = "host_graph_replay_to_event_completion"
    elif len(modes) > 1:
        summary["timing"] = "host_submit_to_completion"
    summary["flows"] = {}
    for name in dict.fromkeys(row["flow"] for row in measurements):
        rows = [row for row in measurements if row["flow"] == name]
        first = rows[0]
        flow = dict(device_id=first.get("device_id"), execution_mode=first["execution_mode"],
                    calls_per_round=first["calls_per_round"],
                    payload_bytes_per_round=first["payload_bytes"])
        if first["execution_mode"] == "graph":
            device_ms = sum(row["device_elapsed_ms"] for row in rows)
            flow.update(device_event_mean_us_per_replay=device_ms * 1000 / len(rows),
                        device_event_mean_us_per_get=device_ms * 1000 / len(rows) / first["calls_per_round"],
                        device_event_GBps=sum(row["payload_bytes"] for row in rows) / (device_ms * 1_000_000),
                        profile_scope="last_get_in_replay")
        summary["flows"][name] = flow
    if output_dir is not None:
        output = Path(output_dir)
        for filename, rows in zip(REPORT_FILES[:2], (samples, measurements)):
            with (output / filename).open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        with (output / REPORT_FILES[2]).open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
            stream.write("\n")
    emit("BATCH_GET_CONCURRENT_SUMMARY " + " ".join(
        f"{key}={value}" for key, value in summary.items() if key != "flows"))
    for name, flow in summary["flows"].items():
        emit(f"BATCH_GET_CONCURRENT_FLOW_SUMMARY flow={name} " + " ".join(
            f"{key}={value}" for key, value in flow.items()))
    return summary
