"""Manual two-NPU benchmark: random packets in a registered HBM space.

A reads B with batch_get_raw_addr without a CPU service or DDR allocation.
Payload travels only from B HBM to A HBM. Setup, producer
writes, metadata preparation, verification and TCP notifications are outside timing.
The producer listens for application TCP connections; the reader connects to it.
The default local role spawns B but exchanges all control messages over TCP.
Use --role reader / --role producer to launch the peers on separate hosts.
"""
import argparse
from dataclasses import dataclass
import hashlib
from math import gcd
import multiprocessing as mp
import os
import time
import traceback
import uuid

from pull_bench_utils import print_stages, scattered_offsets, summarize_us
from test_remote_cpu_offload import retain

GUARD_BYTES = 64


def packet_group(value):
    try:
        count, size = (int(part) for part in value.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError("packet-group must be COUNT:BYTES") from None
    if count < 1 or not 1 <= size <= (1 << 32) - 1:
        raise argparse.ArgumentTypeError("group count must be positive; bytes must be in 1..UINT32_MAX")
    return count, size


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("local", "reader", "producer"), default="local")
    parser.add_argument("--control-listen", help="producer application TCP host:port (default: 127.0.0.1:29100)")
    parser.add_argument("--control-connect", help="reader connects to the producer application TCP address")
    parser.add_argument("--control-timeout", type=float, default=180, help="TCP operation timeout in seconds")
    parser.add_argument("--hbm-listen", default="127.0.0.1:0", help="independent KVHR hardware setup listener")
    parser.add_argument("--hbm-advertise", default="", help="producer-reachable KVHR host:port")
    parser.add_argument("--device-a", type=int, default=6, help="reader NPU")
    parser.add_argument("--device-b", type=int, default=7, help="producer NPU")
    parser.add_argument("--npu-eid-a-raw", default="", help="A EID for the verified NPU-to-NPU route")
    parser.add_argument("--npu-eid-b-raw", default="", help="B EID for the verified NPU-to-NPU route")
    parser.add_argument("--packets", "--blocks", type=int, help="uniform packet count (default: 4096)")
    parser.add_argument("--packet-bytes", "--block-bytes", type=int, help="uniform packet bytes (default: 656)")
    parser.add_argument("--packet-group", "--block-group", dest="packet_groups", action="append", type=packet_group,
                        help="repeat COUNT:BYTES for mixed sizes in one READ; excludes uniform size/count flags")
    parser.add_argument("--source-space-bytes", type=int, default=256 * 1024 * 1024, help="bytes per registered tensor")
    parser.add_argument("--source-count", type=int, default=1, help="1..4096 independent MRs sharing one connection")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--graph", action="store_true",
                        help="capture one raw READ per MR and measure replay through stream completion")
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-iterations", type=int, default=None,
                        help="eager-only instrumented rounds (default: 20 eager, 0 graph); 0 disables")
    parser.add_argument("--validate-ranges", action="store_true",
                        help="enable optional length/source/output bounds diagnostics; no overlap detection")
    parser.add_argument("--verify-source", action="store_true",
                        help="diagnostic: copy B's source to CPU and verify written packets before each ready")
    parser.add_argument("--aiv-cores", type=int, default=2)
    parser.add_argument("--jettys-per-core", type=int, default=2)
    parser.add_argument("--max-inflight-total", type=int, default=2048)
    parser.add_argument("--raw-kernel", default="build-npu/bin/kvaf_execute_batch_get_raw_addr_kernels.o")
    args = parser.parse_args(argv)
    if args.graph and os.environ.get("KVAF_HBM_RAW_READ_DIAGNOSTICS") == "1":
        parser.error("KVAF_HBM_RAW_READ_DIAGNOSTICS=1 requires eager execution; remove --graph")
    if args.role != "producer" and not args.npu_eid_a_raw:
        parser.error("reader/local requires --npu-eid-a-raw")
    if args.role != "reader" and not args.npu_eid_b_raw:
        parser.error("producer/local requires --npu-eid-b-raw")
    if args.role == "reader" and not args.control_connect:
        parser.error("reader requires --control-connect")
    if args.role != "reader" and args.control_connect:
        parser.error("--control-connect is only used by the reader role")
    if args.role == "reader" and args.control_listen:
        parser.error("--control-listen is only used by producer/local roles")
    if args.role != "reader":
        args.control_listen = args.control_listen or "127.0.0.1:29100"
    if args.role == "local":
        try:
            local_port = int(args.control_listen.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            parser.error("local role requires --control-listen host:port")
        if not 1 <= local_port <= 65535:
            parser.error("local role requires a producer port in 1..65535")
    if not 0 < args.control_timeout < float("inf"):
        parser.error("control-timeout must be finite and positive")
    if args.profile_iterations is None:
        args.profile_iterations = 0 if args.graph else 20
    if args.graph and args.profile_iterations:
        parser.error("graph replay bypasses Host call profiling; use --profile-iterations 0")
    if not 1 <= args.source_count <= 4096:
        parser.error("source-count must be in 1..4096")
    # Visit every MR/output pair before timed samples; retain single-MR defaults.
    if args.source_count > 1 and args.warmup >= 0:
        args.warmup = max(args.warmup, args.source_count)
    if args.role == "local" and args.device_a == args.device_b:
        parser.error("use two distinct NPUs")
    if args.packet_groups:
        if args.packets is not None or args.packet_bytes is not None:
            parser.error("packet-group cannot be combined with packets/packet-bytes")
        args.packets = sum(count for count, _ in args.packet_groups)
    else:
        args.packets = 4096 if args.packets is None else args.packets
        args.packet_bytes = 656 if args.packet_bytes is None else args.packet_bytes
        if not 1 <= args.packet_bytes <= (1 << 32) - 1:
            parser.error("packet-bytes must be in 1..UINT32_MAX")
        args.packet_groups = [(args.packets, args.packet_bytes)]
    if not 1 <= args.packets <= 4096:
        parser.error("total packets must be in 1..4096")
    args.payload_bytes = sum(count * size for count, size in args.packet_groups)
    if not args.payload_bytes <= args.source_space_bytes <= (1 << 63) - 1:
        parser.error("source-space-bytes must fit all packets and signed int64 offsets")
    if args.iterations < 1 or min(args.warmup, args.profile_warmup, args.profile_iterations) < 0:
        parser.error("iterations must be positive; warmup/profile counts must be nonnegative")
    if args.aiv_cores < 1 or args.jettys_per_core < 1 or args.aiv_cores * args.jettys_per_core > 4:
        parser.error("raw READ supports 1..4 total Jettys")
    if not 0 <= args.max_inflight_total <= (1 << 32) - 1 or (
            args.max_inflight_total and args.max_inflight_total < 2 * args.aiv_cores):
        parser.error("max-inflight-total must be 0 or at least twice aiv-cores, within uint32")
    return args


_CONFIG_INTS = ("source_space_bytes", "source_count", "seed", "warmup", "iterations",
                "profile_warmup", "profile_iterations", "aiv_cores", "jettys_per_core", "max_inflight_total")
_CONFIG_BOOLS = ("graph", "validate_ranges", "verify_source")


def benchmark_config(args):
    return {name: getattr(args, name) for name in (*_CONFIG_INTS, *_CONFIG_BOOLS, "packet_groups")}


def configured_producer(args, config):
    """A chooses the workload; B retains its own device, EID and TCP settings."""
    if not isinstance(config, dict) or set(config) != set((*_CONFIG_INTS, *_CONFIG_BOOLS, "packet_groups")):
        raise ValueError("invalid benchmark configuration fields")
    argv = ["--role", "producer", "--control-listen", args.control_listen,
            "--control-timeout", str(args.control_timeout), "--device-b", str(args.device_b),
            "--npu-eid-b-raw", args.npu_eid_b_raw]
    for name in _CONFIG_INTS:
        if type(config[name]) is not int:
            raise ValueError(f"invalid benchmark integer: {name}")
        argv.extend(("--" + name.replace("_", "-"), str(config[name])))
    for name in _CONFIG_BOOLS:
        if type(config[name]) is not bool:
            raise ValueError(f"invalid benchmark flag: {name}")
        if config[name]:
            argv.append("--" + name.replace("_", "-"))
    groups = config["packet_groups"]
    if not isinstance(groups, (list, tuple)) or not 1 <= len(groups) <= 4096:
        raise ValueError("invalid benchmark packet groups")
    for group in groups:
        if not isinstance(group, (list, tuple)) or len(group) != 2 or any(type(v) is not int for v in group):
            raise ValueError("invalid benchmark packet group")
        argv.extend(("--packet-group", f"{group[0]}:{group[1]}"))
    return parse_args(argv)


class ProducerStartupFailed(RuntimeError):
    """B confirmed native registration rollback before reporting failure."""
    def __init__(self, session, detail):
        self.session = session
        super().__init__("producer registration failed; source cleanup confirmed:\n" + detail)


def receive(channel, expected):
    kind, payload = channel.recv()
    if expected == "connected" and kind == "startup_failed":
        if (not isinstance(payload, dict) or payload.get("cleanup") != "done" or
                not isinstance(payload.get("session"), str) or not isinstance(payload.get("detail"), str)):
            raise RuntimeError("invalid producer startup cleanup acknowledgement")
        raise ProducerStartupFailed(payload["session"], payload["detail"])
    if kind != expected:
        raise RuntimeError(f"expected {expected}, received {kind}: {payload}")
    return payload


def batch_receipt(ready, iteration):
    """Bind completion to the exact registration/MR/batch, not just the loop ID."""
    result = dict(request_id=iteration, source_id=ready.source.source_id,
                  source_epoch=ready.source.source_epoch, sequence=ready.sequence, mr_id=ready.mr_id)
    if any(type(value) is not int or value < (0 if name == "request_id" else 1)
           for name, value in result.items()):
        raise ValueError("invalid NPU batch identity")
    return result


def matches_receipt(received, expected):
    return (isinstance(received, dict) and received == expected and
            all(type(value) is int for value in received.values()))


def open_reader_channel(args):
    from kvaf.tcp_control import TcpControlChannel

    if args.role == "reader":
        return TcpControlChannel.connect(args.control_connect, timeout=args.control_timeout), None
    address = args.control_listen
    if address.startswith("0.0.0.0:"):
        address = "127.0.0.1:" + address.rsplit(":", 1)[1]
    worker = mp.get_context("spawn").Process(target=producer_main, args=(args,))
    worker.start()
    try:
        return TcpControlChannel.connect(address, timeout=args.control_timeout), worker
    except BaseException:
        # No HELLO has been sent, so B cannot have registered HBM yet.
        worker.terminate()
        worker.join(10)
        raise


def producer_main(args):
    from kvaf.tcp_control import TcpControlListener

    with TcpControlListener(args.control_listen, timeout=args.control_timeout) as listener:
        print(f"NPU_HBM_TCP_LISTEN role=producer address={listener.address}", flush=True)
        with listener.accept() as channel:
            hello = receive(channel, "hello")
            if (not isinstance(hello, dict) or hello.get("version") != 1 or
                    not isinstance(hello.get("session"), str) or len(hello["session"]) != 32):
                raise ValueError("unsupported benchmark notification protocol")
            args = configured_producer(args, hello["config"])
            producer(channel, args, packet_layout(args), hello["receiver"], session=hello["session"])


@dataclass(frozen=True)
class PacketGroup:
    packet_bytes: int
    source_base: int
    source_bytes: int
    output_base: int
    offsets: tuple


@dataclass(frozen=True)
class PacketLayout:
    groups: tuple
    sources: tuple
    destinations: tuple
    lengths: tuple


def packet_layout(args):
    """Disjoint source regions; group-contiguous output; interleaved READ segments."""
    groups = []
    payload_prefix = source_base = 0
    alignment = 0
    for _, size in args.packet_groups:
        alignment = gcd(alignment, size)
    for index, (count, size) in enumerate(args.packet_groups):
        payload_end = payload_prefix + count * size
        source_end = args.source_space_bytes * payload_end // args.payload_bytes
        if index + 1 < len(args.packet_groups):
            source_end -= source_end % alignment
        region_bytes = source_end - source_base
        offsets = tuple(scattered_offsets(count, size, region_bytes, args.seed + index))
        groups.append(PacketGroup(size, source_base, region_bytes, GUARD_BYTES + payload_prefix, offsets))
        payload_prefix, source_base = payload_end, source_end
    sources, destinations, lengths = [], [], []
    # Interleave sizes so contiguous core partitions each receive both groups.
    for row in range(max(len(group.offsets) for group in groups)):
        for group in groups:
            if row < len(group.offsets):
                sources.append(group.source_base + group.offsets[row])
                destinations.append(group.output_base + row * group.packet_bytes)
                lengths.append(group.packet_bytes)
    return PacketLayout(tuple(groups), tuple(sources), tuple(destinations), tuple(lengths))


def round_counts(args):
    normal = args.warmup + args.iterations
    return normal, normal + (args.profile_warmup + args.profile_iterations if args.profile_iterations else 0)


def pattern_phase(iteration, source_count):
    # Different tensors have different data; each reuse cycle changes it again.
    return 2 * (iteration % source_count) + (iteration // source_count) % 2


def raw_profile_sample(result, elapsed_us):
    """Split one profiled call; keep nested totals and disjoint stages distinct."""
    profile = result["profile"]
    if "binding_total_us" not in profile:
        raise RuntimeError("raw READ binding timings missing; rebuild the NPU Python extension")
    sample = {name[:-3]: value for name, value in profile.items()
              if name.endswith("_us") and value is not None}
    sample["python_total"] = elapsed_us
    # Use the same call's measurements, not the separate unprofiled bandwidth run.
    # Preserve signed residuals so accounting/clock problems cannot be hidden.
    sample["binding_native_other"] = profile["binding_native_us"] - profile["host_total_us"]
    sample["python_and_binding_other"] = elapsed_us - profile["binding_total_us"]
    return sample


def measure_raw_read(runtime, ready, src, dst, lengths, output, stream, *, profile=False, validate_ranges=False):
    """Measure through stream completion, including the caller's async wait."""
    begin = time.perf_counter_ns()
    result = runtime.batch_get_raw_addr(ready, src, dst, lengths, output, stream=stream,
                                       profile=profile, validate_ranges=validate_ranges, synchronize=profile)
    if not profile:
        stream.synchronize()
    elapsed_us = (time.perf_counter_ns() - begin) / 1000
    runtime.drain_raw_addr(ready)  # Retire metadata holders after completion; outside the READ metric.
    return result, elapsed_us


def capture_raw_read(runtime, graph, ready, src, dst, lengths, output, stream, *, validate_ranges=False):
    """Prepare resources and pin one graph lease outside capture."""
    runtime.batch_get_raw_addr(ready, src, dst, lengths, output, stream=stream,
                               profile=False, validate_ranges=validate_ranges, synchronize=False)
    stream.synchronize()
    runtime.drain_raw_addr(ready)
    lease = runtime.begin_raw_graph(ready, src, dst, lengths, output, stream=stream)
    try:
        graph.capture_begin(stream)
        try:
            runtime.batch_get_raw_addr(ready, src, dst, lengths, output, stream=stream,
                                       profile=False, validate_ranges=validate_ranges,
                                       synchronize=False, graph_lease=lease)
        finally:
            graph.capture_end()
    except BaseException:
        graph.reset()
        runtime.end_raw_graph(lease)
        raise
    return lease


def measure_graph_read(graph, stream):
    """Include both replay submission and actual READ completion in bandwidth."""
    begin = time.perf_counter_ns()
    graph.replay(stream)
    stream.synchronize()
    return (time.perf_counter_ns() - begin) / 1000


def packet_payload(torch, offsets, packet_bytes, phase):
    # Same row/byte pattern family as BatchGet, with distinct source-slot IDs
    # and alternating contents so stale output cannot pass the next round.
    rows = torch.tensor([(offset // packet_bytes) % 256 for offset in offsets], dtype=torch.int64).unsqueeze(1)
    columns = torch.arange(packet_bytes, dtype=torch.int64).unsqueeze(0)
    return ((rows * 131 + columns * 17 + (columns >> 3) + phase * 31) & 255).to(torch.uint8)


def layout_digest(layout):
    return hashlib.sha256(repr((layout.sources, layout.destinations, layout.lengths)).encode("ascii")).hexdigest()[:16]


def byte_mismatch(actual, expected, *, limit=8):
    """Bounded report from CPU byte buffers; no NPU comparison/gather needed."""
    actual, expected = memoryview(actual).cast("B"), memoryview(expected).cast("B")
    if len(actual) != len(expected):
        return f"actual_bytes={len(actual)} expected_bytes={len(expected)}"
    differences = []
    for offset, (got, want) in enumerate(zip(actual, expected)):
        if got != want:
            differences.append(f"{offset}:{got:02x}/{want:02x}")
            if len(differences) == limit:
                break
    if not differences:
        return "host_equal=1"
    first = int(differences[0].split(":", 1)[0])
    end = min(first + 32, len(actual))
    return (f"host_equal=0 first_offset={first} "
            f"actual_hex={actual[first:end].hex()} expected_hex={expected[first:end].hex()} "
            f"differences_offset_actual_expected={','.join(differences)}")


def verify_source_packets(torch, hbm, payloads, layout, iteration, mr_id, registered_address):
    # Diagnostic only: the source-space copy and CPU verification are outside
    # timing, and their payload never enters the TCP control channel.
    if hbm.data_ptr() != registered_address:
        raise RuntimeError(f"NPU producer storage changed at iteration {iteration}, mr_id={mr_id}")
    source = hbm.cpu()
    for group_index, (group, payload) in enumerate(zip(layout.groups, payloads)):
        for row, offset in enumerate(group.offsets):
            start = group.source_base + offset
            actual = source[start:start + group.packet_bytes]
            expected = payload[row]
            if not torch.equal(actual, expected):
                detail = byte_mismatch(actual.numpy(), expected.numpy())
                raise RuntimeError(f"NPU producer write mismatch at iteration {iteration}, mr_id={mr_id}, "
                                   f"group={group_index}, row={row}, source_offset={start}: {detail}")
    print(f"NPU_HBM_SOURCE_CHECK_PASS iteration={iteration} mr_id={mr_id} "
          f"packets={len(layout.sources)} layout_digest={layout_digest(layout)}", flush=True)


def report_read_mismatch(torch, output_cpu, expected, expected_hbm, layout, iteration, ready, npu_equal):
    # Do not classify a failed NPU equal as payload corruption until the CPU
    # readback is compared too. Preserve the NPU consumer failure either way.
    host_equal = torch.equal(output_cpu, expected)
    expected_equal = torch.equal(expected_hbm.cpu(), expected)
    print(f"NPU_HBM_READ_MISMATCH iteration={iteration} mr_id={ready.mr_id} sequence={ready.sequence} "
          f"npu_equal={int(npu_equal)} expected_hbm_host_equal={int(expected_equal)} "
          f"layout_digest={layout_digest(layout)} {byte_mismatch(output_cpu.numpy(), expected.numpy())}", flush=True)
    actual = memoryview(output_cpu.numpy()).cast("B")
    wanted = memoryview(expected.numpy()).cast("B")
    print(f"NPU_HBM_GUARD_CHECK prefix_ok={int(actual[:GUARD_BYTES] == wanted[:GUARD_BYTES])} "
          f"suffix_ok={int(actual[-GUARD_BYTES:] == wanted[-GUARD_BYTES:])}", flush=True)
    if not host_equal:
        for request, (src, dst, length) in enumerate(zip(layout.sources, layout.destinations, layout.lengths)):
            if actual[dst:dst + length] != wanted[dst:dst + length]:
                detail = byte_mismatch(actual[dst:dst + length], wanted[dst:dst + length])
                print(f"NPU_HBM_PACKET_MISMATCH request={request} source_offset={src} "
                      f"output_offset={dst} packet_bytes={length} {detail}", flush=True)
                break


def producer(channel, args, layout, receiver, *, session=""):
    import torch
    import torch_npu  # noqa: F401
    import kvaf

    torch.npu.set_device(args.device_b)
    device = f"npu:{args.device_b}"
    stream = torch.npu.Stream(device=args.device_b)
    source = kvaf.NpuPullHbmSource(args.device_b, args.npu_eid_b_raw)
    ready = None
    registered = False
    shutdown_authorized = False
    stages = {"ready": [], "release_batch": []}
    try:
        tensors = [torch.zeros(args.source_space_bytes, dtype=torch.uint8, device=device)
                   for _ in range(args.source_count)]
        registered_addresses = [tensor.data_ptr() for tensor in tensors]
        indices = [torch.tensor([offset // group.packet_bytes for offset in group.offsets],
                                dtype=torch.int64, device=device) for group in layout.groups]
        torch.npu.synchronize()
        source.register(tensors if args.source_count > 1 else tensors[0], receiver=receiver, stream=stream)
        registered = True
        channel.send(("connected", session))
        normal_rounds, total_rounds = round_counts(args)
        for iteration in range(total_rounds):
            request = receive(channel, "request")
            if type(request) is not int or request != iteration:
                raise RuntimeError("request notification belongs to another round")
            hbm = tensors[iteration % args.source_count]
            phase = pattern_phase(iteration, args.source_count)
            cpu_payloads = [packet_payload(torch, group.offsets, group.packet_bytes, phase + index * 17)
                            for index, group in enumerate(layout.groups)]
            payloads = [payload.to(device) for payload in cpu_payloads]
            torch.npu.synchronize()
            with torch.npu.stream(stream):
                for group, index, payload in zip(layout.groups, indices, payloads):
                    slots = group.source_bytes // group.packet_bytes
                    rows = hbm[group.source_base:group.source_base + slots * group.packet_bytes].view(slots, group.packet_bytes)
                    rows.index_copy_(0, index, payload)
            stream.synchronize()
            if args.verify_source:
                mr = iteration % args.source_count
                verify_source_packets(torch, hbm, cpu_payloads, layout, iteration, mr + 1, registered_addresses[mr])
            begin = time.perf_counter_ns()
            ready = source.ready(hbm) if args.source_count > 1 else source.ready()
            ready_us = (time.perf_counter_ns() - begin) / 1000
            receipt = batch_receipt(ready, iteration)
            channel.send(("ready", (iteration, ready)))
            # The application owns completion notification; Runtime only reads.
            if not matches_receipt(receive(channel, "reuse"), receipt):
                raise RuntimeError("reuse notification belongs to another batch")
            begin = time.perf_counter_ns()
            source.release_batch(ready)
            release_us = (time.perf_counter_ns() - begin) / 1000
            if args.warmup <= iteration < normal_rounds:
                stages["ready"].append(ready_us)
                stages["release_batch"].append(release_us)
            channel.send(("released", receipt))
        if receive(channel, "shutdown") != session:
            raise RuntimeError("shutdown notification belongs to another session")
        shutdown_authorized = True
        source.shutdown()
        print_stages("NPU_HBM_BENCH_PROFILE", "producer", stages, args.warmup)
        channel.send(("closed", session))
    except BaseException:
        detail = traceback.format_exc()
        if not registered:
            # A failed register may have created resources or run READ probes.
            # Only native Shutdown success proves rollback; never infer it from
            # an error code, a disconnected socket, or frozen/connected flags.
            try:
                source.shutdown()
            except Exception as error:
                try:
                    channel.send(("error", detail + "\nSource cleanup remains unresolved: " + str(error)))
                except Exception:
                    pass
                retain(str(error))
                raise
            try:
                channel.send(("startup_failed", dict(session=session, cleanup="done", detail=detail)))
            except Exception:
                pass  # A must retain resources if this confirmation is lost.
            raise
        try:
            channel.send(("error", detail))
        except Exception:
            pass  # A disconnected application channel cannot authorize reuse.
        try:
            # Even an unfrozen source can still be referenced by A's captured
            # graph. Only a session shutdown after graph reset authorizes close.
            if registered and not shutdown_authorized:
                retain("TCP session failed without reader shutdown; retain NPU source")
            source.shutdown()
        except Exception as error:
            retain(str(error))
        raise


def reader(args):
    import torch
    import torch_npu  # noqa: F401
    import kvaf

    diagnostics = os.environ.get("KVAF_HBM_RAW_READ_DIAGNOSTICS") == "1"
    layout = packet_layout(args)
    payload_bytes = args.payload_bytes
    group_description = ",".join(f"{count}x{size}" for count, size in args.packet_groups)
    torch.npu.set_device(args.device_a)
    device = f"npu:{args.device_a}"
    runtime = kvaf.NpuRuntime(
        device_id=args.device_a,
        npu_pull_options=kvaf.NpuPullOptions(kernel_path=args.raw_kernel,
            cores=args.aiv_cores, jettys=args.jettys_per_core,
            max_inflight_total=args.max_inflight_total))
    worker = None
    channel = None
    peer_active = False
    graphs = {}
    graph_leases = {}
    try:
        runtime.initialize()
        assert runtime.initialized and not runtime.ddr_initialized
        receiver = runtime.listen_hbm(args.hbm_listen, advertise_address=args.hbm_advertise,
                                      local_eid=args.npu_eid_a_raw)
        channel, worker = open_reader_channel(args)
        session = uuid.uuid4().hex
        peer_active = True
        channel.send(("hello", dict(version=1, session=session, receiver=receiver, config=benchmark_config(args))))
        try:
            connected_session = receive(channel, "connected")
        except ProducerStartupFailed as error:
            if error.session == session:
                peer_active = False
            raise
        if connected_session != session:
            raise RuntimeError("registration acknowledgement belongs to another session")
        stream = torch.npu.Stream(device=args.device_a)
        with torch.npu.stream(stream):
            output = torch.empty(payload_bytes + 2 * GUARD_BYTES, dtype=torch.uint8, device=device)
            src = torch.tensor(layout.sources, dtype=torch.int64, device=device)
            dst = torch.tensor(layout.destinations, dtype=torch.int64, device=device)
            lengths = torch.tensor(layout.lengths, dtype=torch.int64, device=device)
        stream.synchronize()
        print(f"NPU_HBM_BENCH_READY reader={args.device_a} producer={args.device_b} "
              f"packets={args.packets} packet_bytes={args.packet_bytes if args.packet_bytes is not None else 'mixed'} "
              f"packet_groups={group_description} packet_order=round_robin_groups payload_bytes={payload_bytes} "
              f"source_count={args.source_count} source_space_bytes_per_mr={args.source_space_bytes} seed={args.seed} "
              f"min_source_offset={min(layout.sources)} "
              f"max_source_end={max(offset + size for offset, size in zip(layout.sources, layout.lengths))} "
              f"aiv_cores={args.aiv_cores} jettys_per_core={args.jettys_per_core} "
              f"max_inflight_total={args.max_inflight_total} "
              f"validate_ranges={int(args.validate_ranges)} output_registration_cache=1 "
              f"verify_source={int(args.verify_source)} layout_digest={layout_digest(layout)} "
              "source_pattern=random_disjoint_slots output_pattern=contiguous metadata=npu_int64 "
              "payload_uses_cpu_ddr=0 ddr_initialized=0 cpu_service_required=0 control_listener=producer "
              "hbm_control_listener=reader acquire_raw=0 "
              f"status_reset={'diagnostic_each_call' if diagnostics else 'init_only'} status_generation=0 "
              f"status_readback={'diagnostic' if diagnostics else 'none'} completion=caller_stream_sync "
              f"execution={'graph' if args.graph else 'eager'} graph={int(args.graph)} binding_profile={int(not args.graph)} "
              f"completion_notification=application application_transport=tcp role={args.role} "
              "batch_control_rpcs=0", flush=True)
        read_stage = "graph_replay" if args.graph else "batch_get_raw_addr"
        stages = {read_stage: [], "verification": []}
        internal_stages = {}
        event_errors = set()
        normal_rounds, total_rounds = round_counts(args)
        registration = None
        last_sequence = 0
        for iteration in range(total_rounds):
            channel.send(("request", iteration))
            request_id, ready = receive(channel, "ready")
            if type(request_id) is not int or request_id != iteration or ready.mr_id != iteration % args.source_count + 1:
                raise RuntimeError("ready notification belongs to another request or MR")
            receipt = batch_receipt(ready, iteration)
            identity = receipt["source_id"], receipt["source_epoch"]
            if (registration is not None and identity != registration) or receipt["sequence"] <= last_sequence:
                raise RuntimeError("ready notification changed source or replayed an old batch")
            registration, last_sequence = identity, receipt["sequence"]
            expected = torch.full((output.numel(),), 0xCC, dtype=torch.uint8)
            phase = pattern_phase(iteration, args.source_count)
            for index, group in enumerate(layout.groups):
                group_bytes = len(group.offsets) * group.packet_bytes
                expected[group.output_base:group.output_base + group_bytes] = packet_payload(
                    torch, group.offsets, group.packet_bytes, phase + index * 17).flatten()
            expected_hbm = expected.to(device)
            torch.npu.synchronize()
            if args.graph and ready.mr_id not in graphs:
                # This source's MRs have distinct fixed device descriptors. Keep
                # one graph per MR, sharing stable output and metadata buffers.
                graph = kvaf.AclGraph()
                graphs[ready.mr_id] = graph
                graph_leases[ready.mr_id] = capture_raw_read(runtime, graph, ready, src, dst, lengths, output, stream,
                                 validate_ranges=args.validate_ranges)
            # Clear even after capture warmup so stale warmup output cannot pass.
            with torch.npu.stream(stream):
                output.fill_(0xCC)
            stream.synchronize()
            instrumented = iteration >= normal_rounds
            if args.graph:
                elapsed_us = measure_graph_read(graphs[ready.mr_id], stream)
            else:
                result, elapsed_us = measure_raw_read(runtime, ready, src, dst, lengths, output, stream,
                                                     profile=instrumented, validate_ranges=args.validate_ranges)
            verify_begin = time.perf_counter_ns()
            npu_equal = torch.equal(output, expected_hbm)
            output_cpu = output.cpu()
            host_equal = torch.equal(output_cpu, expected)
            if not npu_equal or not host_equal:
                try:
                    report_read_mismatch(torch, output_cpu, expected, expected_hbm, layout, iteration, ready, npu_equal)
                    for name, tensor, values in (("source", src, layout.sources),
                                                 ("destination", dst, layout.destinations),
                                                 ("length", lengths, layout.lengths)):
                        actual_values = tuple(tensor.cpu().tolist())
                        print(f"NPU_HBM_METADATA_CHECK array={name} matches={int(actual_values == values)} "
                              f"actual_head={actual_values[:8]} expected_head={values[:8]}", flush=True)
                except Exception as error:
                    print(f"NPU_HBM_MISMATCH_DIAGNOSTIC_FAILED error={error}", flush=True)
                kind = "NPU consumer" if not npu_equal else "payload/guard"
                raise RuntimeError(f"{kind} mismatch at iteration {iteration}")
            verify_us = (time.perf_counter_ns() - verify_begin) / 1000
            if args.warmup <= iteration < normal_rounds:
                stages[read_stage].append(elapsed_us)
                stages["verification"].append(verify_us)
            elif iteration >= normal_rounds + args.profile_warmup:
                for name, value in raw_profile_sample(result, elapsed_us).items():
                    internal_stages.setdefault(name, []).append(value)
                if result["profile"]["kernel_event_us"] is None:
                    event_errors.add(result["profile"]["kernel_event_error"])
            channel.send(("reuse", receipt))
            if not matches_receipt(receive(channel, "released"), receipt):
                raise RuntimeError("release acknowledgement belongs to another batch")
        capture_count = len(graphs)
        for mr_id, graph in graphs.items():
            graph.reset()
            runtime.end_raw_graph(graph_leases.pop(mr_id))
        graphs.clear()
        channel.send(("shutdown", session))
        if receive(channel, "closed") != session:
            raise RuntimeError("close acknowledgement belongs to another session")
        peer_active = False
        channel.close()
        if worker is not None:
            worker.join(10)
            if worker.exitcode != 0:
                raise RuntimeError("producer did not exit successfully after shutdown")
        runtime.shutdown()
        print_stages("NPU_HBM_BENCH_PROFILE", "reader", stages, args.warmup)
        if args.profile_iterations:
            print_stages("NPU_HBM_INTERNAL_PROFILE", "reader", internal_stages, args.profile_warmup)
            print("NPU_HBM_INTERNAL_PROFILE_INFO kernel_timing=device_events "
                  f"kernel_event_samples={len(internal_stages.get('kernel_event', []))} "
                  f"profile_iterations={args.profile_iterations} event_errors={sorted(event_errors)} "
                  "device_time_overlaps_host_launch_and_synchronize=1 instrumented_rounds_excluded_from_bandwidth=1",
                  flush=True)
        stats = summarize_us(stages[read_stage])
        metric = "graph_replay_to_stream_sync" if args.graph else "python_batch_get_raw_addr_to_stream_sync"
        print(f"NPU_HBM_BENCH_PASS metric={metric} graph={int(args.graph)} "
              f"graph_captures={capture_count} graph_setup_excluded=1 "
              f"packets={args.packets} packet_groups={group_description} payload_bytes={payload_bytes} "
              f"source_count={args.source_count} iterations={stats['samples']} warmup_excluded={args.warmup} "
              f"mean_us={stats['mean_us']:.3f} p50_us={stats['p50_us']:.3f} "
              f"p95_us={stats['p95_us']:.3f} p99_us={stats['p99_us']:.3f} "
              f"effective_GBps={payload_bytes / stats['mean_us'] / 1000:.3f} "
              "return_condition=caller_stream_complete drain_cleanup_excluded=1 completion_notification=application application_transport=tcp "
              f"profile_iterations={args.profile_iterations} instrumented_rounds_excluded_from_bandwidth=1 "
              f"validate_ranges={int(args.validate_ranges)} output_registration_cache=1 "
              "percentile=linear verification_included=0 setup_and_producer_and_notifications_excluded=1 "
              "changing_inputs=1 verification=every_round_full_output npu_consumer_checked=1 guard_checked=1",
              flush=True)
    except BaseException:
        traceback.print_exc()
        if channel is not None:
            channel.close()
        if graphs or runtime.pending_raw_addr or peer_active:
            retain("NPU HBM benchmark output or source remains retained")
        runtime.shutdown()
        raise


def main():
    args = parse_args()
    if args.role == "producer":
        producer_main(args)
    else:
        reader(args)


if __name__ == "__main__":
    main()
