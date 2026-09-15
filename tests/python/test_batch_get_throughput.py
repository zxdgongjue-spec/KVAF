#!/usr/bin/env python3
"""Portable timing, synchronization and failure tests for continuous BatchGet."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import test_npu_batch_get_throughput as throughput


class FakeWorkload:
    def __init__(self, args, rank):
        self.args, self.rank = args, rank
        self.events = []

    def prepare(self):
        if getattr(self.args, "fail_setup", -1) == self.rank:
            raise RuntimeError("injected setup failure")
        eids = self.args.npu_eids or ["1" * 32]
        if getattr(self.args, "wrong_eid_rank", -1) == self.rank:
            eids = ["f" * 32]
        return dict(rank=self.rank, fake=True, cpu_service=self.args.cpu_service,
                    requested_npu_eids=self.args.npu_eids,
                    selected_path=dict(npu_eids=eids, cpu_eid="a" * 32, cpu_candidate=(
                        self.rank if getattr(self.args, "different_paths", False) else 0)))

    def arm(self):
        self.events.append(dict(event="arm", time_ns=time.monotonic_ns()))

    def measure(self):
        # A slower second worker exercises the wait before validation.
        result = throughput.measure_batches(
            lambda plan: None, lambda: time.sleep(0.003 + self.rank * 0.02),
            [0, 1], self.args.iterations)
        self.events.append(dict(event="measured", time_ns=result["end_ns"]))
        return result

    def validate(self):
        self.events.append(dict(event="validate", time_ns=time.monotonic_ns()))
        if getattr(self.args, "fail_validation", -1) == self.rank:
            raise RuntimeError("injected payload failure")

    def close(self):
        (self.args.trace_dir / f"{self.rank}.json").write_text(json.dumps(self.events))


def fake_worker(args, rank, connection):
    throughput.serve_worker(connection, rank, FakeWorkload(args, rank))


class BatchGetThroughputTest(unittest.TestCase):
    def args(self, *options):
        with patch.dict(os.environ, {}, clear=True):
            return throughput.parse_args([
                "--build-dir", "/mock/build", "--request-length", "16",
                "--miss-count", "8", "--token-bytes", "16", "--iterations", "3",
                "--repeats", "2", "--timeout-seconds", "15", *options])

    def test_submission_bank_reuse_and_one_final_sync(self):
        events = []
        times = iter([100, 300, 700])
        row = throughput.measure_batches(
            lambda plan: events.append(plan), lambda: events.append("sync"),
            ["a", "b"], 5, lambda: next(times))
        self.assertEqual(events, ["a", "b", "a", "b", "a", "sync"])
        self.assertEqual(row, dict(start_ns=100, submitted_ns=300, end_ns=700, batches=5))

    def test_union_window_does_not_sum_local_bandwidth(self):
        rows = [dict(rank=0, start_ns=0, submitted_ns=100, end_ns=1000, batches=1),
                dict(rank=1, start_ns=1000, submitted_ns=1100, end_ns=2000, batches=1)]
        row = throughput.aggregate_window(rows, 1000)
        self.assertEqual(row["aggregate_GBps"], 1.0)
        self.assertEqual(row["host_window_overlap_fraction"], 0)
        self.assertEqual(row["payload_bytes"], 2000)
        self.assertEqual(sum(w["common_window_GBps"] for w in row["workers"]), 1.0)
        rows[1].update(start_ns=250, submitted_ns=300, end_ns=1500)
        row = throughput.aggregate_window(rows, 1000)
        self.assertEqual(row["host_window_overlap_fraction"], 0.5)
        self.assertEqual(row["start_skew_us"], 0.25)
        self.assertEqual(row["finish_skew_us"], 0.5)
        self.assertEqual(throughput.aggregate_window(rows, 0)["aggregate_GBps"], 0)

    def test_weighted_summary_and_zero_payload(self):
        rows = [dict(phase="single", start_ns=0, end_ns=100, payload_bytes=100, aggregate_GBps=1),
                dict(phase="single", start_ns=0, end_ns=300, payload_bytes=100, aggregate_GBps=1 / 3)]
        self.assertEqual(throughput.summarize(rows)["single"]["aggregate_GBps"], 0.5)
        zero = [dict(phase=phase, start_ns=0, end_ns=100, payload_bytes=0, aggregate_GBps=0)
                for phase in ("solo_0", "solo_1", "concurrent")]
        self.assertIsNone(throughput.summarize(zero)["concurrent"]["speedup_vs_best_solo"])
        self.assertIsNone(throughput.summarize(zero)["concurrent"]["fraction_of_solo_sum"])
        self.assertIsNone(throughput.summarize(zero)["concurrent"]["fraction_of_independent_equal_work"])
        asymmetric = [dict(phase=phase, start_ns=0, end_ns=100,
                           payload_bytes=bandwidth * 100, aggregate_GBps=bandwidth)
                      for phase, bandwidth in (("solo_0", 10), ("solo_1", 20), ("concurrent", 24))]
        combined = throughput.summarize(asymmetric)["concurrent"]
        self.assertEqual(combined["speedup_vs_best_solo"], 1.2)
        self.assertEqual(combined["solo_sum_GBps"], 30)
        self.assertEqual(combined["fraction_of_solo_sum"], 0.8)
        self.assertEqual(combined["independent_equal_work_GBps"], 20)
        # Two equal 1000-byte jobs at 10 and 20 GB/s finish in 100 ns, so
        # 20 GB/s aggregate means no slowdown despite being below 10 + 20.
        independent = [dict(phase=phase, start_ns=0, end_ns=duration,
                            payload_bytes=payload, aggregate_GBps=payload / duration)
                       for phase, duration, payload in (("solo_0", 100, 1000),
                                                        ("solo_1", 50, 1000),
                                                        ("concurrent", 100, 2000))]
        combined = throughput.summarize(independent)["concurrent"]
        self.assertEqual(combined["fraction_of_solo_sum"], 2 / 3)
        self.assertEqual(combined["fraction_of_independent_equal_work"], 1.0)
        with self.assertRaises(ValueError):
            throughput.aggregate_window([dict(start_ns=2, submitted_ns=1, end_ns=3, batches=1)], 10)

    def test_configuration_boundaries(self):
        args = self.args("--process-count", "2", "--stream-count", "2", "--transport-count", "2")
        self.assertIsNone(throughput.validate_args(args))
        for name, value in (("transport_count", 3), ("aiv_cores", 25),
                            ("jettys_per_core", 0), ("request_bank_size", 0),
                            ("iterations", 0), ("repeats", 0), ("timeout_seconds", float("nan")),
                            ("miss_count", 17), ("tokens_per_block", 1 << 31)):
            old = getattr(args, name)
            setattr(args, name, value)
            self.assertIsNotNone(throughput.validate_args(args), name)
            setattr(args, name, old)
        args.stream_count = 1
        self.assertIsNotNone(throughput.validate_args(args))

    def test_selected_path_uses_successful_final_candidate(self):
        log = ("[KVAF_AIV] PEER_LINK_EID_PAIR_ATTEMPT peer_candidate=0/2 npu_eid=" + "1" * 32 + "\n" +
               "[KVAF_AIV] PEER_LINK_EID_GROUP_SELECTED peer_candidate=1/2 npu_eids=" +
               "2" * 32 + "," + "3" * 32 + "\n")
        self.assertEqual(throughput.selected_path(log),
                         dict(npu_eids=["2" * 32, "3" * 32], cpu_candidate=1, cpu_eid=None))
        with self.assertRaises(RuntimeError):
            throughput.selected_path("initialization failed")

    def test_cpu_eid_belongs_to_final_successful_attempt(self):
        log = ("PEER_LINK_EID_PAIR_ATTEMPT peer_candidate=0/2\n"
               "DIRECT_AIV_WQE_EID_ENCODING ra_host_eid=" + "a" * 32 + "\n"
               "PEER_LINK_EID_PAIR_ATTEMPT peer_candidate=1/2\n"
               "DIRECT_AIV_WQE_EID_ENCODING ra_host_eid=" + "b" * 32 + "\n"
               "PEER_LINK_EID_PAIR_SELECTED peer_candidate=1/2 npu_eid=" + "1" * 32 + "\n")
        self.assertEqual(throughput.selected_path(log),
                         dict(npu_eids=["1" * 32], cpu_candidate=1, cpu_eid="b" * 32))
        missing = log.replace("DIRECT_AIV_WQE_EID_ENCODING ra_host_eid=" + "b" * 32 + "\n", "")
        self.assertIsNone(throughput.selected_path(missing)["cpu_eid"])

    def test_per_process_route_options_and_precedence(self):
        eids = ["1" * 32, "2" * 32]
        args = self.args("--process-count", "2", "--npu-eids", "3" * 32,
                         "--cpu-service", "shared:123", "--process-npu-eids", *eids,
                         "--process-cpu-services", "cpu-a:123", "cpu-b:456")
        for rank in (0, 1):
            worker = throughput.worker_options(args, rank)
            self.assertEqual(worker.npu_eids, [eids[rank]])
            self.assertEqual(worker.cpu_service, ["cpu-a:123", "cpu-b:456"][rank])
        self.assertEqual(args.npu_eids, ["3" * 32])
        self.assertEqual(args.cpu_service, "shared:123")
        args.process_npu_eids = None
        self.assertEqual(throughput.worker_options(args, 1).npu_eids, ["3" * 32])
        args.process_cpu_services = None
        self.assertEqual(throughput.routing_mode(args), "shared")
        self.assertEqual(throughput.worker_options(args, 1).cpu_service, "shared:123")
        args.process_npu_eids = [[eids[0]]]
        self.assertIsNotNone(throughput.validate_args(args))
        args.process_npu_eids = [[eids[0]], eids]
        args.aiv_cores = 1
        self.assertIn("process 1", throughput.validate_args(args))
        args.process_npu_eids = [[eids[0]], []]
        args.process_cpu_services = ["cpu-a:123", " "]
        self.assertIn("process 1", throughput.validate_args(args))

    def test_initialization_captures_native_log_and_preserves_errors(self):
        class Runtime:
            def initialize(self, ddr, block):
                os.write(2, ("[KVAF_AIV] PEER_LINK_EID_PAIR_SELECTED peer_candidate=0/1 "
                             "npu_eid=" + "1" * 32 + "\n").encode())
                if ddr == 0:
                    raise RuntimeError("initialization failed")

        log = io.StringIO()
        with contextlib.redirect_stderr(log):
            self.assertEqual(throughput.initialize_with_log(Runtime(), 1, 1, 0),
                             dict(npu_eids=["1" * 32], cpu_candidate=0, cpu_eid=None))
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                throughput.initialize_with_log(Runtime(), 0, 1, 1)
        self.assertIn("INITIALIZE rank=0", log.getvalue())
        self.assertIn("INITIALIZE rank=1", log.getvalue())

    def test_spawned_processes_and_postflight_barrier(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args("--process-count", "2")
            args.trace_dir = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                report = throughput.run_benchmark(args, target=fake_worker)
            self.assertEqual([row["phase"] for row in report["results"]],
                             ["solo_0", "solo_1", "concurrent", "solo_1", "concurrent", "solo_0"])
            for row in report["results"]:
                self.assertEqual(row["payload_bytes"], len(row["workers"]) * 3 * 8 * 16)
                self.assertEqual(row["total_jettys"], len(row["workers"]) * 2)
            traces = {rank: json.loads((Path(directory) / f"{rank}.json").read_text()) for rank in (0, 1)}
            for rank, events in traces.items():
                windows = [row for row in report["results"] if any(w["rank"] == rank for w in row["workers"])]
                validations = [event["time_ns"] for event in events if event["event"] == "validate"]
                self.assertEqual(len(validations), len(windows))
                for validation, window in zip(validations, windows):
                    self.assertGreaterEqual(validation, window["end_ns"])
            self.assertFalse(report["kernel_profile"])

    def test_spawned_workers_use_individual_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args("--process-count", "2", "--repeats", "1",
                             "--process-npu-eids", "1" * 32, "2" * 32)
            args.trace_dir = Path(directory)
            for services in (None, ["cpu-a:123", "cpu-b:456"]):
                args.process_cpu_services = services
                # Candidate IDs are service-local; this difference is allowed
                # when routes were explicitly configured per process.
                args.different_paths = services is not None
                with contextlib.redirect_stdout(io.StringIO()):
                    report = throughput.run_benchmark(args, target=fake_worker)
                self.assertEqual(report["routing_mode"], "per_process")
                workers = report["results"][-1]["workers"]
                for rank, row in enumerate(workers):
                    self.assertEqual(row["requested_npu_eids"], [str(rank + 1) * 32])
                    self.assertEqual(row["selected_path"]["npu_eids"], [str(rank + 1) * 32])
                    self.assertEqual(row["cpu_service"], services[rank] if services else args.cpu_service)
                self.assertEqual(sum(row["common_window_GBps"] for row in workers),
                                 report["results"][-1]["aggregate_GBps"])
            args.wrong_eid_rank = 1
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "instead of requested"):
                throughput.run_benchmark(args, target=fake_worker)

    def test_single_process_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args("--repeats", "1")
            args.trace_dir = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                report = throughput.run_benchmark(args, target=fake_worker)
            self.assertEqual(list(report["summary"]), ["single"])
            args.process_count = 2
            for fault in ("fail_setup", "fail_validation"):
                setattr(args, fault, 1)
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "injected"):
                    throughput.run_benchmark(args, target=fake_worker)
                delattr(args, fault)
            args.different_paths = True
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "different NPU EIDs"):
                throughput.run_benchmark(args, target=fake_worker)


if __name__ == "__main__":
    unittest.main()
