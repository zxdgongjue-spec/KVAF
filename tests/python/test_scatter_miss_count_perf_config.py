"""Reference alignment, exact accounting and fail-fast sweep checks without NPU."""
import contextlib
import ast
import csv
import io
import json
import inspect
import os
from pathlib import Path
import random
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_aiv_scatter_miss_count_perf as perf
import test_npu_batch_get_perf as reference
from test_aiv_scatter_perf import write_results
from test_scatter_perf_metrics import profiles
from scatter_perf_common import HOST_TIMINGS


class Vector:
    """Small integer Tensor double with slice views; no numpy/torch dependency."""
    def __init__(self, values, backing=None, positions=None):
        self.backing = list(values) if backing is None else backing
        self.positions = list(range(len(self.backing))) if positions is None else positions
    def values(self):
        return [self.backing[i] for i in self.positions]
    def __getitem__(self, key):
        if isinstance(key, slice):
            return Vector([], self.backing, self.positions[key])
        return self.backing[self.positions[key]]
    def __add__(self, other):
        return Vector(a + b for a, b in zip(self.values(), other.values()))
    def __mod__(self, value):
        return Vector(x % value for x in self.values())
    def __floordiv__(self, value):
        return Vector(x // value for x in self.values())
    def __mul__(self, value):
        return Vector(x * value for x in self.values())
    def to(self, dtype):
        return self
    def clone(self):
        return Vector(self.values())
    def copy_(self, source):
        assert len(self.positions) == len(source.positions)
        for i, value in zip(self.positions, source.values()):
            self.backing[i] = value
        return self
    def index_select(self, dim, indices):
        assert dim == 0
        return Vector(self[i] for i in indices.values())


class Torch:
    int32, int64 = "int32", "int64"
    def __init__(self, seed=7):
        self.rng = random.Random(seed)
        self.ranges = []
    def randint(self, low, high, shape, **kwargs):
        self.ranges.append((low, high, shape))
        return Vector(self.rng.randrange(low, high) for _ in range(shape[0]))
    def randperm(self, count, **kwargs):
        values = list(range(count)); self.rng.shuffle(values)
        return Vector(values)
    def sort(self, keys):
        order = sorted(range(len(keys.positions)), key=lambda i: keys[i])
        return Vector(keys[i] for i in order), Vector(order)


class AlignedScatterTests(unittest.TestCase):
    def parse(self, *args):
        with patch.dict(os.environ, {}, clear=True):
            return perf.arguments(["--build", "unused-build", *args])

    def test_reference_defaults_and_supported_sweep_match(self):
        args = self.parse()
        with patch.dict(os.environ, {}, clear=True):
            old = reference.parse_args([])
        for key in ("request_length", "table_capacity", "tokens_per_block", "warmup",
                    "iterations", "seed", "source_access_pattern", "output_access_pattern",
                    "sort_input_indices", "populate_chunk_entries", "kernel_profile_clock_mhz"):
            self.assertEqual(getattr(args, key), getattr(old, key), key)
        self.assertEqual(len(perf.cases(args)), 7 * 9 * 4)
        self.assertEqual(set(c.aiv_cores for c in perf.cases(args)), {1, 2, 4, 8})
        self.assertTrue(args.kernel_profile)
        self.assertFalse(args.host_profile)
        self.assertEqual(args.completion_mode, "reference")
        self.assertEqual(args.host_submit, "reference")
        self.assertEqual(args.table_capacity, 1277952)
        self.assertEqual(args.table_capacity * 128 * 4096, 670014898176)

    def test_scalar_axes_and_environment_round_trip(self):
        args = self.parse("--token-bytes", "656", "--miss-count", "1024", "--aiv-cores", "4",
                          "--pipeline", "0", "--host-submit", "prepared")
        self.assertEqual(len(perf.cases(args)), 1)
        case = perf.cases(args)[0]
        case.json, case.csv = Path("case.json"), Path("case.csv")
        child = self.parse(*perf.case_command(case)[4:])
        for key in ("token_bytes", "miss_count", "aiv_cores", "table_capacity",
                    "warmup", "kernel_profile", "host_profile", "completion_mode", "pipeline", "host_submit"):
            self.assertEqual(getattr(child, key), getattr(case, key))
        with patch.dict(os.environ, {"KVAF_BATCH_GET_PERF_TOKEN_BYTES": "256",
                                    "KVAF_BATCH_GET_PERF_MISS_COUNT": "64",
                                    "KVAF_BATCH_GET_PERF_AIV_CORES": "16",
                                    "KVAF_BATCH_GET_PERF_JETTYS_PER_CORE": "8",
                                    "KVAF_BATCH_GET_PERF_KERNEL_PROFILE": "0"}, clear=True):
            env = perf.arguments(["--build", "unused"])
            self.assertEqual(env.token_sizes, perf.TOKEN_SIZES)
            self.assertEqual(env.miss_counts, perf.MISS_COUNTS)
            self.assertTrue(env.kernel_profile)  # Reference shell forces this CLI flag.
            self.assertEqual(len(perf.cases(env)), 252)
            self.assertEqual({c.aiv_cores for c in perf.cases(env)}, {1, 2, 4, 8})
            self.assertEqual(env.jettys_per_core, 2)

    def test_unsupported_and_inconsistent_cases_fail_before_hardware(self):
        for args in (("--aiv-cores", "16"), ("--core-counts", "1,32"),
                     ("--jettys-per-core", "4"), ("--adaptive-ub-device",),
                     ("--miss-count", "0"), ("--request-length", "65537"),
                     ("--token-sizes", "656,656"), ("--miss-counts", "65537"),
                     ("--no-kernel-profile", "--host-profile"),
                     ("--single", "--core-counts", "1,2"),
                     ("--single", "--json", "same.log"),
                     ("--case-timeout", "0"), ("--json", "result.json")):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parse(*args)

    def test_shared_generator_matches_reference_random_domain(self):
        args = self.parse("--single", "--request-length", "32", "--miss-count", "7")
        torch = Torch()
        indices, offsets, rows = reference.generate_request_plan(torch, args, "npu", Vector(range(32)), None)
        self.assertEqual(torch.ranges, [(0, args.table_capacity, (32,)), (0, 128, (32,))])
        self.assertTrue(all(0 <= x < args.table_capacity for x in indices.values()))
        self.assertTrue(all(x % 656 == 0 and x < 128 * 656 for x in offsets.values()))
        self.assertEqual(rows.values(), list(range(32)))

    def test_contiguous_shuffled_random_output_and_sorted_prefix(self):
        args = self.parse("--single", "--table-capacity", "3", "--tokens-per-block", "4",
                          "--request-length", "16", "--miss-count", "7",
                          "--source-access-pattern", "contiguous")
        ids, addresses = Vector(range(16)), Vector([3000, 1000, 2000])
        linear = reference.generate_request_plan(Torch(), args, "npu", ids, addresses)
        def tokens(plan):
            return [i * 4 + o // args.token_bytes for i, o in zip(plan[0].values(), plan[1].values())]
        self.assertEqual(tokens(linear), [i % 12 for i in range(16)])
        args.source_access_pattern = "shuffled-contiguous"
        args.output_access_pattern = "random"
        shuffled = reference.generate_request_plan(Torch(), args, "npu", ids, addresses)
        self.assertEqual(sorted(tokens(shuffled)[:7]), list(range(7)))
        self.assertEqual(tokens(shuffled)[7:], tokens(linear)[7:])
        self.assertEqual(sorted(shuffled[2].values()), list(range(16)))
        original = list(zip(*(x.values() for x in shuffled)))
        args.sort_input_indices = True
        ordered = reference.generate_request_plan(Torch(), args, "npu", ids, addresses)
        actual = list(zip(*(x.values() for x in ordered)))
        self.assertEqual(actual[:7], sorted(original[:7], key=lambda row: addresses[row[0]] + row[1]))
        self.assertEqual(actual[7:], original[7:])

    def test_reference_call_sync_and_validation_boundaries(self):
        args = self.parse("--single", "--aiv-cores", "4", "--miss-count", "1024", "--host-profile")
        now = [0]
        events = []
        class Runtime:
            pending = False
            def batch_get(self, *values, **kwargs):
                assert kwargs == {}  # Exactly the reference positional/default-stream call.
                assert not self.pending, "previous validation spilled into this sample"
                self.pending = True
                events.append("enqueue")
                now[0] += 10000
            def scatter_profile(self):
                self.pending = False
                events.append("snapshot_and_validate")
                now[0] += 1000000
                records = profiles(4, 1024, 656)
                host = dict.fromkeys(HOST_TIMINGS, 0)
                host.update(host_binding_ns=5000, host_tensor_metadata_ns=1000,
                            host_native_ns=3000, host_operator_ns=2000)
                records[0]["host_profile"] = host
                return records
            def stop_scatter_profile(self):
                self.pending = False
                events.append("baseline_validate")
                now[0] += 1000000
        def sync():
            events.append("global_sync")
            now[0] += 90000
        torch = SimpleNamespace(npu=SimpleNamespace(synchronize=sync))
        runtime = Runtime()
        class Inputs:
            def __iter__(self):
                now[0] += 1234  # Request unpacking must stay outside the timing window.
                return iter((1, 2, 3))
        sample = perf.measure_call(runtime, "table", Inputs(), "count", "output", torch,
                                   args, 0, clock=lambda: now[0])
        self.assertEqual(sample["end_to_end_us"], 100)
        self.assertEqual(sample["host_call_ns"], 10000)
        self.assertEqual(sample["device_sync_us"], 90)
        self.assertEqual(sample["host_stream_sync_us"], 0)  # Native stream timer is not external global sync.
        self.assertEqual(sample["host_python_dispatch_us"], 5)
        self.assertEqual(sample["snapshot_ns"], 1000000)
        self.assertEqual(sample["payload_bytes"], 1024 * 656)
        args.kernel_profile = False
        args.host_profile = False
        sample = perf.measure_call(runtime, "table", (1, 2, 3), "count", "output", torch,
                                   args, 1, clock=lambda: now[0])
        self.assertNotIn("workers", sample)
        self.assertEqual(sample["status_validation_ns"], 1000000)
        self.assertEqual(sample["end_to_end_us"], 100)
        self.assertEqual(sample["device_sync_us"], 90)
        self.assertFalse(runtime.pending)
        self.assertEqual(events, ["enqueue", "global_sync", "snapshot_and_validate",
                                  "enqueue", "global_sync", "baseline_validate"])

    def test_validated_mode_has_no_extra_global_sync(self):
        args = self.parse("--single", "--completion-mode", "validated", "--no-kernel-profile")
        now = [0]
        class Runtime:
            def batch_get(self, *values, **kwargs):
                assert kwargs == {"synchronize": True}
                now[0] += 100000
        def unexpected():
            self.fail("validated mode added a redundant global synchronization")
        torch = SimpleNamespace(npu=SimpleNamespace(synchronize=unexpected))
        sample = perf.measure_call(Runtime(), "table", (1, 2, 3), "count", "output", torch,
                                   args, 0, clock=lambda: now[0])
        self.assertEqual(sample["end_to_end_us"], 100)

    def test_prepared_buffers_refresh_every_plan_without_rebinding(self):
        fixed = tuple(Vector([0, 0, -1]) for _ in range(3))
        original_ids = [id(t) for t in fixed]
        for values in ((7, 11, 2), (3, 19, 0)):
            plan = tuple(Vector([v, v + 1, -1]) for v in values)
            current = perf.refresh_request_buffers(plan, fixed)
            self.assertIs(current, fixed)
            self.assertEqual([id(t) for t in current], original_ids)
            self.assertEqual([t.values() for t in current], [t.values() for t in plan])
            self.assertIs(perf.refresh_request_buffers(plan, None), plan)

    def test_prepared_measurement_keeps_external_sync_and_profile_outside(self):
        for completion in ("reference", "validated"):
            args = self.parse("--single", "--host-submit", "prepared", "--host-profile",
                              "--miss-count", "1024", "--aiv-cores", "4",
                              "--completion-mode", completion)
            now, events = [0], []
            class Runtime:
                def batch_get(self, *args, **kwargs):
                    raise AssertionError("prepared sample used dynamic submission")
                def scatter_profile(self):
                    events.append("snapshot")
                    now[0] += 1000000
                    records = profiles(4, 1024, 656)
                    host = dict.fromkeys(HOST_TIMINGS, 0)
                    host.update(host_binding_ns=5000, host_stream_resolve_ns=300,
                                host_native_ns=4000, host_operator_ns=3000)
                    records[0]["host_profile"] = host
                    return records
            def submit():
                events.append("submit")
                now[0] += 10000 if completion == "reference" else 100000
            def sync():
                events.append("sync")
                now[0] += 90000
            sample = perf.measure_call(Runtime(), "table", (1, 2, 3), "count", "output",
                SimpleNamespace(npu=SimpleNamespace(synchronize=sync)), args, 0,
                clock=lambda: now[0], submit=submit)
            self.assertEqual(sample["end_to_end_us"], 100)
            self.assertEqual(sample["snapshot_us"], 1000)
            self.assertEqual(sample["host_tensor_metadata_us"], 0)
            self.assertEqual(sample["host_stream_resolve_us"], .3)
            self.assertEqual(events, ["submit", "sync", "snapshot"] if completion == "reference"
                             else ["submit", "snapshot"])

    def test_warmup_drains_last_status_and_does_not_hide_errors(self):
        args = self.parse("--single", "--warmup", "3")
        events = []
        class Runtime:
            pending = False
            def batch_get(self, *values, **kwargs):
                assert kwargs == {}
                self.pending = True
                events.append("enqueue")
            def scatter_profile(self):
                self.pending = False
                events.append("validate")
                return []
            def stop_scatter_profile(self):
                raise RuntimeError("business status failed")
        runtime = Runtime()
        torch = SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: events.append("sync")))
        perf.warmup_calls(runtime, "table", "count", "output", torch, args, lambda: (1, 2, 3))
        self.assertFalse(runtime.pending)
        self.assertEqual(events, ["sync", "enqueue", "sync"] * 3 + ["validate"])
        with self.assertRaisesRegex(RuntimeError, "business status failed"):
            perf.complete_outside_measurement(runtime, False)

    def test_reference_enqueue_signature_and_matrix_axes_are_shared(self):
        # Check against the actual reference, not a separately invented shape.
        old_tree = ast.parse(inspect.getsource(reference.main))
        old_calls = [n for n in ast.walk(old_tree) if isinstance(n, ast.Call) and
                     isinstance(n.func, ast.Attribute) and n.func.attr == "batch_get"]
        self.assertEqual(len(old_calls), 3)
        new_tree = ast.parse(inspect.getsource(perf.measure_call))
        branch = next(n.body for n in new_tree.body[0].body if isinstance(n, ast.If))
        new_call = next(n for statement in branch for n in ast.walk(statement)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "batch_get")
        self.assertEqual(len(new_call.args), len(old_calls[-1].args))
        self.assertEqual(new_call.keywords, old_calls[-1].keywords)
        script = Path(reference.__file__).parents[1] / "run_remote_cpu_batch_get_miss_count_perf.sh"
        source = script.read_text(encoding="utf-8")
        for name, expected in (("TOKEN_BYTES", perf.TOKEN_SIZES), ("MISS_COUNTS", perf.MISS_COUNTS)):
            values = source.split("KVAF_BATCH_GET_" + name + "=(", 1)[1].split(")", 1)[0]
            self.assertEqual(tuple(map(int, values.split())), expected)

    def test_output_count_and_address_table_allocation_order_match(self):
        for function in (reference.main, perf.run_case):
            expected = [("torch", "empty"), ("torch", "tensor"),
                        ("runtime", "batch_put") if function == reference.main else (None, "populate_table"),
                        ("runtime", "access_table_view")]
            tree = ast.parse(inspect.getsource(function))
            positions = []
            for owner, name in expected:
                positions.append(min(n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and (isinstance(n.func, ast.Name) and owner is None and n.func.id == name or
                         isinstance(n.func, ast.Attribute) and n.func.attr == name and
                         isinstance(n.func.value, ast.Name) and n.func.value.id == owner)))
            self.assertEqual(positions, sorted(positions), function.__name__)

    def test_full_table_population_keeps_chunks_and_reports_progress(self):
        args = self.parse("--single", "--table-capacity", "7", "--tokens-per-block", "128")
        block_bytes = args.tokens_per_block * args.token_bytes
        calls = []
        runtime = SimpleNamespace(batch_put=lambda *values: calls.append(values))
        source = ["block0", "block1", "block2"]
        offsets = [i * block_bytes for i in range(3)]
        lengths = [block_bytes] * 3
        capture = io.StringIO()
        with patch.object(perf.time, "perf_counter_ns", side_effect=[0, 10**9, 6*10**9, 7*10**9]), \
                contextlib.redirect_stdout(capture):
            perf.populate_table(runtime, args, source, offsets, lengths, 3)
        self.assertEqual([key for call in calls for key in call[0]], list(range(7)))
        self.assertEqual([len(call[1]) for call in calls], [3, 3, 1])
        for keys, values, actual_offsets, actual_lengths in calls:
            self.assertEqual(values, source[:len(keys)])
            self.assertEqual(actual_offsets, offsets[:len(keys)])
            self.assertEqual(actual_lengths, lengths[:len(keys)])
        self.assertNotIn("entries=3 ", capture.getvalue())  # Throttle setup logging.
        self.assertIn("entries=6 total_entries=7", capture.getvalue())
        self.assertIn(f"entries=7 total_entries=7 bytes={7 * block_bytes}", capture.getvalue())

    def test_failed_setup_phase_does_not_become_success(self):
        report = {"setup_us": {}}
        capture = io.StringIO()
        with patch.object(perf.time, "perf_counter_ns", side_effect=[0, 5000]), \
                contextlib.redirect_stdout(capture), self.assertRaisesRegex(RuntimeError, "register failed"):
            with perf.setup_phase(report, "initialize", ddr_bytes=670014898176):
                raise RuntimeError("register failed")
        self.assertEqual(report["setup_us"], {})
        self.assertIn("phase=initialize event=failed elapsed_us=5.000", capture.getvalue())
        self.assertNotIn("event=end", capture.getvalue())

    def test_statistics_use_payload_not_capacity_and_keep_quantiles(self):
        stats = perf.statistics_ns([100, 200, 300], 656 * 32)
        self.assertEqual(stats["mean_us"], .2)
        self.assertEqual(stats["p50_us"], .2)
        self.assertAlmostEqual(stats["p90_us"], .28)
        self.assertAlmostEqual(stats["p99_us"], .298)
        self.assertEqual(stats["effective_GBps"], 656 * 32 / 200)
        with self.assertRaises(ValueError):
            perf.statistics_ns([0], 656)
        self.assertNotIn("effective_GBps", perf.statistics_ns([0]))

    def test_dry_run_does_not_create_output_or_import_hardware(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "absent"
            capture = io.StringIO()
            with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(capture):
                self.assertEqual(perf.main(["--build", "absent-build", "--dry-run", "--output-dir", str(output),
                                            "--token-sizes", "656", "--miss-counts", "32,1024",
                                            "--core-counts", "4"]), 0)
            plan = json.loads(capture.getvalue())
            self.assertEqual(plan["combinations"], 2)
            self.assertEqual(plan["cases"][0]["payload_bytes"], 656 * 32)
            self.assertEqual(plan["cpu_jettys"], 4)
            self.assertEqual(plan["completion_mode"], "reference")
            self.assertEqual(plan["cases"][0]["queue_matched_direct_jettys_per_core"], 1)
            self.assertFalse(output.exists())

    def test_worker_dry_run_never_enters_hardware_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "absent"
            capture = io.StringIO()
            with patch.dict(os.environ, {"KVAF_RUN_NPU_TESTS": "1"}, clear=True), \
                    patch.object(perf, "run_case", side_effect=AssertionError("dry-run accessed hardware")), \
                    contextlib.redirect_stdout(capture):
                self.assertEqual(perf.main(["--build", "unused", "--single", "--worker", "--dry-run",
                                            "--json", str(root / "case.json"),
                                            "--csv", str(root / "case.csv")]), 0)
            plan = json.loads(capture.getvalue())
            self.assertTrue(plan["dry_run"])
            self.assertEqual(plan["combinations"], 1)
            self.assertFalse(root.exists())

    def test_sweep_summaries_and_failure_stop(self):
        def success(command, **kwargs):
            child = self.parse(*command[2:])
            sample = dict(iteration=0, payload_bytes=child.miss_count * child.token_bytes,
                          end_to_end_ns=1000, aiv_body_ns=700)
            report = dict(config=vars(child).copy(), samples={"profiled": [sample]}, summary={"profiled": {
                key: perf.statistics_ns([sample[key]], sample["payload_bytes"])
                for key in ("end_to_end_ns", "aiv_body_ns")}})
            report["config"] = {k: str(v) if isinstance(v, Path) else v for k,v in report["config"].items()}
            write_results(child, report)
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "success"
            argv = ["--build", "unused", "--output-dir", str(output), "--token-sizes", "656,4096",
                    "--miss-counts", "32", "--core-counts", "4", "--iterations", "1"]
            with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(perf.main(argv, runner=success), 0)
            with (output / "summary.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual({r["stage"] for r in rows}, {"kernel_total", "end_to_end"})
            self.assertEqual({r["token_bytes"] for r in rows}, {"656", "4096"})
            failed = Path(temporary) / "failure"
            calls = []
            def fail(command, **kwargs):
                calls.append(command)
                if len(calls) == 1:
                    return success(command, **kwargs)
                return SimpleNamespace(returncode=23)
            argv[3] = str(failed)
            argv[7] = "32,64"  # Four cases: stop after case 2 and retain case 1.
            with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(perf.main(argv, runner=fail), 23)
            self.assertEqual(len(calls), 2)
            with (failed / "summary.csv").open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_single_case_timeout_and_original_cleanup_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            def timeout(command, **kwargs):
                self.assertEqual(kwargs["timeout"], 3)
                self.assertEqual(kwargs["env"]["PYTHONUNBUFFERED"], "1")
                kwargs["stdout"].write("old line\n" * 10000 +
                                       "SCATTER_ALIGNED_PHASE phase=initialize event=begin ddr_bytes=670014898176\n")
                raise subprocess.TimeoutExpired(command, 3)
            capture = io.StringIO()
            with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(capture):
                code = perf.main(["--build", "unused", "--single", "--case-timeout", "3",
                                  "--output-dir", temporary], runner=timeout)
            self.assertEqual(code, 124)
            self.assertIn("timeout_scope=initialize+populate", capture.getvalue())
            self.assertIn("phase=initialize event=begin", capture.getvalue())
            self.assertLess(len(capture.getvalue()), 2000)
        events = []
        class Runtime:
            initialized = True
            def release_access_table(self, table):
                events.append("release")
                raise RuntimeError("release failed")
            def shutdown(self):
                events.append("shutdown")
                raise RuntimeError("shutdown failed")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(RuntimeError, "release failed"):
            perf.close_case(Runtime(), "table")
        self.assertEqual(events, ["release", "shutdown"])


if __name__ == "__main__":
    unittest.main()
