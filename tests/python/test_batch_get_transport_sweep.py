#!/usr/bin/env python3
"""Exercise the real sweep with a fake runner and real benchmark reporting."""

import contextlib
import csv
import io
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

from test_batch_get_stream_config import perf


def fake_runner():
    args = perf.parse_args(sys.argv[3:])
    error = perf.validate_args(args)
    if error:
        sys.exit(error)
    assert args.kernel_profile and args.iterations == 1
    streams, transports = args.stream_count, args.transport_count
    cores, jettys, miss = args.aiv_cores, args.jettys_per_core, args.miss_count
    paths = len(args.npu_eids) or 1
    fault = os.environ.get("KVAF_SWEEP_TEST_FAULT", "")
    if fault == "fail_second" and miss == 1:
        sys.exit(7)
    print(f"BATCH_GET_PERF_CONFIG stream_count={streams} transport_count={transports} "
          f"aiv_cores={cores} total_aiv_cores={streams * cores} jettys_per_core={jettys} "
          f"miss_count={miss} token_bytes={args.token_bytes} eid_paths={paths}")
    # An abandoned candidate must not contaminate the final resource group.
    print("[KVAF_AIV] BATCH_GET_TRANSPORT_READY transport_id=0 workers=999")
    print("[KVAF_AIV] BATCH_GET_TRANSPORT_CONTEXT transport_id=0 eid_path=0 context=0xdead")

    def context(t, p):
        return 0x1000 + (0 if fault == "shared_context" else t) * 0x10000 + p * 0x100

    for t in range(transports):
        first_stream = 0 if transports == 1 else t
        print(f"[KVAF_AIV] BATCH_GET_TRANSPORT_READY transport_id={t} first_stream={first_stream} "
              f"stream_count={streams // transports} cores_per_stream={cores} "
              f"workers={streams * cores * jettys // transports} eid_paths={paths}")
        for p in range(paths):
            print(f"[KVAF_AIV] BATCH_GET_TRANSPORT_CONTEXT transport_id={t} eid_path={p} "
                  f"context={context(t, p):#x} session=0xabcd")
        for s in range(first_stream, first_stream + streams // transports):
            for c in range(cores):
                for j in range(jettys):
                    p = (j * (streams * cores // transports) + (s - first_stream) * cores + c) % paths
                    worker = (s * cores + c) * jettys + j
                    sq = 0x100000 + (0 if fault == "shared_sq" else worker) * 4096
                    print(f"[KVAF_AIV] BATCH_GET_TRANSPORT_WORKER transport_id={t} stream={s} "
                          f"stream_core={c} jetty_slot={j} eid_path={p} context={context(t, p):#x} "
                          f"sq_va={sq:#x} cq_va={sq + 2048:#x} local_tp=0x10 peer_tp=0x20")
    reported = 3 - transports if fault == "wrong_transport" else transports
    print(f"[KVAF_AIV] BATCH_GET_STREAMS_READY streams={streams} transport_count={reported} "
          f"cores_per_stream={cores} total_cores={streams * cores} "
          f"jettys_per_core={jettys} total_jettys={streams * cores * jettys}")
    print(f"BATCH_GET_PERF_CORRECTNESS_PASS miss_count={miss}")
    profiles = []
    for core in range(streams * cores):
        start = (core // cores) * 1_000_000
        profiles.append(dict(
            core=core, kernel_start_cycle=start, setup_done_cycle=start,
            compact_done_cycle=start, first_doorbell_cycle=start + 1,
            post_done_cycle=start + 500_000, completion_done_cycle=start + 1_000_000,
            kernel_finish_cycle=start + 1_000_000, active_requests=1, total_posts=1,
            jetty_completion_wait_cycles=[0] * 64, jetty_completion_polls=[0] * 64,
        ))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        perf.print_samples("batch_get", [3.0], 0, args.request_length, miss, args.token_bytes)
        perf.print_kernel_profile([profiles], 0, args.kernel_profile_clock_mhz,
                                  streams * cores, jettys, args.request_length,
                                  miss, args.token_bytes, streams, transports)
    for line in output.getvalue().splitlines():
        if line.startswith("KERNEL_PROFILE_CORE "):
            continue
        if fault == "missing_wall" and "operation=batch_get_kernel_wall " in line:
            continue
        print(line)


class BatchGetTransportSweepTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="kvaf_transport_sweep_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        tests = self.root / "tests"
        tests.mkdir()
        self.script = tests / "run_remote_cpu_batch_get_dual_stream_perf.sh"
        shutil.copy2(Path(__file__).resolve().parents[1] / self.script.name, self.script)
        runner = tests / "run_remote_cpu_batch_get_perf.sh"
        runner.write_text("#!/usr/bin/env bash\nexec " + shlex.quote(sys.executable) + " " +
                          shlex.quote(str(Path(__file__).resolve())) + ' --fake-runner "$@"\n')
        runner.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("KVAF_")}
        self.env.update(
            KVAF_BATCH_GET_DUAL_STREAM_TOKEN_BYTES="1000",
            KVAF_BATCH_GET_DUAL_STREAM_MISS_COUNTS="1000",
            KVAF_BATCH_GET_DUAL_STREAM_AIV_CORES="2",
            KVAF_BATCH_GET_DUAL_STREAM_JETTYS_PER_CORE="1",
            KVAF_BATCH_GET_PERF_ITERATIONS="1",
            KVAF_BATCH_GET_KERNEL_CLOCK_MHZ="800",
            KVAF_BATCH_GET_DUAL_STREAM_LOG_DIR=str(self.root / "logs"),
        )

    def run_sweep(self, *options, **env):
        return subprocess.run(
            ["bash", str(self.script), "/mock/build path", *options],
            env=dict(self.env, **env), text=True, capture_output=True, timeout=30,
        )

    def rows(self, streams=2, transports=1):
        label = "single" if streams == 1 else "dual"
        suffix = "_2_transports" if transports == 2 else ""
        path = self.root / "logs" / f"batch_get_{label}_stream_summary{suffix}.csv"
        with path.open(newline="") as source:
            return list(csv.DictReader(source))

    def test_modes_resources_and_wall_bandwidth(self):
        for streams, transports in ((1, 1), (2, 1), (2, 2)):
            with self.subTest(streams=streams, transports=transports):
                result = self.run_sweep("--stream-count", str(streams), "--transport-count", str(transports))
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                row, = self.rows(streams, transports)
                self.assertEqual(int(row["transport_count"]), transports)
                self.assertEqual(int(row["context_count"]), transports)
                self.assertEqual(int(row["total_jettys"]), streams * 2)
                self.assertEqual(row["kernel_wall_p50_us"], f"{streams * 1250:.3f}")
                self.assertEqual(row["kernel_wall_p50_GBps"], f"{0.8 / streams:.3f}")
        self.assertEqual(len(list((self.root / "logs").glob("*.log"))), 3)

    def test_environment_override_and_multi_eid(self):
        result = self.run_sweep(KVAF_NPU_BATCH_GET_TRANSPORT_COUNT="2")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_sweep("--transport-count=1", KVAF_NPU_BATCH_GET_TRANSPORT_COUNT="2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.rows()[0]["transport_count"], "1")
        result = self.run_sweep("--transport-count", "2", "--npu-eids", "1" * 32 + "," + "2" * 32)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.rows(2, 2)[0]["context_count"], "4")

    def test_invalid_configuration(self):
        for options in (("--transport-count",), ("--transport-count=0",),
                        ("--transport-count=3",), ("--stream-count=1", "--transport-count=2")):
            with self.subTest(options=options):
                self.assertEqual(self.run_sweep(*options).returncode, 2)

    def test_resources_and_profiles_fail_closed(self):
        for fault in ("shared_context", "shared_sq", "wrong_transport", "missing_wall"):
            with self.subTest(fault=fault):
                result = self.run_sweep("--transport-count=2", KVAF_SWEEP_TEST_FAULT=fault)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("inconsistent", result.stderr)
                self.assertEqual(self.rows(2, 2), [])

    def test_failed_run_preserves_previous_results(self):
        result = self.run_sweep("--transport-count=2", KVAF_SWEEP_TEST_FAULT="fail_second",
                                KVAF_BATCH_GET_DUAL_STREAM_MISS_COUNTS="0 1")
        self.assertEqual(result.returncode, 7, result.stderr)
        row, = self.rows(2, 2)
        self.assertEqual(row["payload_bytes"], "0")
        self.assertEqual(row["kernel_wall_p50_GBps"], "0.000")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fake-runner":
        fake_runner()
    else:
        unittest.main()
