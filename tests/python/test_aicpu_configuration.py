"""Host-only regression for AICPU CLI, performance statistics and package installation."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "access_operator/demo/aicpu_batch_get/python"))
import test_batch_get as functional
import benchmark_gather_read as benchmark

EIDS = "00112233445566778899aabbccddeeff,102132435465768798a9bacbdcedfe0f"
SCATTER = "KvafAicpuCpuPushScatterA12V24"


class PerformanceStatisticsTests(unittest.TestCase):
    @staticmethod
    def samples():
        fields = ("request_us", "source_write_us", "notify_us", "read_us", "wr_build_us",
                  "ready_wait_us", "post_us", "cq_wait_us", "scatter_us", "ack_us")
        # Python durations deliberately differ: headline statistics must use
        # kernel time and exclude Host call overhead.
        return [{"python_call_us": value * 2, "device_span_us": value,
                 "workers": [dict.fromkeys(fields, 1), dict.fromkeys(fields, 2)]}
                for value in (10, 20, 100)]

    def test_rates_correspond_to_latency_statistics(self):
        stats = functional.timing_stats([10, 20, 100], 1_000_000)
        self.assertAlmostEqual(stats["mean_us"], 130 / 3)
        self.assertEqual(stats["p50_us"], 20)
        self.assertAlmostEqual(stats["p99_us"], 98.4)
        self.assertAlmostEqual(stats["mean_GBps"], 1_000_000 / (130 / 3) / 1000)
        self.assertEqual(stats["p50_GBps"], 50)
        self.assertAlmostEqual(stats["p99_GBps"], 1_000_000 / 98.4 / 1000)
        self.assertEqual(stats["GBps"], stats["mean_GBps"])
        for payload in (0, 1_000_000):
            single = functional.timing_stats([12.5], payload)
            for metric in ("mean", "p50", "p99"):
                self.assertEqual(single[metric + "_us"], 12.5)
                self.assertEqual(single[metric + "_GBps"], payload / 12.5 / 1000)

    def test_both_modes_report_six_kernel_metrics(self):
        expected = {"mean_us": "43.3", "mean_GBps": "23.077",
                    "p50_us": "20.0", "p50_GBps": "50.000",
                    "p99_us": "98.4", "p99_GBps": "10.163"}
        for mode, prefix in benchmark.PREFIXES.items():
            with self.subTest(mode=mode):
                summary = benchmark.summarize_measurements(self.samples(), 1_000_000, mode=mode)
                args = benchmark.parse_args(["--token-bytes", "1000", "--miss-count", "1000"], mode=mode)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    benchmark.print_measurements(args, summary, mode=mode)
                for suffix in ("_PROFILE ", "_RESULT "):
                    line = next(line for line in output.getvalue().splitlines()
                                if line.startswith(prefix + suffix))
                    fields = dict(word.split("=", 1) for word in line.split()[1:])
                    self.assertEqual(fields["metric"], "kernel_wall")
                    self.assertFalse(any(key.startswith("device_") for key in fields))
                    for key, value in expected.items():
                        self.assertEqual(fields[key], value)

    def test_both_sweeps_keep_metrics_in_csv_and_terminal(self):
        # Exercise the real summarizer/formatter and shell CSV path with
        # deterministic kernel timings, without allocating NPU/DDR buffers.
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            runner = directory / "python-profile-fixture"
            runner.write_text(f"#!{sys.executable}\n" +
                "import sys\nfrom pathlib import Path\n" +
                f"sys.path.insert(0, {str(ROOT / 'access_operator/demo/aicpu_batch_get/python')!r})\n" +
                "import benchmark_gather_read as b\n" +
                "mode = 'cpu_push_scatter' if Path(sys.argv[1]).name == 'benchmark_cpu_push_scatter.py' else 'gather_read'\n" +
                "a = b.parse_args(sys.argv[2:], mode=mode)\n" +
                f"profiles = {self.samples()!r}\n" +
                "b.print_measurements(a, b.summarize_measurements(profiles, a.token_bytes * a.miss_count, mode=mode), mode=mode)\n")
            runner.chmod(0o755)
            for mode, script in (("gather_read", "run_aicpu_gather_read_sweep.sh"),
                                 ("cpu_push_scatter", "run_aicpu_cpu_push_scatter_sweep.sh")):
                with self.subTest(mode=mode):
                    log_dir = directory / mode
                    env = dict(os.environ, PYTHON=str(runner), KVAF_AICPU_TOKEN_BYTES="1000",
                               KVAF_AICPU_MISS_COUNTS="1000", KVAF_AICPU_WORKERS="2",
                               KVAF_AICPU_SWEEP_LOG_DIR=str(log_dir))
                    result = subprocess.run(["bash", str(ROOT / "tests" / script), str(directory / "build")],
                                            env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    with (log_dir / "summary.csv").open() as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(len(rows), 1)
                    row = rows[0]
                    keys = ("mean_us", "mean_GBps", "p50_us", "p50_GBps", "p99_us", "p99_GBps")
                    self.assertEqual(row["metric"], "kernel_wall")
                    self.assertFalse(any(key.startswith("device_") for key in row))
                    self.assertEqual([row[key] for key in keys],
                                     ["43.3", "23.077", "20.0", "50.000", "98.4", "10.163"])
                    self.assertIn("token_bytes miss_count workers " + " ".join(keys), result.stdout)
                    self.assertIn("1000 1000 2 43.3 23.077 20.0 50.000 98.4 10.163", result.stdout)


class PublicCliTests(unittest.TestCase):
    def test_both_algorithms_accept_candidate_lists_without_json(self):
        with patch.dict(os.environ, {}, clear=True):
            args = functional.parse_args(["--npu-eids", EIDS, "--modes", "gather_read", "cpu_push_scatter"])
            self.assertEqual(args.npu_eids, EIDS.split(","))
            for mode in ("gather_read", "cpu_push_scatter"):
                args = benchmark.parse_args(["--npu-eids", EIDS], mode=mode)
                self.assertEqual(args.npu_eids, EIDS.split(","))
                self.assertFalse(hasattr(args, "kernel_json"))

    def test_auto_discovery_and_existing_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(functional.parse_args([]).npu_eids, [])
            self.assertEqual(benchmark.parse_args([]).npu_eids, [])
        with patch.dict(os.environ, {"KVAF_NPU_EIDS": EIDS}, clear=True):
            self.assertEqual(functional.parse_args([]).npu_eids, EIDS.split(","))
            self.assertEqual(benchmark.parse_args([]).npu_eids, EIDS.split(","))

    def test_explicit_table_capacity_can_exceed_default_ddr_budget(self):
        for mode in benchmark.PREFIXES:
            with self.subTest(mode=mode):
                defaults = benchmark.parse_args([], mode=mode)
                default_capacity, _ = benchmark.pool_layout(
                    defaults.token_bytes, defaults.tokens_per_block, mode=mode)
                self.assertEqual(defaults.table_capacity, default_capacity)

                block_bytes = defaults.token_bytes * defaults.tokens_per_block
                capacity = benchmark.DDR_BUDGET_BYTES // block_bytes + 1
                args = benchmark.parse_args(["--table-capacity", str(capacity)], mode=mode)
                self.assertEqual(args.table_capacity, capacity)

    def test_table_capacity_must_fit_positive_int32(self):
        for mode in benchmark.PREFIXES:
            for capacity in (1, 0x7FFFFFFF):
                with self.subTest(mode=mode, capacity=capacity):
                    args = benchmark.parse_args(["--table-capacity", str(capacity)], mode=mode)
                    self.assertEqual(args.table_capacity, capacity)
            for capacity in (-1, 0, 0x80000000):
                with self.subTest(mode=mode, capacity=capacity):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        benchmark.parse_args(["--table-capacity", str(capacity)], mode=mode)

    def test_retired_options_are_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            for parse in (functional.parse_args, benchmark.parse_args):
                for argv in (["--kernel-json", "/tmp/kernel.json"], ["--source-mode", "direct"],
                             ["--urma-device", "udma0"], ["--eid-index", "0"], ["--npu-eid", EIDS[:32]]):
                    with self.subTest(parse=parse, argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        parse(argv)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                functional.parse_args(["--modes", "gather_read_direct"])


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "cann"
        (self.root / "conf").mkdir(parents=True)
        (self.root / "conf/ascend_package_load.ini").write_text("# loader\n")
        self.artifacts = Path(self.temp.name) / "artifacts"
        self.artifacts.mkdir()
        self.entries = {name: {} for name in ("KvafAicpuBatchPut", "KvafAicpuGatherRead", "KvafAicpuUmdkControl")}
        self.revision = 0
        self.write_json()
        self.write_archive(b"old kernel")

    def write_archive(self, data):
        with tarfile.open(self.artifacts / "kvaf-aicpu-urma.tar.gz", "w:gz") as archive:
            info = tarfile.TarInfo("aicpu_kernels_device/libkvaf_aicpu_urma.so")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    def write_json(self):
        (self.artifacts / "kernel.json").write_text(json.dumps(self.entries))

    def installer(self, *args, env=None):
        return subprocess.run(["bash", str(ROOT / "access_operator/src/aicpu_urma/package/install_kernel.sh"), *map(str, args)],
                              capture_output=True, text=True, env=env)

    def installed_paths(self):
        base = self.root / "opp/vendors/cust/op_impl/aicpu"
        return [base / "kernel/kvaf-aicpu-urma.tar.gz", base / "config/kvaf_aicpu_urma.json",
                self.root / "conf/ascend_package_load.ini"]

    def snapshot(self):
        return [path.read_bytes() if path.exists() else None for path in self.installed_paths()]

    def update_artifacts(self):
        self.revision += 1
        self.entries[SCATTER] = {"revision": self.revision}
        self.write_json()
        self.write_archive(f"new kernel {self.revision}".encode())

    def assert_no_transaction_files(self):
        self.assertFalse((self.root / "conf/.kvaf-aicpu-urma.lock").exists())
        self.assertFalse(list(self.root.rglob("*.kvaf.*")))

    def fault_environment(self, command, target=None):
        # Fail the real installer's command, allowing all rollback commands to run.
        directory = Path(self.temp.name) / "bin"
        directory.mkdir(exist_ok=True)
        wrapper = directory / command
        wrapper.write_text(f"#!{sys.executable}\n" +
            "import os, sys\nfrom pathlib import Path\n" +
            ("if sys.argv[-2].endswith('/new') and Path(sys.argv[-1]).resolve() == Path(" + repr(str(target)) + ").resolve()" +
             ":\n    sys.exit(42)\n" if command == "mv" else "sys.exit(42)\n") +
            f"os.execv({shutil.which(command)!r}, [{command!r}] + sys.argv[1:])\n")
        wrapper.chmod(0o755)
        return dict(os.environ, PATH=str(directory) + os.pathsep + os.environ.get("PATH", ""))

    def test_malformed_config_does_not_replace_existing_package(self):
        self.assertEqual(self.installer("--install", self.artifacts, self.root).returncode, 0)
        config = self.installed_paths()[2]
        config.write_text(config.read_text().replace("optional:true", "optional:false"))
        before = self.snapshot()
        self.update_artifacts()
        result = self.installer("--install", self.artifacts, self.root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("malformed", result.stderr)
        self.assertEqual(self.snapshot(), before)
        self.assert_no_transaction_files()

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root bypasses directory permissions")
    def test_unwritable_conf_preserves_existing_package(self):
        self.assertEqual(self.installer("--install", self.artifacts, self.root).returncode, 0)
        before = self.snapshot()
        self.update_artifacts()
        conf = self.root / "conf"
        conf.chmod(0o555)
        try:
            result = self.installer("--install", self.artifacts, self.root)
        finally:
            conf.chmod(0o755)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.snapshot(), before)
        self.assert_no_transaction_files()

    def test_publication_failures_roll_back_fresh_install_and_upgrade(self):
        for upgrade in (False, True):
            if upgrade:
                self.assertEqual(self.installer("--install", self.artifacts, self.root).returncode, 0)
            before = self.snapshot()
            self.update_artifacts()
            for target in self.installed_paths():
                with self.subTest(upgrade=upgrade, target=target):
                    result = self.installer("--install", self.artifacts, self.root,
                                            env=self.fault_environment("mv", target))
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(self.snapshot(), before)
                    self.assert_no_transaction_files()

    def test_post_install_check_failure_rolls_back_all_files(self):
        self.assertEqual(self.installer("--install", self.artifacts, self.root).returncode, 0)
        before = self.snapshot()
        self.update_artifacts()
        result = self.installer("--install", self.artifacts, self.root,
                                env=self.fault_environment("cmp"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("differs from build artifact", result.stderr)
        self.assertEqual(self.snapshot(), before)
        self.assert_no_transaction_files()

    def resolve_root(self, **environment):
        env = dict(os.environ)
        env.pop("ASCEND_HOME_PATH", None)
        env.pop("ASCEND_CANN_PACKAGE_PATH", None)
        env.update(environment)
        return subprocess.run(["bash", "-c", 'source "$1"; kvaf_find_cann_root', "bash", str(ROOT / "scripts/cann_root.sh")],
                              env=env, capture_output=True, text=True)

    def test_root_selection_matches_runtime_contract(self):
        alias = Path(self.temp.name) / "latest"
        alias.symlink_to(self.root, target_is_directory=True)
        result = self.resolve_root(ASCEND_HOME_PATH=str(alias), ASCEND_CANN_PACKAGE_PATH=str(self.root))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.strip()), self.root.resolve())
        other = Path(self.temp.name) / "other"
        other.mkdir()
        result = self.resolve_root(ASCEND_HOME_PATH=str(self.root), ASCEND_CANN_PACKAGE_PATH=str(other))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("roots differ", result.stderr)
        self.assertNotEqual(self.resolve_root(ASCEND_HOME_PATH=str(other / "missing"),
                                            ASCEND_CANN_PACKAGE_PATH=str(self.root)).returncode, 0)

    def test_install_and_scatter_requirement(self):
        result = self.installer("--require-scatter", "--install", self.artifacts, self.root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Scatter missing", result.stderr)
        self.assertFalse((self.root / "opp").exists())
        result = self.installer("--install", self.artifacts, self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.installer("--require-scatter", "--check", self.root)
        self.assertNotEqual(result.returncode, 0)
        self.entries[SCATTER] = {}
        self.write_json()
        result = self.installer("--require-scatter", "--install", self.artifacts, self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.installer("--require-scatter", "--check", self.root, self.artifacts)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.entries["different-build"] = {}
        self.write_json()
        result = self.installer("--check", self.root, self.artifacts)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("differs from build artifact", result.stderr)


if __name__ == "__main__":
    unittest.main()
