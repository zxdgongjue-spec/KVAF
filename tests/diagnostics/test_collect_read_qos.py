from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from collect_read_qos import main, parse_perf_stats, parse_port_group
from collect_read_qos_info import (
    Collector, collect_sources, configured_response_pairs, main as info_main, normalize_eid, parse_devices,
    resolve_debug_dir, select_device,
)


CPU_EID = "00000000003f020000100000df080b00"
OTHER_EID = "00000000003f030000100000df00eb00"
DEVICES = """name : udmac0d1e2
eid0 : 0000:0000:003f:0200:0010:0000:df08:0b00
name : udmac0d0e3
eid0 : 0000:0000:0000:0000:0000:0000:0000:0000
"""
MULTI_DEVICES = DEVICES.replace("0000:0000:0000:0000:0000:0000:0000:0000", OTHER_EID)
CAPS = """logic_port_bitmap: 0x30
chip_id: 1
die_id: 1
"""
OTHER_CAPS = CAPS.replace("chip_id: 1", "chip_id: 0")
PERF = """perf_stats_period: 100(ms)
port bandwidth info:
    port_id: 0
    port_tx_bw: 64000(kbps)
    port_rx_bw: 1000(kbps)
    vl   tx_bw(kbps)          rx_bw(kbps)
    4    0                   1000
    5    64000               0

    port_id: 1
    vl   tx_bw(kbps)          rx_bw(kbps)
    8    0                   0
    9    32000               0
"""


class ReadQosTest(unittest.TestCase):
    def test_changed_server_eid_rejects_old_value_and_reports_current_value(self):
        current_eid = "00000000003f020000100000df200b00"
        devices = parse_devices(DEVICES.replace("df08:0b00", "df20:0b00"))
        with self.assertRaisesRegex(ValueError, current_eid):
            select_device(devices, CPU_EID, "udmac0d1e2")
        self.assertEqual(select_device(devices, current_eid, "udmac0d1e2"), "udmac0d1e2")

    def test_device_name_still_requires_matching_eid(self):
        devices = parse_devices(DEVICES)
        self.assertEqual(select_device(devices, CPU_EID), "udmac0d1e2")
        with self.assertRaises(ValueError):
            select_device(devices, OTHER_EID, "udmac0d0e3")
        devices["duplicate"] = [CPU_EID]
        with self.assertRaises(ValueError):
            select_device(devices, CPU_EID)

    def test_invalid_eid_is_not_silently_accepted(self):
        self.assertEqual(normalize_eid("0000:0000:003F:0200:0010:0000:DF08:0B00"), CPU_EID)
        with self.assertRaises(ValueError):
            normalize_eid("::1")

    def test_debugfs_uses_immediate_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = root / "devices/00001/00005/00009/ubcore/udmac0d1e3"
            device.mkdir(parents=True)
            debug = root / "debug/00009"
            debug.mkdir(parents=True)
            (root / "debug/00005").mkdir()
            self.assertEqual(resolve_debug_dir(device, root / "debug"), debug)

    def test_perf_preserves_ports_directions_and_real_zeros(self):
        rows = parse_perf_stats(PERF)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0], dict(port=0, vl=4, tx_kbps=0, rx_kbps=1000))
        self.assertEqual(rows[3], dict(port=1, vl=9, tx_kbps=32000, rx_kbps=0))
        self.assertEqual(parse_perf_stats("Permission denied"), [])
        self.assertEqual(parse_perf_stats("sl sw_vl hw_vl\n1 4 4\n"), [])

    def test_configured_pairs_are_not_observed_response_mapping(self):
        text = "ctp_req_vl: 8 4\nctp_resp_vl_offset: 1\n"
        self.assertEqual(configured_response_pairs(text), [[8, 9], [4, 5]])
        self.assertIsNone(configured_response_pairs("ctp_resp_vl_offset: 1\n"))

    def test_port_group_requires_complete_unambiguous_numeric_identity(self):
        self.assertEqual(parse_port_group(CAPS), (1, 1, 48))
        self.assertEqual(parse_port_group(CAPS.replace("\n", "\r\n")), (1, 1, 48))
        self.assertEqual(parse_port_group("  chip_id: 0X01\n\tdie_id: 01\nlogic_port_bitmap: 048\n"),
                         parse_port_group(CAPS))
        for field in ("chip_id", "die_id", "logic_port_bitmap"):
            lines = [line for line in CAPS.splitlines() if not line.startswith(field + ":")]
            missing = "\n".join(lines) + "\n"
            for invalid in (missing, CAPS + f"{field}: 1\n", missing + f"{field}: unknown\n",
                            missing + f"{field}: -1\n", missing + f"{field}:\n"):
                with self.subTest(field=field, caps=invalid), self.assertRaises(ValueError):
                    parse_port_group(invalid)
        with self.assertRaises(ValueError):
            parse_port_group(CAPS.replace("0x30", "0x0"))

    def test_information_snapshot_only_queries_cpu_information(self):
        def run(argv, **kwargs):
            self.assertIn(argv[0], ("cat", "modinfo", "urma_admin"))
            self.assertNotIn("/proc/modules", argv)
            for name in ("npu", "hccp", "ascend"):
                self.assertNotIn(name, " ".join(argv).lower())
            if argv[0] == "modinfo":
                self.assertIn(argv[1], ("ubase", "udma", "ubcore", "uburma"))
            self.assertFalse(any(str(arg).endswith("perf_stats") for arg in argv))
            self.assertNotIn("jetty", argv)
            self.assertNotIn("tp", argv)
            if argv[:2] == ["urma_admin", "show"] and "--all" in argv:
                output = DEVICES
            elif argv[0] == "cat" and str(argv[1]).endswith("adev_qos"):
                output = "ctp_req_vl: 8 4\nctp_resp_vl_offset: 1\n"
            else:
                output = "unavailable in fixture"
            return subprocess.CompletedProcess(argv, 0, output.encode())

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            with patch("collect_read_qos_info.sys.platform", "linux"), \
                 patch("collect_read_qos_info.subprocess.run", side_effect=run), \
                 patch("collect_read_qos_info.resolve_debug_dir", return_value=Path(directory)):
                self.assertEqual(info_main(["--eid", CPU_EID,
                                            "--output", str(output)]), 0)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["configured_ctp_req_resp_vl_pairs"], [[8, 9], [4, 5]])
            self.assertEqual(report["role"], "cpu")
            for key in ("request_packet_sl", "response_packet_sl", "response_inherits_request_sl"):
                self.assertNotIn(key, report)
            report_text = json.dumps(report).lower()
            for name in ("npu", "hccp", "ascend"):
                self.assertNotIn(name, report_text)
            for key in ("valid_perf_samples", "requested_perf_samples", "sampling_status", "sampling_error", "label"):
                self.assertNotIn(key, report)
            self.assertFalse(any(record["name"].startswith("perf_") for record in report["commands"]))
            self.assertFalse(list(output.glob("perf_*.txt")))
            self.assertFalse((output / "vl_bandwidth.csv").exists())
            self.assertFalse((output / "sampling.log").exists())
            self.assertTrue((output / "modinfo_udma.txt").is_file())
            self.assertTrue((output / "qos_sl_vl_map.txt").is_file())

    def test_sampling_only_writes_raw_data_and_timing_log(self):
        def run(argv, **kwargs):
            if argv == ["urma_admin", "show", "--all", "--whole"]:
                output = DEVICES
            elif argv[0] == "cat" and str(argv[1]).endswith("caps_info"):
                output = CAPS
            elif argv[0] == "cat" and str(argv[1]).endswith("perf_stats"):
                output = PERF
            else:
                self.fail(f"Unexpected command during sampling: {argv}")
            return subprocess.CompletedProcess(argv, 0, output.encode())

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "samples"
            with patch("collect_read_qos.sys.platform", "linux"), \
                 patch("collect_read_qos_info.subprocess.run", side_effect=run) as command, \
                 patch("collect_read_qos.resolve_debug_dir", return_value=Path(directory)), \
                 patch("collect_read_qos.time.sleep") as sleep:
                self.assertEqual(main(["--eid", CPU_EID,
                                       "--samples", "2", "--interval", "0.1", "--label", "p15-only",
                                       "--output", str(output)]), 0)
            self.assertEqual(command.call_count, 4)
            sleep.assert_called_once_with(0.1)
            self.assertEqual({path.name for path in output.iterdir()},
                             {"udmac0d1e2", "sampling.log"})
            device_output = output / "udmac0d1e2"
            self.assertEqual({path.name for path in device_output.iterdir()},
                             {"perf_0000.txt", "perf_0001.txt", "sampling.log"})
            for name in ("perf_0000.txt", "perf_0001.txt"):
                self.assertEqual((device_output / name).read_text(encoding="utf-8"), PERF)
            log = (device_output / "sampling.log").read_text(encoding="utf-8")
            self.assertIn("label='p15-only'", log)
            self.assertIn("SAMPLING_DONE status=complete device=udmac0d1e2 valid_perf_samples=2 requested_perf_samples=2", log)
            self.assertIn("role=cpu", log)
            self.assertEqual(log.count("start_unix="), 2)
            self.assertEqual(log.count("end_unix="), 2)
            self.assertNotIn("64000", log)
            root_log = (output / "sampling.log").read_text(encoding="utf-8")
            self.assertEqual(root_log.count("name=urma_devices"), 1)
            self.assertEqual(root_log.count("name=caps_info_udmac0d1e2"), 1)
            self.assertIn("COLLECTION_DONE status=complete completed_devices=1 requested_devices=1", root_log)

    def test_sampling_failures_are_visible_and_return_nonzero(self):
        for stage in ("device_identity", "debugfs_resolution", "perf_stats_read", "perf_stats_parse"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                def run(argv, **kwargs):
                    code, output = 0, "unavailable in fixture"
                    if argv[:2] == ["urma_admin", "show"] and "--all" in argv:
                        output = "" if stage == "device_identity" else DEVICES
                    elif argv[0] == "cat" and str(argv[1]).endswith("caps_info"):
                        output = CAPS
                    elif argv[0] == "cat" and str(argv[1]).endswith("perf_stats"):
                        code = 1 if stage == "perf_stats_read" else 0
                        output = "Permission denied" if code else "unknown bandwidth format"
                    return subprocess.CompletedProcess(argv, code, output.encode())

                error = ValueError("Controller debugfs unavailable") if stage == "debugfs_resolution" else None
                output, terminal = Path(directory) / "report", io.StringIO()
                with patch("collect_read_qos.sys.platform", "linux"), \
                     patch("collect_read_qos_info.subprocess.run", side_effect=run), \
                     patch("collect_read_qos.resolve_debug_dir", return_value=Path(directory), side_effect=error), \
                     redirect_stdout(terminal):
                    result = main(["--eid", CPU_EID,
                                   "--output", str(output)])
                root_log = (output / "sampling.log").read_text(encoding="utf-8")
                log_path = output / "udmac0d1e2/sampling.log"
                log = log_path.read_text(encoding="utf-8") if log_path.exists() else root_log
                self.assertEqual(result, 1)
                self.assertIn("COLLECTION_DONE status=incomplete completed_devices=0", root_log)
                self.assertIn(f"stage={stage}", log)
                self.assertIn(f"stage={stage}", terminal.getvalue())
                self.assertFalse((output / "report.json").exists())
                if stage in ("device_identity", "debugfs_resolution"):
                    self.assertNotIn("SAMPLING_READY", terminal.getvalue())
                    self.assertFalse(list(output.glob("perf_*.txt")))

    def test_sampling_timeout_preserves_partial_output_and_completed_count(self):
        reads = 0

        def run(argv, **kwargs):
            nonlocal reads
            if argv[0] == "urma_admin":
                return subprocess.CompletedProcess(argv, 0, DEVICES.encode())
            if Path(argv[1]).name == "caps_info":
                return subprocess.CompletedProcess(argv, 0, CAPS.encode())
            reads += 1
            if reads == 2:
                raise subprocess.TimeoutExpired(argv, 15, output=b"partial read")
            return subprocess.CompletedProcess(argv, 0, PERF.encode())

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "samples"
            with patch("collect_read_qos.sys.platform", "linux"), \
                 patch("collect_read_qos_info.subprocess.run", side_effect=run), \
                 patch("collect_read_qos.resolve_debug_dir", return_value=Path(directory)), \
                 patch("collect_read_qos.time.sleep"):
                self.assertEqual(main(["--eid", CPU_EID,
                                       "--samples", "3", "--output", str(output)]), 1)
            device_output = output / "udmac0d1e2"
            log = (device_output / "sampling.log").read_text(encoding="utf-8")
            self.assertIn("status=incomplete device=udmac0d1e2 valid_perf_samples=1 requested_perf_samples=3", log)
            self.assertIn("status=timeout", log)
            self.assertEqual((device_output / "perf_0001.txt").read_text(encoding="utf-8"), "partial read\nTIMEOUT\n")
            self.assertFalse((device_output / "perf_0002.txt").exists())

    def test_distinct_port_groups_sample_concurrently_into_separate_directories(self):
        barrier = threading.Barrier(2, timeout=5)
        other_perf = PERF.replace("64000", "128000")

        def run(argv, **kwargs):
            if argv == ["urma_admin", "show", "--all", "--whole"]:
                return subprocess.CompletedProcess(argv, 0, MULTI_DEVICES.encode())
            self.assertEqual(argv[0], "cat")
            device = Path(argv[1]).parent.name
            self.assertIn(device, ("udmac0d1e2", "udmac0d0e3"))
            if Path(argv[1]).name == "caps_info":
                output = CAPS if device == "udmac0d1e2" else other_caps
                return subprocess.CompletedProcess(argv, 0, output.encode())
            # A serial implementation cannot get both readers past this barrier.
            barrier.wait()
            output = PERF if device == "udmac0d1e2" else other_perf
            return subprocess.CompletedProcess(argv, 0, output.encode())

        # A difference in any field must keep the groups separate, even for overlapping bitmaps.
        for other_caps in (OTHER_CAPS, CAPS.replace("die_id: 1", "die_id: 0"), CAPS.replace("0x30", "0x10")):
            with self.subTest(caps=other_caps), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "samples"
                with patch("collect_read_qos.sys.platform", "linux"), \
                     patch("collect_read_qos_info.subprocess.run", side_effect=run) as command, \
                     patch("collect_read_qos.resolve_debug_dir",
                           side_effect=lambda device, root: Path(directory) / "debug" / device.name), \
                     patch("collect_read_qos.time.sleep"), redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["--urma-device", "udmac0d1e2", "udmac0d0e3",
                                           "--eid", CPU_EID, OTHER_EID, "--samples", "2",
                                           "--output", str(output)]), 0)
                self.assertEqual(command.call_count, 7)
                self.assertEqual({path.name for path in output.iterdir()},
                                 {"sampling.log", "udmac0d1e2", "udmac0d0e3"})
                for device, eid, expected in (("udmac0d1e2", CPU_EID, PERF),
                                              ("udmac0d0e3", OTHER_EID, other_perf)):
                    device_output = output / device
                    self.assertEqual({path.name for path in device_output.iterdir()},
                                     {"sampling.log", "perf_0000.txt", "perf_0001.txt"})
                    for sample in range(2):
                        self.assertEqual((device_output / f"perf_{sample:04d}.txt").read_text(encoding="utf-8"), expected)
                    log = (device_output / "sampling.log").read_text(encoding="utf-8")
                    self.assertIn(f"eid={eid}", log)
                    self.assertNotIn("SHARED_PORT_GROUP", log)
                    self.assertIn(f"status=complete device={device} valid_perf_samples=2", log)
                self.assertIn("completed_devices=2 requested_devices=2",
                              (output / "sampling.log").read_text(encoding="utf-8"))

    def test_shared_ports_are_read_once_across_debugfs_paths_with_identical_data_and_timestamps(self):
        def run(argv, **kwargs):
            if argv[0] == "urma_admin":
                output = MULTI_DEVICES
            elif Path(argv[1]).name == "caps_info":
                output = CAPS if Path(argv[1]).parent.name == "00007" else CAPS.replace("0x30", "48")
            else:
                self.assertEqual(Path(argv[1]).parent.name, "00007")
                output = PERF
            return subprocess.CompletedProcess(argv, 0, output.encode())

        for explicit, shared_path in ((True, True), (True, False), (False, False)):
            with self.subTest(explicit=explicit, shared_path=shared_path), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "samples"
                first_debug = Path(directory) / "00007"
                second_debug = first_debug if shared_path else Path(directory) / "00008"
                arguments = ["--eid", CPU_EID, "--eid", OTHER_EID, "--samples", "2", "--output", str(output)]
                if explicit:
                    arguments += ["--urma-device", "udmac0d1e2", "--urma-device", "udmac0d0e3"]
                with patch("collect_read_qos.sys.platform", "linux"), \
                     patch("collect_read_qos_info.subprocess.run", side_effect=run) as command, \
                     patch("collect_read_qos.resolve_debug_dir", side_effect=[first_debug, second_debug]), \
                     patch("collect_read_qos.time.sleep"), redirect_stdout(io.StringIO()):
                    self.assertEqual(main(arguments), 0)
                self.assertEqual(command.call_count, 5)
                first, second = output / "udmac0d1e2", output / "udmac0d0e3"
                for sample in range(2):
                    filename = f"perf_{sample:04d}.txt"
                    self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())
                first_log = (first / "sampling.log").read_text(encoding="utf-8")
                second_log = (second / "sampling.log").read_text(encoding="utf-8")
                for log in (first_log, second_log):
                    self.assertIn("SHARED_PORT_GROUP devices=udmac0d1e2,udmac0d0e3", log)
                    self.assertIn("chip_id=1 die_id=1 logic_port_bitmap=0x30", log)
                    self.assertIn(f"source_device=udmac0d1e2 path={first_debug / 'perf_stats'}", log)
                self.assertIn(f"perf_stats={second_debug / 'perf_stats'}", second_log)
                root_log = (output / "sampling.log").read_text(encoding="utf-8")
                self.assertEqual(root_log.count("SAMPLING_READY"), 1)
                self.assertEqual([line for line in first_log.splitlines() if line.startswith("COMMAND")],
                                 [line for line in second_log.splitlines() if line.startswith("COMMAND")])

    def test_one_device_failure_does_not_stop_other_device(self):
        for stage in ("device_identity", "debugfs_resolution", "perf_stats_read", "perf_stats_parse", "output_write"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "samples"
                reads = []

                def run(argv, **kwargs):
                    code, text = 0, MULTI_DEVICES
                    if argv[0] == "cat":
                        device = Path(argv[1]).parent.name
                        if Path(argv[1]).name == "caps_info":
                            text = CAPS if device == "udmac0d1e2" else OTHER_CAPS
                            return subprocess.CompletedProcess(argv, 0, text.encode())
                        reads.append(device)
                        text = PERF
                        if device == "udmac0d0e3":
                            if stage == "perf_stats_read":
                                code, text = 1, "Permission denied"
                            elif stage == "perf_stats_parse":
                                text = "unknown bandwidth format"
                    return subprocess.CompletedProcess(argv, code, text.encode())

                def resolve(device, root):
                    if stage == "debugfs_resolution" and device.name == "udmac0d0e3":
                        raise ValueError("Controller debugfs unavailable")
                    return Path(directory) / "debug" / device.name

                original_write = Path.write_text

                def write(path, *args, **kwargs):
                    if stage == "output_write" and path.parent.name == "udmac0d0e3" and path.name.startswith("perf_"):
                        raise OSError("Cannot write raw sample")
                    return original_write(path, *args, **kwargs)

                other_eid = CPU_EID if stage == "device_identity" else OTHER_EID
                with patch("collect_read_qos.sys.platform", "linux"), \
                     patch("collect_read_qos_info.subprocess.run", side_effect=run), \
                     patch("collect_read_qos.resolve_debug_dir", side_effect=resolve), \
                     patch("collect_read_qos.time.sleep"), \
                     patch.object(Path, "write_text", autospec=True, side_effect=write), \
                     redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["--urma-device", "udmac0d1e2", "udmac0d0e3",
                                           "--eid", CPU_EID, other_eid, "--samples", "2",
                                           "--output", str(output)]), 1)
                self.assertEqual(reads.count("udmac0d1e2"), 2)
                healthy_log = (output / "udmac0d1e2/sampling.log").read_text(encoding="utf-8")
                self.assertIn("status=complete device=udmac0d1e2 valid_perf_samples=2", healthy_log)
                root_log = (output / "sampling.log").read_text(encoding="utf-8")
                self.assertIn("COLLECTION_DONE status=incomplete completed_devices=1 requested_devices=2", root_log)
                if stage in ("device_identity", "debugfs_resolution"):
                    self.assertIn(f"stage={stage}", root_log)
                    self.assertEqual(reads.count("udmac0d0e3"), 0)
                    self.assertFalse((output / "udmac0d0e3").exists())
                else:
                    failed_log = (output / "udmac0d0e3/sampling.log").read_text(encoding="utf-8")
                    self.assertIn(f"stage={stage}", failed_log)
                    self.assertIn("status=incomplete device=udmac0d0e3 valid_perf_samples=0", failed_log)
                    self.assertEqual(reads.count("udmac0d0e3"), 1)

    def test_unverified_port_group_is_skipped_while_healthy_device_keeps_sampling(self):
        for failure in ("read", "timeout", "truncated", "missing_field"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "samples"
                reads = []

                def run(argv, **kwargs):
                    if argv[0] == "urma_admin":
                        return subprocess.CompletedProcess(argv, 0, MULTI_DEVICES.encode())
                    path = Path(argv[1])
                    code, text = 0, CAPS
                    if path.name == "caps_info":
                        if path.parent.name == "udmac0d0e3":
                            if failure == "read":
                                code, text = 1, "Permission denied"
                            elif failure == "timeout":
                                raise subprocess.TimeoutExpired(argv, 15)
                            elif failure == "truncated":
                                text += "x" * (2 * 1024 * 1024)
                            else:
                                text = CAPS.replace("chip_id: 1\n", "")
                    else:
                        self.assertEqual(path.name, "perf_stats")
                        reads.append(path.parent.name)
                        text = PERF
                    return subprocess.CompletedProcess(argv, code, text.encode())

                with patch("collect_read_qos.sys.platform", "linux"), \
                     patch("collect_read_qos_info.subprocess.run", side_effect=run), \
                     patch("collect_read_qos.resolve_debug_dir",
                           side_effect=lambda device, root: Path(directory) / device.name), \
                     patch("collect_read_qos.time.sleep"), redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["--urma-device", "udmac0d1e2", "udmac0d0e3",
                                           "--eid", CPU_EID, OTHER_EID, "--samples", "2",
                                           "--output", str(output)]), 1)
                self.assertEqual(reads, ["udmac0d1e2", "udmac0d1e2"])
                self.assertFalse((output / "udmac0d0e3").exists())
                root_log = (output / "sampling.log").read_text(encoding="utf-8")
                self.assertIn("stage=port_group_identity", root_log)
                self.assertIn("COLLECTION_DONE status=incomplete completed_devices=1 requested_devices=2", root_log)

    def test_invalid_multi_device_arguments_fail_before_creating_output(self):
        options = (
            ["--urma-device", "udmac0d1e2", "udmac0d0e3", "--eid", CPU_EID],
            ["--urma-device", "udmac0d1e2", "--eid", CPU_EID, OTHER_EID],
            ["--urma-device", "udmac0d1e2", "udmac0d1e2", "--eid", CPU_EID, CPU_EID],
            ["--urma-device", "../outside", "--eid", CPU_EID],
            ["--urma-device", "..", "--eid", CPU_EID],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            for arguments in options:
                with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), \
                     patch("collect_read_qos_info.subprocess.run") as command:
                    with self.assertRaises(SystemExit) as error:
                        main(arguments + ["--output", str(output)])
                    self.assertEqual(error.exception.code, 2)
                    command.assert_not_called()
                    self.assertFalse(output.exists())

    def test_information_device_failure_still_saves_report_without_sampling_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "info"
            with patch("collect_read_qos_info.sys.platform", "linux"), \
                 patch("collect_read_qos_info.subprocess.run", return_value=subprocess.CompletedProcess([], 0, b"")):
                self.assertEqual(info_main(["--eid", CPU_EID,
                                            "--output", str(output)]), 1)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["local_device_error"]["stage"], "device_identity")
            self.assertNotIn("sampling_error", report)
            self.assertNotIn("valid_perf_samples", report)


    def test_removed_options_fail_before_creating_output(self):
        options = (["--role", "npu"], ["--role", "cpu"], ["--device-id", "7"],
                   ["--build-dir", "build-npu"], ["--hccp-include", "sdk/include"],
                   ["--jetty-id", "123"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            for entry in (info_main, main):
                for option in options:
                    with self.subTest(entry=entry.__module__, option=option), \
                         redirect_stderr(io.StringIO()), \
                         patch("collect_read_qos_info.subprocess.run") as command:
                        with self.assertRaises(SystemExit) as error:
                            entry(["--eid", CPU_EID, "--output", str(output)] + option)
                        self.assertEqual(error.exception.code, 2)
                        command.assert_not_called()
                        self.assertFalse(output.exists())

    def test_source_excerpts_only_include_cpu_driver_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "udma_jetty.c").write_text("ctx->sl = cfg->priority;\n", encoding="utf-8")
            (source / "hccp_ctx.h").write_text("priority SL RaGetDevBaseAttr\n", encoding="utf-8")
            (source / "hccp_types.h").write_text("priority SL\n", encoding="utf-8")
            output = Path(directory) / "info"
            records = collect_sources(Collector(output, 15), [source])
            self.assertEqual(len(records), 1)
            self.assertEqual(Path(records[0]["path"]).name, "udma_jetty.c")
            self.assertEqual({path.name for path in output.iterdir()}, {"source_000_udma_jetty.c.txt"})
            self.assertIn("1: ctx->sl = cfg->priority;",
                          (output / records[0]["file"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
