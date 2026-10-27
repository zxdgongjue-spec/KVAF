import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from collect_read_qos import (
    configured_response_pairs, main, normalize_eid, parse_devices,
    parse_perf_stats, resolve_debug_dir, select_device,
)


CPU_EID = "00000000003f020000100000df080b00"
NPU_EID = "00000000003f030000100000df00eb00"
DEVICES = """name : udmac0d1e2
eid0 : 0000:0000:003f:0200:0010:0000:df08:0b00
name : udmac0d0e3
eid0 : 0000:0000:0000:0000:0000:0000:0000:0000
"""
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
    def test_same_device_name_does_not_identify_hdc_npu(self):
        devices = parse_devices(DEVICES)
        self.assertEqual(select_device(devices, CPU_EID), "udmac0d1e2")
        with self.assertRaises(ValueError):
            select_device(devices, NPU_EID, "udmac0d0e3")
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

    def test_collection_does_not_promote_vl_activity_to_sl_inheritance(self):
        def run(argv, **kwargs):
            if argv[:2] == ["urma_admin", "show"] and "--all" in argv:
                output = DEVICES
            elif argv[0] == "cat" and str(argv[1]).endswith("perf_stats"):
                output = PERF
            elif argv[0] == "cat" and str(argv[1]).endswith("adev_qos"):
                output = "ctp_req_vl: 8 4\nctp_resp_vl_offset: 1\n"
            else:
                output = "unavailable in fixture"
            return subprocess.CompletedProcess(argv, 0, output.encode())

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            with patch("collect_read_qos.sys.platform", "linux"), \
                 patch("collect_read_qos.subprocess.run", side_effect=run), \
                 patch("collect_read_qos.resolve_debug_dir", return_value=Path(directory)):
                self.assertEqual(main(["--role", "cpu", "--eid", CPU_EID,
                                       "--output", str(output)]), 0)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["valid_perf_samples"], 1)
            self.assertEqual(report["configured_ctp_req_resp_vl_pairs"], [[8, 9], [4, 5]])
            self.assertEqual(report["response_inherits_request_sl"], "unverified")
            self.assertEqual(report["response_packet_sl"], "unverified")


if __name__ == "__main__":
    unittest.main()
