from contextlib import redirect_stderr, redirect_stdout
import io
from itertools import count
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import npu_port_stat as stat


# Format from the official hccn_tool parser/tests; values below are synthetic.
PORTS = """+-------+--------+-------+----------+-------------+------------+
| UdieID | PortID | Speed | PortType | Link Status | Media Type |
+-------+--------+-------+----------+-------------+------------+
| 0      | 4      | 200   | ETH      | UP          | Optical    |
| 1      | 8      | 200   | UB       | DOWN        | Optical    |
| 1      | 9      | 200   | UB       | UP          | Optical    |
+-------+--------+-------+----------+-------------+------------+
"""
BW = "Bandwidth TX: 1.25 MB/sec\nBandwidth RX: 0.00 MB/sec\n"


class NpuPortStatTest(unittest.TestCase):
    def test_port_discovery_preserves_die_and_excludes_ethernet_and_second_table(self):
        self.assertEqual(stat.parse_ports(PORTS + PORTS.replace("9 ", "10 ")),
                         {(1, 8): "DOWN", (1, 9): "UP"})
        with self.assertRaises(ValueError):
            stat.parse_ports(PORTS.replace("UB", "ETH") + PORTS)

    def test_bandwidth_is_already_a_rate_and_zero_is_valid(self):
        self.assertEqual(stat.parse_bandwidth(BW), (1.25, 0.0))
        for output in ("", BW.splitlines()[0], BW.replace("MB/sec", "Gbps"),
                       BW.replace("1.25", "NaN"), BW.replace("1.25", "-1"), BW + BW):
            with self.subTest(output=output), self.assertRaises(ValueError):
                stat.parse_bandwidth(output)

    def test_counters_preserve_large_integer_precision_and_missing_values(self):
        parsed = stat.parse_counters("tx_busi_flit_num: 9007199254740993\n"
                                     "rx_busi_flit_num: 0x10\n"
                                     "ub_umoc_ctph_cnt_tx: 000012\n"
                                     "ub_umoc_ctph_cnt_rx: N/A\n")
        self.assertEqual(parsed, {"tx_busi_flit_num": 9007199254740993,
                                  "rx_busi_flit_num": 16, "ub_umoc_ctph_cnt_tx": 12})
        with self.assertRaises(ValueError):
            stat.parse_counters("rx_busi_flit_num: -1\n")

    def test_rates_use_per_port_elapsed_time_and_do_not_guess_wrap(self):
        before = (10, {"tx_busi_flit_num": 100, "ub_umoc_ctph_cnt_tx": 10,
                       "rx_busi_flit_num": 50})
        after = (12, {"tx_busi_flit_num": 140, "ub_umoc_ctph_cnt_tx": 10,
                      "rx_busi_flit_num": 2})
        self.assertEqual(stat.counter_rates(before, after),
                         {"counter_dt": 2, "tx_flit_ps": 20, "tx_ctp_pps": 0})
        self.assertEqual(stat.counter_rates(None, after), {})
        self.assertEqual(stat.counter_rates(after, after), {})

    def test_main_discovers_ports_samples_and_prints_counters(self):
        calls, samples = [], 0

        def run(tool, args, timeout):
            nonlocal samples
            calls.append(args)
            if "-dev_info" in args:
                return PORTS
            self.assertEqual(args[2:8], ["-i", 7, "-u", 1, "-p", 9])
            if "-stat" in args:
                samples += 1
                return f"tx_busi_flit_num: {samples * 100}\n"
            self.assertEqual(args[-2:], ["-time", 100])
            return BW

        terminal = io.StringIO()
        with patch.object(stat, "run_hccn", side_effect=run), \
                patch.object(stat.time, "sleep"), \
                patch.object(stat.time, "monotonic", side_effect=count(10)), \
                redirect_stdout(terminal), redirect_stderr(io.StringIO()):
            status = stat.main(["--device-id", "7", "--counters"])
        self.assertEqual(status, 0)
        row = terminal.getvalue().splitlines()[2].split()
        self.assertEqual(row[:3], ["7", "1", "9"])
        self.assertEqual(row[4:6], ["1.250", "0.000"])
        self.assertEqual(row[7:9], ["N/A", "N/A"])
        self.assertGreater(float(row[9]), 0)
        self.assertEqual(len(calls), 4)  # discovery, baseline, stat, bandwidth

    def test_default_queries_in_parallel_and_prints_one_complete_sample(self):
        rendezvous = Barrier(2)
        completed, writes = [], []

        class Output(io.StringIO):
            def write(self, text):
                if text.strip():
                    writes.append((text, len(completed)))
                return super().write(text)

        def run(tool, args, timeout):
            if "-dev_info" in args:
                return PORTS.replace("DOWN", "UP")
            rendezvous.wait(timeout=3)  # Serial collection cannot pass this barrier.
            completed.append(args[7])
            return BW

        with patch.object(stat, "run_hccn", side_effect=run), \
                redirect_stdout(Output()), redirect_stderr(io.StringIO()):
            status = stat.main(["--device-id", "7"])
        self.assertEqual(status, 0)
        self.assertEqual(sorted(completed), [8, 9])
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][1], 2)
        self.assertIn("workers=8", writes[0][0])
        self.assertEqual(len(writes[0][0].strip().splitlines()), 4)

    def test_interval_includes_collection_time_and_does_not_sleep_after_overrun(self):
        for duration, expected_starts, expected_sleeps in (
                (0.25, [0, 1], [0.75]), (1.5, [0, 1.5], [])):
            with self.subTest(duration=duration):
                now, starts, sleeps = [0.0], [], []

                def collect(args, key):
                    starts.append(now[0])
                    now[0] += duration
                    return key, dict(query_seconds=duration, tx_MBps=0, rx_MBps=0, error=""), None

                def sleep(seconds):
                    sleeps.append(seconds)
                    now[0] += seconds

                args = SimpleNamespace(device_id=7, workers=1, counters=False, interval=1, count=2)
                with patch.object(stat, "collect_port", side_effect=collect), \
                        patch.object(stat.time, "monotonic", side_effect=lambda: now[0]), \
                        patch.object(stat.time, "sleep", side_effect=sleep), \
                        redirect_stdout(io.StringIO()):
                    self.assertEqual(stat.sample(args, [(1, 9)]), 0)
                self.assertEqual(starts, expected_starts)
                self.assertEqual(sleeps, expected_sleeps)

    def test_failed_sampling_exits_after_requested_count_and_records_no_fake_zero(self):
        def run(tool, args, timeout):
            if "-dev_info" in args:
                return PORTS
            raise RuntimeError("unsupported bandwidth query")

        terminal = io.StringIO()
        with patch.object(stat, "run_hccn", side_effect=run) as command, \
                patch.object(stat.time, "sleep"), redirect_stdout(terminal), \
                redirect_stderr(io.StringIO()):
            status = stat.main(["--device-id", "7", "-n", "2"])
        self.assertEqual(status, 1)
        self.assertEqual(command.call_count, 3)
        self.assertEqual(terminal.getvalue().count("N/A"), 4)

    def test_explicit_port_must_belong_to_selected_npu_and_be_ub(self):
        with patch.object(stat, "run_hccn", return_value=PORTS) as command, \
                redirect_stderr(io.StringIO()):
            self.assertEqual(stat.main(["--device-id", "7", "--ports", "0:4"]), 1)
        self.assertEqual(command.call_count, 1)


if __name__ == "__main__":
    unittest.main()
