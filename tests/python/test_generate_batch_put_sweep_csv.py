#!/usr/bin/env python3
"""Tests for rebuilding BatchPut sweep CSV files from retained logs."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_batch_put_sweep_csv


def result_line(entry_bytes: int, row_count: int, total_cycles: int) -> str:
    execute_cycles = total_cycles - total_cycles // 10
    non_execute_cycles = total_cycles - execute_cycles
    return (
        "BATCH_PUT_PERF_CONFIG "
        f"row_count={row_count} entry_bytes={entry_bytes} "
        f"payload_bytes={entry_bytes * row_count} aiv_cores=32 jettys_per_core=2 "
        "warmup=1 iterations=10 kernel_profile=1 host_profile=0\n"
        "PERF_RESULT operation=batch_put_kernel samples=10 warmup_excluded=1 "
        f"row_count={row_count} entry_bytes={entry_bytes} "
        f"mean_execute_cycles={execute_cycles}.000 "
        f"p50_execute_cycles={execute_cycles}.000 "
        f"p90_execute_cycles={execute_cycles}.000 "
        f"p99_execute_cycles={execute_cycles}.000 "
        f"mean_non_execute_cycles={non_execute_cycles}.000 "
        f"p50_non_execute_cycles={non_execute_cycles}.000 "
        f"p90_non_execute_cycles={non_execute_cycles}.000 "
        f"p99_non_execute_cycles={non_execute_cycles}.000 "
        "aggregation=slowest_total_elapsed_core\n"
    )


class GenerateBatchPutSweepCsvTest(unittest.TestCase):
    def test_rebuild_includes_stage_level_host_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_dir = Path(temporary)
            output = log_dir / "summary.csv"
            host_fields = " ".join(
                f"{name}={index + 1}.000"
                for index, name in enumerate(generate_batch_put_sweep_csv.HOST_FIELDS)
            )
            (log_dir / "host_profile.log").write_text(
                result_line(1024, 64, 1000)
                + "PERF_RESULT operation=batch_put_host "
                + host_fields
                + "\n"
            )

            status = generate_batch_put_sweep_csv.main(
                [str(log_dir), "--output", str(output), "--clock-mhz", "1000"]
            )

            self.assertEqual(status, 0)
            with output.open(newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["host_profile"], "1")
            self.assertEqual(row["mean_prepare_put_us"], "6.000")
            self.assertEqual(row["mean_commit_put_us"], "13.000")

    def test_rebuild_skips_incomplete_and_prefers_latest_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_dir = Path(temporary)
            old = log_dir / "old.log"
            new = log_dir / "new.log"
            second = log_dir / "second.log"
            incomplete = log_dir / "incomplete.log"
            output = log_dir / "summary.csv"

            old.write_text(result_line(1024, 64, 1000))
            new.write_text(result_line(1024, 64, 2000))
            second.write_text(result_line(128, 32, 500))
            incomplete.write_text("BATCH_PUT_PERF_PHASE_BEGIN phase=batch_put\n")
            os.utime(old, ns=(1_000_000_000, 1_000_000_000))
            os.utime(new, ns=(2_000_000_000, 2_000_000_000))

            status = generate_batch_put_sweep_csv.main(
                [str(log_dir), "--output", str(output), "--clock-mhz", "1000"]
            )

            self.assertEqual(status, 0)
            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                [(row["entry_bytes"], row["row_count"]) for row in rows],
                [("128", "32"), ("1024", "64")],
            )
            self.assertEqual(rows[1]["mean_kernel_execute_cycles"], "1800.000")
            self.assertEqual(rows[1]["mean_kernel_non_execute_cycles"], "200.000")
            self.assertEqual(rows[1]["mean_us"], "2.000")
            self.assertEqual(rows[1]["mean_GBps"], "32.768")


if __name__ == "__main__":
    unittest.main()
