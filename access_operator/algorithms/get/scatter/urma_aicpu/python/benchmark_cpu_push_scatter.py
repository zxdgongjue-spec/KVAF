#!/usr/bin/env python3
"""Benchmark CPU push/scatter with the shared full-pool random workload."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "common" / "aicpu" / "python"))
from benchmark_gather_read import main


if __name__ == "__main__":
    main(mode="cpu_push_scatter")
