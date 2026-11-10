#!/usr/bin/env python3
"""Benchmark CPU push/scatter with the shared full-pool random workload."""
from benchmark_gather_read import main


if __name__ == "__main__":
    main(mode="cpu_push_scatter")
