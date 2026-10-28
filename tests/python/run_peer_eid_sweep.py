#!/usr/bin/env python3
"""Compare shared and distinct CPU EIDs at 2/4/8 AIV cores on real hardware."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess

from npu_eids import parse_npu_eids


def build_cases(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    if not args.cores or not args.jettys or any(c < 1 or c > 48 for c in args.cores):
        raise ValueError("cores must be in [1, 48]")
    if any(j < 1 or j > 64 for j in args.jettys):
        raise ValueError("jettys must be in [1, 64]")
    if len(args.peer_eids) < max(args.cores):
        raise ValueError("provide at least one distinct CPU EID per core in the largest case")
    if not 0 <= args.max_inflight_total <= 0xFFFFFFFF or (
        args.max_inflight_total and args.max_inflight_total < 2 * max(args.cores)
    ):
        raise ValueError("max-inflight-total must be zero or at least twice the largest core count")
    if args.requests < max(args.cores) * ((max(args.jettys) - 1) * 128 + 1):
        raise ValueError("requests must cover every core and Jetty (at least cores * ((jettys-1)*128+1))")
    runner = Path(__file__).resolve().parents[1] / "run_remote_cpu_batch_get_perf.sh"
    cases = []
    for cores in args.cores:
        for jettys in args.jettys:
            for mode in ("shared", "distinct"):
                eids = [args.peer_eids[0]] * cores if mode == "shared" else args.peer_eids[:cores]
                command = [
                    "bash", str(runner), str(args.build_dir.resolve()),
                    "--aiv-cores", str(cores), "--jettys-per-core", str(jettys),
                    "--peer-eids-by-core", ",".join(eids),
                    "--max-inflight-total", str(args.max_inflight_total),
                    "--request-length", str(args.requests), "--miss-count", str(args.requests),
                    "--table-capacity", str(args.table_capacity),
                    "--token-bytes", str(args.token_bytes),
                    "--tokens-per-block", str(args.tokens_per_block),
                    "--warmup", str(args.warmup), "--iterations", str(args.iterations),
                    "--seed", "20260903", "--kernel-profile",
                ]
                if args.npu_eid:
                    command += ["--npu-eids", args.npu_eid]
                cases.append((f"cores_{cores}_jettys_{jettys}_{mode}", command))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--peer-eids", type=parse_npu_eids,
                        default=parse_npu_eids(os.environ.get("KVAF_PEER_EID_CANDIDATES")))
    parser.add_argument("--npu-eid", default="", help="one fixed NPU EID for comparable runs")
    parser.add_argument("--cores", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--jettys", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-inflight-total", type=int, default=2048)
    parser.add_argument("--requests", type=int, default=8192)
    parser.add_argument("--table-capacity", type=int, default=64)
    parser.add_argument("--token-bytes", type=int, default=656)
    parser.add_argument("--tokens-per-block", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("peer_eid_sweep_logs"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        local_eids = parse_npu_eids(args.npu_eid or os.environ.get("KVAF_NPU_EIDS"))
        if len(local_eids) != 1:
            raise ValueError("set --npu-eid (or KVAF_NPU_EIDS) to exactly one NPU EID for the sweep")
        args.npu_eid = local_eids[0]
        if min(args.table_capacity, args.token_bytes, args.tokens_per_block, args.iterations) <= 0 or args.warmup < 0:
            raise ValueError("shape and iteration counts must be positive; warmup must be non-negative")
        cases = build_cases(args)
    except ValueError as error:
        parser.error(str(error))
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, command in cases:
        print(f"{name}: {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        log = args.output_dir / f"{name}.log"
        # Refuse to overwrite an earlier experiment's results.
        with log.open("x") as output:
            subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
        contents = log.read_text()
        if ("BATCH_GET_PERF_CORRECTNESS_PASS" not in contents or
                "BATCH_GET_PERF_PHASE_DONE phase=batch_get" not in contents or
                "PEER_ENDPOINT_READ_VALIDATED round=1" not in contents):
            raise RuntimeError(f"case skipped or validation/results incomplete: {log}")
        print(f"PASS {name}: {log}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
