"""Portable CLI coverage for per-core CPU EID mappings."""

import os
import unittest
from unittest.mock import patch

from npu_eids import parse_peer_eids_by_core
import test_npu_batch_get_perf as perf
import run_peer_eid_sweep as sweep
from argparse import Namespace
from pathlib import Path


class PeerEidConfigTests(unittest.TestCase):
    def parse(self, *arguments):
        with patch.dict(os.environ, {}, clear=True):
            return perf.parse_args(list(arguments))

    def test_core_counts_and_shared_controls(self):
        for cores in (2, 4, 8):
            for distinct in (False, True):
                eids = [f"{i + 1 if distinct else 1:032x}" for i in range(cores)]
                args = self.parse("--aiv-cores", str(cores), "--peer-eids-by-core", ",".join(eids))
                self.assertEqual(args.peer_eids_by_core, eids)
                self.assertIsNone(perf.validate_args(args))

    def test_invalid_combinations(self):
        mapping = ",".join(["1" * 32] * 4)
        for extra in (("--aiv-cores", "8"), ("--adaptive-ub-device",),
                      ("--npu-eids", "1" * 32 + "," + "2" * 32)):
            args = self.parse("--aiv-cores", "4", "--peer-eids-by-core", mapping, *extra)
            self.assertIn("peer-eids-by-core", perf.validate_args(args))

    def test_encoding_environment_and_override(self):
        self.assertEqual(parse_peer_eids_by_core(None), [])
        self.assertEqual(parse_peer_eids_by_core("A" * 32 + "," + "A" * 32), ["a" * 32] * 2)
        for invalid in ("abc", "0" * 32, "z" * 32, "1" * 32 + ",", "," + "1" * 32):
            with self.assertRaises(ValueError):
                parse_peer_eids_by_core(invalid)
        with patch.dict(os.environ, {"KVAF_PEER_EIDS_BY_CORE": ",".join(["1" * 32] * 2)}, clear=True):
            self.assertEqual(perf.parse_args([]).peer_eids_by_core, ["1" * 32] * 2)
            self.assertEqual(perf.parse_args(["--peer-eids-by-core", ""]).peer_eids_by_core, [])

    def test_sweep_keeps_workload_and_inflight_fixed(self):
        args = Namespace(cores=[2, 4, 8], jettys=[1, 2, 4],
                         peer_eids=[f"{i + 1:032x}" for i in range(8)],
                         npu_eid="a" * 32, max_inflight_total=2048, requests=8192,
                         build_dir=Path("/tmp/build"), table_capacity=64,
                         token_bytes=656, tokens_per_block=128, warmup=1, iterations=5)
        cases = sweep.build_cases(args)
        self.assertEqual(len(cases), 18)
        for i in range(0, len(cases), 2):
            baseline, distinct = cases[i][1], cases[i + 1][1]
            location = baseline.index("--peer-eids-by-core") + 1
            self.assertEqual(len(set(baseline[location].split(","))), 1)
            core_count = int(distinct[distinct.index("--aiv-cores") + 1])
            self.assertEqual(len(set(distinct[location].split(","))), core_count)
            baseline[location] = distinct[location]
            self.assertEqual(baseline, distinct)
        args.peer_eids.pop()
        with self.assertRaises(ValueError):
            sweep.build_cases(args)


if __name__ == "__main__":
    unittest.main()
