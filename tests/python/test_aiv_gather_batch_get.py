#!/usr/bin/env python3
"""AIV Gather correctness test. Edit the defaults below or override them via the environment."""
import argparse
import os

from aiv_batch_get_test_common import run

# ---- Edit test settings here. ----------------------------------------------
DEVICE_ID = 0
NPU_CORE_COUNT = 2
REQUEST_COUNT = 8192
TOKEN_BYTES = 656
GATHER_SGL_SIZE = 6
GRAPH_MODE = False              # Use --graph for ACL Graph replay.
GATHER_CHUNK_BYTES = None       # None uses Runtime defaults. Set 0 for whole-batch ACK.
GATHER_PREPARE_AHEAD = None     # None uses Runtime defaults. Set 1 to prebuild READ WQEs.
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--graph", dest="graph", action="store_true",
                      help="capture once and validate each ACL Graph replay")
    mode.add_argument("--eager", dest="graph", action="store_false",
                      help="run the eager correctness test")
    parser.set_defaults(graph=None)
    args = parser.parse_args(argv)
    for name, value in (
        ("DEVICE_ID", DEVICE_ID),
        ("KVAF_NPU_CORE_COUNT", NPU_CORE_COUNT),
        ("KVAF_NPU_REQUESTS", REQUEST_COUNT),
        ("KVAF_NPU_TOKEN_BYTES", TOKEN_BYTES),
        ("KVAF_AIV_GATHER_SGL_SIZE", GATHER_SGL_SIZE),
        ("KVAF_AIV_GATHER_GRAPH", int(GRAPH_MODE)),
        ("KVAF_AIV_GATHER_CHUNK_BYTES", GATHER_CHUNK_BYTES),
        ("KVAF_AIV_GATHER_PREPARE_AHEAD", GATHER_PREPARE_AHEAD),
    ):
        if value is not None:
            os.environ.setdefault(name, str(value))
    if args.graph is not None:
        os.environ["KVAF_AIV_GATHER_GRAPH"] = str(int(args.graph))
    return run("gather_read")


if __name__ == "__main__":
    raise SystemExit(main())
