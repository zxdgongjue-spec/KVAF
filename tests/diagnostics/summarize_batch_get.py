#!/usr/bin/env python3
"""Export baseline/lean BatchGet logs; no benchmark is executed."""

from summarize_small_post import main as export_logs


def main(argv=None):
    return export_logs(argv, experiment="lean")


if __name__ == "__main__":
    raise SystemExit(main())
