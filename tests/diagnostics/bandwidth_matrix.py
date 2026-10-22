#!/usr/bin/env python3
"""Controlled queue, request-size and locality sweeps; separate process per case."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys


def cases(group):
    # Constant 2 AIVs. 2, 4, 8, 16 total Jettys.
    queue = [(f'queue_j{j}', 656, 5000, j, 'random') for j in (1, 2, 4, 8)]
    # Approximately constant 3.28 MB payload; round request count, report exact bytes.
    size = [(f'size_t{t}', t, round(3280000/t), 1, 'random') for t in (656, 2048, 4096, 8192)]
    locality = [(f'locality_{p}', 656, 5000, 1, p) for p in ('random', 'contiguous', 'shuffled-contiguous')]
    return {'queues': queue, 'sizes': size, 'locality': locality, 'all': queue+size+locality}[group]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--clock-mhz', type=int, required=True)
    p.add_argument('--group', choices=('queues', 'sizes', 'locality', 'all'), default='all')
    p.add_argument('--iterations', type=int, default=30)
    p.add_argument('--table-capacity', type=int, default=65536,
                   help='Constant logical remote table size; largest token case needs 64 GiB DDR')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.clock_mhz <= 0 or a.iterations <= 0 or a.table_capacity < 40:
        p.error('Invalid clock, iterations or table capacity')
    if a.output.exists():
        p.error('Use a new output directory')
    a.output.mkdir(parents=True)
    runner = Path(__file__).with_name('bandwidth_scopes.py')
    summary = []
    failed = False
    for name, token, count, jettys, pattern in cases(a.group):
        output = a.output/(name+'.json')
        command = [sys.executable, str(runner), '--build', str(a.build.resolve()),
                   '--output', str(output), '--clock-mhz', str(a.clock_mhz),
                   '--token-bytes', str(token), '--miss-count', str(count),
                   '--request-length', '32768', '--table-capacity', str(a.table_capacity),
                   '--aiv-cores', '2', '--jettys-per-core', str(jettys),
                   '--source-access-pattern', pattern, '--output-access-pattern', 'contiguous',
                   '--warmup', '3', '--iterations', str(a.iterations)]
        (a.output/(name+'.command.json')).write_text(json.dumps(command, indent=2))
        print(name, 'payload_bytes=', token*count, 'total_jettys=', 2*jettys, flush=True)
        if a.dry_run:
            continue
        with (a.output/(name+'.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        row = dict(case=name, returncode=result.returncode, token_bytes=token,
                   requests=count, total_jettys=2*jettys, pattern=pattern, payload_bytes=token*count)
        if result.returncode == 0 and output.exists():
            data = json.loads(output.read_text())
            for field in ('host_us', 'event_us', 'kernel_wall_us', 'transfer_window_us'):
                value = statistics.median(s[field] for s in data['samples'])
                row[field+'_p50'] = value
                row[field+'_GBps'] = token*count/value/1000 if value > 0 else None
        else:
            failed = True
        summary.append(row)
        (a.output/'summary.json').write_text(json.dumps(summary, indent=2))
        keys = list(dict.fromkeys(k for r in summary for k in r))
        with (a.output/'summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(summary)
        print(json.dumps(row), flush=True)
        if failed:
            print('Stopped after failure; inspect case log before reusing device.', flush=True)
            break
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
