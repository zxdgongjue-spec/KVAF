#!/usr/bin/env python3
"""Summarize saved client/server diagnostics without dependencies or device access."""
import argparse
import json
from pathlib import Path
import re
import statistics


def median(values):
    return round(statistics.median(values), 3) if values else None


def client_summary(root):
    rows = []
    for path in sorted(root.rglob('stride-*.json')):
        try:
            data = json.loads(path.read_text())
            samples = data['samples']
            payload = data['payload_bytes']
            config = data.get('config', {})
            row = dict(file=str(path), stride=data.get('record_stride'), samples=len(samples),
                       payload_bytes=payload, clock_mhz=data.get('assumed_clock_mhz'),
                       config={k: config[k] for k in ('miss_count','token_bytes','tokens_per_block','table_capacity',
                               'aiv_cores','jettys_per_core','source_access_pattern','seed') if k in config})
            for field in ('host_us','event_us','kernel_wall_us','transfer_window_us'):
                vals = [s[field] for s in samples if isinstance(s.get(field), (int,float))]
                row[field+'_p50'] = median(vals)
                if vals:
                    t = statistics.median(vals)
                    row[field+'_GBps'] = round(payload/t/1000, 3) if t > 0 else None
                    if field == 'kernel_wall_us':
                        row['kernel_us_min_max'] = [round(min(vals),3),round(max(vals),3)]
            rows.append(row)
        except (ValueError, KeyError, TypeError) as error:
            rows.append(dict(file=str(path), error=str(error)))
    return rows


def large_regions(path):
    regions = []
    current = None
    header = re.compile(r'^([0-9a-f]+)-([0-9a-f]+)\s')
    for line in path.read_text(errors='replace').splitlines():
        match = header.match(line)
        if match:
            if current and current.get('Size',0) >= 65536:
                regions.append(current)
            current = dict(address=match[1], mapping=line)
        elif current is not None:
            match = re.match(r'(Size|Rss|AnonHugePages|KernelPageSize|MMUPageSize|Private_Hugetlb|Shared_Hugetlb):\s+(\d+)',line)
            if match:
                current[match[1]] = int(match[2])
            elif line.startswith('VmFlags:'):
                current['flags'] = line.split(':',1)[1].strip()
    if current and current.get('Size',0) >= 65536:
        regions.append(current)
    return regions


def server_summary(root):
    # Directory may be one experiment or a parent containing several experiments.
    configs = sorted(root.rglob('config.txt'))
    result = []
    for config in configs:
        folder = config.parent
        if not (folder/'service.log').exists():
            continue
        snapshots = []
        for path in sorted(folder.glob('smaps-*.txt')):
            regions = large_regions(path)
            snapshots.append((sum(r.get('Rss',0) for r in regions), path, regions))
        row = dict(directory=str(folder), config=config.read_text().strip(), snapshots=len(snapshots))
        for name in ('thp-enabled.txt','numa-hardware.txt','numa-policy-check.log'):
            path = folder/name
            if path.exists():
                lines = path.read_text(errors='replace').splitlines()
                if name == 'numa-hardware.txt':
                    lines = [s for s in lines if ' cpus:' not in s]
                row[name] = lines[:24]
        lines = (folder/'service.log').read_text(errors='replace').splitlines()
        useful = [s for s in lines if re.search(r'REGION_|page_mode|[Ee][Rr][Rr][Oo][Rr]|FAILED|failed|[Ii]nvalid|[Dd]enied|not permitted|Cannot|cannot',s)]
        row['service_evidence'] = list(dict.fromkeys(useful))[:24]
        if snapshots:
            _, path, regions = max(snapshots, key=lambda x:x[0])
            row['selected_snapshot'] = path.name
            row['selection'] = 'Largest resident footprint in mappings >=64MiB; candidates, not proof of registered-region identity'
            numa = folder/path.name.replace('smaps-','numa_maps-')
            numa_lines = {}
            if numa.exists():
                numa_lines = {line.split()[0]:line for line in numa.read_text(errors='replace').splitlines() if line.split()}
            for region in regions:
                line = numa_lines.get(region['address'],'')
                region['numa'] = line or 'matching numa mapping unavailable'
            row['large_regions_kB'] = sorted(regions,key=lambda r:r.get('Rss',0),reverse=True)[:8]
        result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client', type=Path, nargs='+', default=[])
    parser.add_argument('--server', type=Path, nargs='+', default=[])
    parser.add_argument('--output', type=Path, default=Path('memory-summary.json'))
    args = parser.parse_args()
    if not args.client and not args.server:
        parser.error('Supply --client and/or --server directories')
    for root in args.client+args.server:
        if not root.is_dir():
            parser.error('Not a directory: '+str(root))
    result = dict(clients=[r for root in args.client for r in client_summary(root)],
                  servers=[r for root in args.server for r in server_summary(root)],
                  notes=['All bandwidths are useful payload GB/s; clock frequency is user supplied.',
                         'Selected smaps snapshot is not paired to every client sample.',
                         'AnonHugePages verifies CPU THP backing, not transport translation-page size.'])
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('stride  samples  kernel_us  kernel_GB/s  transfer_us  transfer_GB/s  file')
    for row in result['clients']:
        print(row.get('stride'),row.get('samples'),row.get('kernel_wall_us_p50'),row.get('kernel_wall_us_GBps'),
              row.get('transfer_window_us_p50'),row.get('transfer_window_us_GBps'),row['file'])
    print(f"Saved {args.output}: {len(result['clients'])} client cases, {len(result['servers'])} server cases")


if __name__ == '__main__':
    main()
