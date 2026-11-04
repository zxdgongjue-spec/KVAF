#!/usr/bin/env python3
"""Instrument the existing perf test without modifying its requests or transport."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def instrument(source):
    replacements = [
        ('        for _ in range(args.iterations):\n',
         '        scope_events = []\n        for _ in range(args.iterations):\n'),
        ('            begin = time.perf_counter()\n',
         '            scope_start = torch.npu.Event(enable_timing=True)\n'
         '            scope_end = torch.npu.Event(enable_timing=True)\n'
         '            begin = time.perf_counter()\n'
         '            scope_start.record()\n'),
        ('            submitted = time.perf_counter()\n',
         '            submitted = time.perf_counter()\n            scope_end.record()\n'),
        ('            completed = time.perf_counter()\n',
         '            completed = time.perf_counter()\n'
         '            scope_events.append(float(scope_start.elapsed_time(scope_end))*1000)\n'),
        ('        print_samples(\n            "batch_get_submit",',
         '        scope_save(args, batch_get_samples, submit_samples, scope_events, kernel_profile_samples)\n'
         '        print_samples(\n            "batch_get_submit",'),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError('Unsupported upstream benchmark layout: ' + repr(old))
        source = source.replace(old, new)
    return source


def summarize_sample(profiles, clock, payload, host_us, submit_us, event_us):
    wall_ticks = max(p['kernel_finish_cycle'] for p in profiles) - min(p['kernel_start_cycle'] for p in profiles)
    first = [p['first_doorbell_cycle'] for p in profiles if p['first_doorbell_cycle']]
    transfer_ticks = max(p['completion_done_cycle'] for p in profiles) - min(first) if first else 0
    kernel_us = wall_ticks / clock
    transfer_us = transfer_ticks / clock
    return dict(host_us=host_us, host_submit_us=submit_us, event_us=event_us,
                kernel_wall_ticks=wall_ticks, kernel_wall_us=kernel_us,
                transfer_window_ticks=transfer_ticks, transfer_window_us=transfer_us,
                host_outside_kernel_us=host_us-kernel_us,
                event_outside_kernel_us=event_us-kernel_us,
                GBps={name: payload/us/1000 if us > 0 else None for name, us in
                      [('host', host_us), ('event', event_us), ('kernel', kernel_us), ('transfer_window', transfer_us)]},
                profiles=profiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clock-mhz', type=int, required=True, help='Explicit assumed GetSystemCycle frequency; verify on hardware')
    args, extra = parser.parse_known_args()
    if args.clock_mhz <= 0:
        parser.error('clock-mhz must be positive')
    root = Path(__file__).resolve().parents[2]
    build = args.build.resolve()
    source_path = root/'tests/python/test_npu_batch_get_perf.py'
    source = source_path.read_text()
    patched = instrument(source)
    for name, value in [('KVAF_RUN_NPU_TESTS', '1'), ('KVAF_BATCH_GET_KERNEL', str(build/'bin/kvaf_execute_batch_get_kernels.o')),
                        ('KVAF_BATCH_PUT_KERNEL', str(build/'bin/kvaf_execute_batch_put_kernels.o'))]:
        os.environ[name] = value
    sys.path[:0] = [str(build/'python'), str(source_path.parent)]
    sys.argv = [str(source_path), *extra, '--execution-mode', 'eager', '--kernel-profile', '--kernel-profile-clock-mhz', str(args.clock_mhz)]

    def save(config, host, submit, events, profiles):
        if not (len(host) == len(submit) == len(events) == len(profiles)):
            raise RuntimeError('Paired sample count mismatch')
        payload = config.miss_count * config.token_bytes
        result = dict(payload_bytes=payload, assumed_clock_mhz=args.clock_mhz,
                      config=vars(config), benchmark_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      notes=['Events bracket host submission and may contain dispatch gaps.',
                             'Host timing includes event instrumentation overhead.',
                             'Cross-core wall timestamps assume a common device clock.',
                             'Transfer window overlaps WQE posting; do not add the two.'],
                      samples=[summarize_sample(p,args.clock_mhz,payload,h,s,e)
                               for h,s,e,p in zip(host,submit,events,profiles)])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str))
        print('BANDWIDTH_SCOPES_JSON path=' + str(args.output), flush=True)

    exec(compile(patched, str(source_path), 'exec'),
         {'__name__': '__main__', '__file__': str(source_path), 'scope_save': save})


if __name__ == '__main__':
    main()
