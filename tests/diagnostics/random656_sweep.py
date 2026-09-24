#!/usr/bin/env python3
"""Random 656B concurrency sweep; each case has a fresh runtime/process."""
import argparse
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys


def plan():
    return [(7740, c, j) for c, j in itertools.product((8, 16, 30), (1, 2))] + [
        (16384, 30, 2), (32768, 30, 2)]


def payload(np, ids):
    columns = np.arange(656, dtype=np.uint32)
    return ((ids.astype(np.uint32)[:, None] >> ((columns % 4)*8)) ^
            (columns*17)).astype(np.uint8)


def summarize(root):
    rows = [json.loads(p.read_text()) for p in sorted(root.glob('case-*.json'))]
    groups = {}
    for r in rows:
        key = (r['requests'], r['cores'], r['jettys'])
        groups.setdefault(key, []).append(r)
    import statistics
    compact = []
    for key, runs in sorted(groups.items()):
        us = statistics.median(r['device_p50_us'] for r in runs)
        compact.append(dict(requests=key[0], cores=key[1], jettys=key[2], runs=len(runs),
                            p50_us=round(us, 3), GB_s=round(key[0]*656/us/1000, 3),
                            run_p50_us=[round(r['device_p50_us'], 3) for r in runs],
                            correctness=all(r['correctness'] for r in runs),
                            profile_max_core_ticks=[max((p['kernel_finish_cycle']-p['kernel_start_cycle']
                                for p in r.get('profile', []) if p['active_requests']), default=0) for r in runs],
                            profile_max_post_ticks=[max((p['post_done_cycle']-p['first_doorbell_cycle']
                                for p in r.get('profile', []) if p['first_doorbell_cycle']), default=0) for r in runs],
                            profile_max_wait_ticks=[max((p['completion_done_cycle']-p['post_done_cycle']
                                for p in r.get('profile', []) if p['active_requests']), default=0) for r in runs]))
    result = dict(metric='device_event_graph_replay_including_completion', token_bytes=656,
                  rows=compact, note='Not isolated transport or kernel-profiler bandwidth; THP/NUMA must be verified on server.')
    (root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


def case(args):
    # Import only after build/python is selected by the parent process.
    import numpy as np
    import torch
    import torch_npu  # noqa: F401
    import kvaf
    torch.npu.set_device(args.device)
    device = f'npu:{args.device}'
    total = args.blocks*128
    if args.requests > total:
        raise ValueError('requests exceed source tokens')
    runtime = kvaf.NpuRuntime(str(args.build/'bin/kvaf_execute_batch_get_kernels.o'),
        batch_get_token_bytes=656, batch_get_tokens_per_block=128,
        device_id=args.device, npu_core_count=args.cores, jettys_per_core=args.jettys,
        batch_put_kernel_object_path=str(args.build/'bin/kvaf_execute_batch_put_kernels.o'))
    runtime.link_to(args.server)
    runtime.initialize(total*656, 128*656)
    keys = list(range(1, args.blocks+1))
    # Each token has a source-identity-specific payload, so wrong remote addresses
    # cannot pass merely because all blocks contain the same pattern.
    for begin in range(0, args.blocks, 64):
        end = min(args.blocks, begin+64)
        data = torch.from_numpy(payload(np, np.arange(begin*128, end*128))).to(device)
        runtime.batch_put(keys[begin:end], data.reshape(end-begin, 128*656),
                          [i*128*656 for i in range(end-begin)], [128*656]*(end-begin))
        torch.npu.synchronize()
    table = runtime.prepare_access_table(keys)
    view = runtime.access_table_view(table, device='npu', capacity=args.blocks)
    stream = torch.npu.Stream(device=args.device)
    n = args.requests
    indices = torch.zeros(n, dtype=torch.int32, device=device)
    offsets = torch.zeros_like(indices)
    destinations = torch.arange(n, dtype=torch.int32, device=device)
    count = torch.tensor([n], dtype=torch.int32, device=device)
    output = torch.empty((n, 656), dtype=torch.uint8, device=device)
    torch.npu.synchronize()
    rng = np.random.default_rng(args.seed)

    def prepare():
        ids = rng.choice(total, n, replace=False)
        slots = rng.permutation(n).astype(np.int32)
        with torch.npu.stream(stream):
            indices.copy_(torch.from_numpy((ids//128).astype(np.int32)))
            offsets.copy_(torch.from_numpy(((ids%128)*656).astype(np.int32)))
            destinations.copy_(torch.from_numpy(slots))
            output.fill_(0xA5)
        stream.synchronize()
        return ids, slots

    def execute():
        # Same overload used by the existing Runtime performance test.
        with torch.npu.stream(stream):
            runtime.batch_get(view, indices, offsets, count, destinations, output)

    def verify(ids, slots):
        actual = output.cpu().numpy()
        if not np.array_equal(actual[slots], payload(np, ids)):
            raise AssertionError('Random source/destination payload mismatch')

    ids, slots = prepare()
    execute(); stream.synchronize(); verify(ids, slots)
    graph = kvaf.AclGraph()
    graph.capture_begin(stream)
    execute()
    graph.capture_end()
    samples = []
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    for i in range(args.warmup+args.iterations):
        ids, slots = prepare()
        start.record(stream)
        graph.replay(stream)
        end.record(stream)
        stream.synchronize()
        elapsed = start.elapsed_time(end)*1000
        verify(ids, slots)
        if i >= args.warmup:
            samples.append(elapsed)
    graph.reset()
    # Diagnostic launch is separate from every uninstrumented timing sample.
    runtime.start_batch_get_kernel_profile()
    ids, slots = prepare()
    execute(); stream.synchronize(); verify(ids, slots)
    timeline = runtime.stop_batch_get_kernel_profile()
    runtime.release_access_table(table)
    runtime.batch_free(keys)
    result = dict(requests=n, cores=args.cores, jettys=args.jettys, blocks=args.blocks,
                  source_bytes=total*656, payload_bytes=n*656, seed=args.seed,
                  device_p50_us=float(np.percentile(samples, 50)),
                  device_p95_us=float(np.percentile(samples, 95)), samples_us=samples,
                  profile=timeline, correctness=True, timing='device_event_graph_replay_including_completion')
    args.result.write_text(json.dumps(result, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=Path('build-npu'))
    p.add_argument('--server', default=os.environ.get('KVAF_CPU_SERVICE_ADDR'))
    p.add_argument('--device', type=int, default=int(os.environ.get('DEVICE_ID', '0')))
    p.add_argument('--output', type=Path, default=Path('random656-results'))
    p.add_argument('--blocks', type=int, default=1024)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--seed', type=int, default=20260924)
    p.add_argument('--summarize-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--requests', type=int, default=7740, help=argparse.SUPPRESS)
    p.add_argument('--cores', type=int, default=30, help=argparse.SUPPRESS)
    p.add_argument('--jettys', type=int, default=2, help=argparse.SUPPRESS)
    p.add_argument('--result', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.summarize_only:
        summarize(args.output); return
    if args.blocks < 256 or args.iterations < 1 or args.warmup < 0 or args.repeats < 1:
        p.error('requires blocks>=256, iterations/repeats>=1, warmup>=0')
    args.build = args.build.resolve()
    if args.result:
        case(args); return
    jobs = [(repeat, *config) for repeat in range(args.repeats) for config in plan()]
    random.Random(args.seed).shuffle(jobs)
    if args.dry_run:
        print(json.dumps(jobs)); return
    if not args.server:
        p.error('--server or KVAF_CPU_SERVICE_ADDR is required')
    for relative in ('bin/kvaf_execute_batch_get_kernels.o', 'bin/kvaf_execute_batch_put_kernels.o', 'python/kvaf'):
        if not (args.build/relative).exists():
            p.error(f'missing {args.build/relative}; use the V9-compatible KVAF build')
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob('case-*')):
        p.error('output contains earlier cases; choose a new directory')
    (args.output/'plan.json').write_text(json.dumps(dict(jobs=jobs, server=args.server,
        device=args.device, blocks=args.blocks, iterations=args.iterations, seed=args.seed), indent=2))
    env = os.environ.copy()
    env['PYTHONPATH'] = str(args.build/'python')+os.pathsep+env.get('PYTHONPATH', '')
    for index, (repeat, requests, cores, jettys) in enumerate(jobs):
        stem = args.output/f'case-{index:02d}-r{repeat}-n{requests}-c{cores}-j{jettys}'
        command = [sys.executable, str(Path(__file__).resolve()), '--build', str(args.build),
            '--server', args.server, '--device', str(args.device), '--blocks', str(args.blocks),
            '--warmup', str(args.warmup), '--iterations', str(args.iterations),
            '--seed', str(args.seed+repeat), '--requests', str(requests), '--cores', str(cores),
            '--jettys', str(jettys), '--result', str(stem.with_suffix('.json'))]
        print(f'[{index+1}/{len(jobs)}] requests={requests} cores={cores} jettys={jettys}', flush=True)
        with stem.with_suffix('.log').open('w') as log:
            code = subprocess.call(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if code:
            raise SystemExit(f'case failed ({code}); stopped, inspect {stem.with_suffix(".log")}')
    summarize(args.output)


if __name__ == '__main__':
    main()
