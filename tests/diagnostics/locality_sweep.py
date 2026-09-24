#!/usr/bin/env python3
"""Read-only transport diagnosis using supported Runtime arguments."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys


def plan(suite):
    cases = []
    if suite in ('locality', 'all'):
        # Fixed registration size and request count. Only the sampled range changes.
        for blocks in (192, 1024, 8192):
            for pattern in ('random', 'contiguous'):
                cases.append(dict(label=f'{pattern}-{blocks*128*656//1048576}MiB',
                    token_bytes=656, requests=7740, blocks=8192,
                    working_tokens=blocks*128, pattern=pattern))
        cases.append(dict(label='sorted-82MiB', token_bytes=656, requests=7740,
            blocks=8192, working_tokens=1024*128, pattern='sorted'))
    if suite in ('size', 'all'):
        for size in (256, 656, 1024, 4096):
            blocks = math.ceil(86_000_000/(128*size))
            cases.append(dict(label=f'size-{size}', token_bytes=size,
                requests=round(5_077_440/size), blocks=blocks,
                working_tokens=blocks*128, pattern='random'))
    return cases


def summarize(root):
    groups = {}
    for p in sorted(root.glob('case-*.json')):
        r = json.loads(p.read_text()); groups.setdefault(r['label'], []).append(r)
    rows = []
    for label, runs in sorted(groups.items()):
        r = runs[0]; us = statistics.median(x['device_p50_us'] for x in runs)
        rows.append(dict(label=label, runs=len(runs), token_bytes=r['token_bytes'],
            requests=r['requests'], working_bytes=r['working_bytes'], registered_bytes=r['source_bytes'],
            p50_us=round(us,3), GB_s=round(r['payload_bytes']/us/1000,3),
            Mrequests_s=round(r['requests']/us,3),
            run_p50_us=[round(x['device_p50_us'],3) for x in runs],
            correctness=all(x['correctness'] for x in runs),
            address_stats=[x.get('address_stats') for x in runs],
            max_post_ticks=[max((p['post_done_cycle']-p['first_doorbell_cycle'] for p in x['profile']
                if p['first_doorbell_cycle']), default=0) for x in runs],
            max_wait_ticks=[max((p['completion_done_cycle']-p['post_done_cycle'] for p in x['profile']
                if p['active_requests']), default=0) for x in runs]))
    result = dict(metric='device_event_graph_replay_including_completion', rows=rows,
        limitations=['Locality does not uniquely separate memory latency from address translation.',
          'Larger tokens are diagnostic, not a production 656B bandwidth claim.',
          'No ordering bits changed; ordering causality requires documented fence semantics.'])
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,default=Path('build-npu'))
    p.add_argument('--server',default=os.environ.get('KVAF_CPU_SERVICE_ADDR'))
    p.add_argument('--device',type=int,default=int(os.environ.get('DEVICE_ID','0')))
    p.add_argument('--suite',choices=('locality','size','all'),default='locality')
    p.add_argument('--output',type=Path,default=Path('locality-results'))
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--iterations',type=int,default=20)
    p.add_argument('--summarize-only',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.summarize_only: summarize(a.output);return
    if a.repeats<1 or a.iterations<1:p.error('repeats and iterations must be positive')
    jobs=[(repeat,c) for repeat in range(a.repeats) for c in plan(a.suite)]
    random.Random(20260924).shuffle(jobs)
    if a.dry_run:print(json.dumps(jobs,indent=2));return
    if not a.server:p.error('--server or KVAF_CPU_SERVICE_ADDR required')
    a.build=a.build.resolve()
    for f in ('python/kvaf','bin/kvaf_execute_batch_get_kernels.o','bin/kvaf_execute_batch_put_kernels.o'):
        if not (a.build/f).exists():p.error(f'missing {a.build/f}')
    a.output.mkdir(parents=True,exist_ok=True)
    if list(a.output.glob('case-*')):p.error('use a new output directory')
    (a.output/'plan.json').write_text(json.dumps(jobs,indent=2))
    env=os.environ.copy();env['PYTHONPATH']=str(a.build/'python')+os.pathsep+env.get('PYTHONPATH','')
    for i,(repeat,c) in enumerate(jobs):
        stem=a.output/f'case-{i:02d}-{c["label"]}-r{repeat}'
        command=[sys.executable,str(Path(__file__).with_name('random656_sweep.py')),
            '--build',str(a.build),'--server',a.server,'--device',str(a.device),
            '--cores','30','--jettys','2','--iterations',str(a.iterations),
            '--seed',str(20260924+repeat),'--result',str(stem.with_suffix('.json'))]
        for key in ('token_bytes','requests','blocks','working_tokens','pattern'):
            command += ['--'+key.replace('_','-'),str(c[key])]
        print(f'[{i+1}/{len(jobs)}] {c["label"]}',flush=True)
        with stem.with_suffix('.log').open('w') as log:
            status=subprocess.call(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        if status:raise SystemExit(f'failed ({status}), stopped: {stem.with_suffix(".log")}')
        file=stem.with_suffix('.json');r=json.loads(file.read_text());r['label']=c['label'];file.write_text(json.dumps(r,indent=2))
    summarize(a.output)


if __name__=='__main__':main()
