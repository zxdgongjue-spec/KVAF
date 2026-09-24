#!/usr/bin/env python3
"""Bounded read-only CPU-service memory and device counter sampling."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time


def command(argv):
    try:
        r=subprocess.run(argv,capture_output=True,text=True,timeout=15)
        return dict(code=r.returncode,stdout=r.stdout,stderr=r.stderr)
    except (OSError,subprocess.TimeoutExpired) as e:return dict(unavailable=str(e))


def memory(pid):
    result={}
    for name in ('smaps_rollup','numa_maps'):
        try:
            text=(Path('/proc')/str(pid)/name).read_text()
            if name=='smaps_rollup':
                result[name]={k:int(v) for k,v in re.findall(r'^(\w+):\s+(\d+) kB',text,re.M)}
            else:
                nodes={}
                for node,pages in re.findall(r'\bN(\d+)=(\d+)',text):
                    nodes[node]=nodes.get(node,0)+int(pages)
                result['numa_node_page_counts']=nodes
        except OSError as e:result[name]=dict(unavailable=str(e))
    return result


def counters(interface):
    data={}
    for path in Path('/sys/class/infiniband').glob('*/ports/*/*counters/*'):
        try:data[str(path)]=int(path.read_text().strip(),0)
        except (ValueError,OSError):pass
    if interface:
        for path in (Path('/sys/class/net')/interface/'statistics').glob('*'):
            try:data[str(path)]=int(path.read_text())
            except (ValueError,OSError):pass
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid',type=int,required=True,help='running CPU service PID')
    p.add_argument('--interface',help='optional relevant network interface')
    p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--output',type=Path,default=Path('path-counters.json'))
    a=p.parse_args()
    if a.seconds<1 or not (Path('/proc')/str(a.pid)).exists():p.error('invalid duration or service PID')
    start=counters(a.interface)
    report=dict(pid=a.pid,started_unix=time.time(),memory_samples=[],counter_start=start,
        rdma_start=command(['rdma','-j','statistic','show']),numa_hardware=command(['numactl','--hardware']),
        iommu_groups=len(list(Path('/sys/kernel/iommu_groups').glob('*'))),
        limitations=['Counter deltas are system-wide over the entire collection, not per case.',
          'CPU process counters do not measure device DMA memory traffic or NIC translation misses.',
          'Unavailable vendor PMU counters are not zero; this collector cannot prove an ordering bottleneck.'])
    if a.interface:report['ethtool_start']=command(['ethtool','-S',a.interface])
    def save():a.output.write_text(json.dumps(report,indent=2)+'\n')
    stop=time.monotonic()+a.seconds
    try:
        while True:
            report['memory_samples'].append(dict(time=time.time(),**memory(a.pid)))
            save()
            remaining=stop-time.monotonic()
            if remaining<=0:break
            time.sleep(min(5,remaining))
    except KeyboardInterrupt:pass
    end=counters(a.interface)
    report['counter_delta']={k:end[k]-v for k,v in start.items() if k in end}
    report['rdma_end']=command(['rdma','-j','statistic','show'])
    if a.interface:report['ethtool_end']=command(['ethtool','-S',a.interface])
    report['finished_unix']=time.time();save()
    print(a.output)


if __name__=='__main__':main()
