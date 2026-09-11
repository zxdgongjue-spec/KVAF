#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
  cat <<'EOF'
Usage: tests/run_cpu_urma_eid_perf_pinned.sh BUILD_DIR [EID test options]

Run the EID bandwidth sweep with its four OpenMP workers pinned to four
different physical cores allowed by the caller's CPU affinity.
Prefer cores in the same NUMA node. Use taskset to restrict CPU selection.
All remaining arguments are passed to kvaf_cpu_urma_eid_perf unchanged.
EOF
  exit 0
fi

kvaf_build_dir="$(cd "$1" && pwd)"
shift
kvaf_benchmark="$kvaf_build_dir/tests/kvaf_cpu_urma_eid_perf"
if [[ ! -x "$kvaf_benchmark" ]]; then
  echo "ERROR: missing executable: $kvaf_benchmark; run ./build.sh cpu first" >&2
  exit 1
fi

kvaf_python="${PYTHON:-python3}"
kvaf_places="$("$kvaf_python" - <<'PY'
import os
import sys
from pathlib import Path

if not hasattr(os, "sched_getaffinity"):
    sys.exit("ERROR: this affinity test requires Linux")

cores = []
seen = set()
for cpu in sorted(os.sched_getaffinity(0)):
    root = Path(f"/sys/devices/system/cpu/cpu{cpu}")
    try:
        siblings = (root / "topology/thread_siblings_list").read_text().strip()
    except OSError as error:
        sys.exit(f"ERROR: cannot read CPU {cpu} topology: {error}")
    if not siblings:
        sys.exit(f"ERROR: empty CPU {cpu} topology")
    if siblings in seen:
        continue
    seen.add(siblings)
    nodes = sorted(root.glob("node[0-9]*"))
    node = nodes[0].name if nodes else "unknown"
    cores.append((cpu, node))

if len(cores) < 4:
    sys.exit("ERROR: allow at least four distinct physical cores using taskset/cpuset")

selected = cores[:4]
for first in cores:
    same_node = [c for c in cores if c[1] == first[1]]
    if len(same_node) >= 4:
        selected = same_node[:4]
        break

for jetty, (cpu, node) in enumerate(selected):
    print(f"EID_PERF_AFFINITY jetty={jetty} target_cpu={cpu} numa={node}", file=sys.stderr)
print(",".join("{" + str(cpu) + "}" for cpu, _ in selected))
PY
)"

# Avoid inherited vendor affinity settings overriding the explicit places.
unset GOMP_CPU_AFFINITY KMP_AFFINITY KMP_HW_SUBSET
export OMP_NUM_THREADS=4 OMP_THREAD_LIMIT=4 OMP_DYNAMIC=FALSE
export OMP_PLACES="$kvaf_places" OMP_PROC_BIND=spread OMP_DISPLAY_AFFINITY=TRUE
exec "$kvaf_benchmark" "$@"
