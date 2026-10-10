#!/usr/bin/env python3
"""Collect reproducible startup diagnostics, with no dependency on old versions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def unsigned(value):
    if not value.isascii() or not value.isdecimal() or int(value) > 0xffffffff:
        raise argparse.ArgumentTypeError('expected an unsigned 32-bit decimal integer')
    return int(value)


def matrix(single_only=False):
    layouts = [(1, 1)] if single_only else [(1, 1), (32, 2)]
    return [(cores, jettys, non_pin) for cores, jettys in layouts for non_pin in (1, 0)]


def capture(command, timeout=15):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors='replace', timeout=timeout)
        return f'$ {command!r}\nexit={result.returncode}\n{result.stdout}'
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f'$ {command!r}\nunavailable: {exc}\n'


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return {'path': str(path.resolve()), 'sha256': digest.hexdigest()}


def run_case(command, env, log, timeout):
    with log.open('w') as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            # Bound even a stuck ACL/RA teardown. Do not start more cases afterwards.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            stream.write('\nDIAG_PROCESS_TIMEOUT cleanup_unconfirmed=1\n')
            return 124, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build_dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--peer-candidate', type=unsigned, default=0)
    parser.add_argument('--npu-candidate', type=unsigned, default=0)
    parser.add_argument('--timeout', type=unsigned, default=120, help='seconds per fresh process')
    parser.add_argument('--single-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.timeout == 0:
        parser.error('--timeout must be positive')
    cases = matrix(args.single_only)
    if args.dry_run:
        print(json.dumps({'cases': cases, 'peer_candidate': args.peer_candidate,
                          'npu_candidate': args.npu_candidate, 'timeout': args.timeout}))
        return 0
    if not os.environ.get('KVAF_CPU_SERVICE_ADDR'):
        parser.error('set KVAF_CPU_SERVICE_ADDR to the running CPU service')
    build = args.build_dir.resolve()
    get_kernel = build / 'bin/kvaf_execute_batch_get_kernels.o'
    put_kernel = build / 'bin/kvaf_execute_batch_put_kernels.o'
    bindings = list((build / 'python/kvaf').glob('_kvaf*.so'))
    if not get_kernel.is_file() or not put_kernel.is_file() or not bindings:
        parser.error('build_dir must contain rebuilt NPU kernels and Python binding')
    output = (args.output or Path('kvaf-diag-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(KVAF_DIAGNOSTICS='1', KVAF_AIV_VERBOSE_DIRECT_HCCP='1', KVAF_DIAG_PEER_CANDIDATE=str(args.peer_candidate),
               KVAF_DIAG_NPU_CANDIDATE=str(args.npu_candidate), PYTHONUNBUFFERED='1',
               KVAF_BATCH_GET_KERNEL=str(get_kernel), KVAF_BATCH_PUT_KERNEL=str(put_kernel),
               PYTHONPATH=str(build / 'python') + os.pathsep + env.get('PYTHONPATH', ''))
    metadata = {'utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
                'python': sys.version, 'executable': sys.executable, 'arguments': vars(args) | {'build_dir': str(build), 'output': str(output)},
                'environment': {k: env.get(k) for k in ('KVAF_CPU_SERVICE_ADDR', 'DEVICE_ID',
                    'ASCEND_CANN_PACKAGE_PATH', 'LD_LIBRARY_PATH', 'ASCEND_RT_VISIBLE_DEVICES')},
                'artifacts': [fingerprint(p) for p in [get_kernel, put_kernel, *bindings,
                    *sorted(build.rglob('libkvaf*.so'))]]}
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2, default=str))
    inventory = capture(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'])
    inventory += capture(['git', '-C', str(ROOT), 'status', '--short'])
    inventory += capture(['npu-smi', 'info']) + capture(['uname', '-a'])
    for path in ('/usr/local/Ascend/driver/version.info', '/proc/asdrv_ub/pair_info'):
        try:
            inventory += f'\nFILE {path}\n' + Path(path).read_text(errors='replace')
        except OSError as exc:
            inventory += f'\nFILE {path} unavailable: {exc}\n'
    (output / 'environment.txt').write_text(inventory)
    (output / 'dmesg-before.txt').write_text(capture(['dmesg', '--ctime']))
    results = []
    for cores, jettys, non_pin in cases:
        name = f'c{cores}-j{jettys}-nonpin{non_pin}'
        case_env = env | {'KVAF_DIAG_CORES': str(cores), 'KVAF_DIAG_JETTYS': str(jettys),
                          'KVAF_DIAG_PUT_NON_PIN': str(non_pin)}
        print(f'DIAG_RUN case={name} log={output / (name + ".log")}', flush=True)
        code, timed_out = run_case([sys.executable, str(ROOT / 'tests/python/diagnose_remote_cpu_case.py')],
                                  case_env, output / (name + '.log'), args.timeout)
        content = (output / (name + '.log')).read_text(errors='replace')
        instrumentation = '[KVAF_DIAG] FIXED_PAIR' in content
        cleanup = 'DIAG_CLEANUP_OK' in content
        results.append({'case': name, 'exit': code, 'timeout': timed_out,
                        'instrumentation': instrumentation, 'cleanup': cleanup})
        (output / 'summary.json').write_text(json.dumps(results, indent=2))
        print(f'DIAG_RESULT {results[-1]}', flush=True)
        if timed_out or not cleanup or not instrumentation:
            print('DIAG_STOP: incomplete cleanup or missing rebuilt instrumentation; inspect logs before rerunning.', flush=True)
            break
    (output / 'dmesg-after.txt').write_text(capture(['dmesg', '--ctime']))
    print(f'DIAG_OUTPUT {output}', flush=True)
    return 0 if len(results) == len(cases) and all(r['exit'] == 0 and r['instrumentation'] and r['cleanup'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
