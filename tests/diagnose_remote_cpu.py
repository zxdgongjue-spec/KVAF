#!/usr/bin/env python3
"""Collect reproducible startup diagnostics, with no dependency on old versions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import traceback
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


class TerminalReport:
    """One budget for all parent output; children never inherit the terminal."""
    def __init__(self, limit=1000):
        self.limit = limit
        self.lines = 0

    def emit(self, text):
        for line in str(text).splitlines():
            if self.lines >= self.limit:
                return
            if self.lines == self.limit - 1:
                print('DIAG_TERMINAL_LIMIT reached; remaining details retained locally.', flush=True)
                self.lines += 1
                return
            # Strip terminal controls; bound unusually long loader/driver messages.
            line = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', line)
            line = ''.join(c if c.isprintable() or c == '\t' else '?' for c in line)
            print(line[:1600] + (' ...[line truncated]' if len(line) > 1600 else ''), flush=True)
            self.lines += 1

    def section(self, name, lines, budget):
        self.emit(f'--- {name} ---')
        lines = list(lines)
        if len(lines) > budget:
            front = budget // 2
            lines = lines[:front] + [f'... {len(lines) - budget + 1} lines omitted ...'] + lines[-(budget - front - 1):]
        for line in lines:
            self.emit(line)


def print_case_report(terminal, name, content):
    lines = content.splitlines()
    priority_markers = ('DEVICE_CAPTURE_V3', 'DEVICE_CQE', 'DEVICE_WQE', 'CQ_DECODE',
                        'WQE_CHECK', 'REQUEST_CHECK', 'CQ_AUX', 'DIRECT_AIV_WQE_EID_ENCODING', 'PEER_LINK_LOCAL_EID_READY')
    priority = [line for line in lines if any(marker in line for marker in priority_markers)]
    evidence = [line for line in lines if ('[KVAF_DIAG]' in line or line.startswith('DIAG_'))
                and not line.startswith('DIAG_LOADED_LIBRARY')
                and not any(marker in line for marker in priority_markers)]
    libraries = list(dict.fromkeys(line for line in lines if line.startswith('DIAG_LOADED_LIBRARY')))
    faults = [line for line in lines if any(marker in line for marker in
        ('ERROR', 'FAILED', 'FAILURE', 'PROBE_', 'EID_PAIR_', 'Traceback', 'Error:', 'Exception:'))]
    terminal.section(name + ' libraries', libraries, 12)
    terminal.section(name + ' device fault and driver auxiliary evidence', priority, 64)
    terminal.section(name + ' evidence', evidence, 46)
    terminal.section(name + ' errors and probe', faults, 45)
    # Preserve final Python exception and teardown even when a message is unrecognized.
    terminal.section(name + ' tail', lines[-18:], 18)


def kernel_delta(before, after):
    # Preserve duplicate events by subtracting counts, not by converting to a set.
    from collections import Counter
    previous = Counter(before.splitlines())
    delta = []
    for line in after.splitlines():
        if previous[line]:
            previous[line] -= 1
        elif ('Adding to iommu group' not in line and
              re.search(r'ummu|udma|urma|asdrv|smmu|iommu|fault|error|fail|token|\bub\b', line, re.I)):
            delta.append(line)
    return delta


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
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            # The child has its own session and does not receive terminal SIGINT.
            # Always stop that process group and return its evidence to main.
            interrupted = isinstance(exc, KeyboardInterrupt)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stream.write('\nDIAG_PROCESS_REAP_TIMEOUT cleanup_unconfirmed=1\n')
            marker = 'INTERRUPTED' if interrupted else 'TIMEOUT'
            stream.write(f'\nDIAG_PROCESS_{marker} cleanup_unconfirmed=1\n')
            return (130, False) if interrupted else (124, True)


def main(terminal=None):
    terminal = terminal or TerminalReport()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build_dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--peer-candidate', type=unsigned, default=0)
    parser.add_argument('--npu-candidate', type=unsigned, default=0)
    parser.add_argument('--timeout', type=unsigned, default=120, help='seconds per fresh process')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--single-only', action='store_true')
    selection.add_argument('--fault-only', action='store_true', help='one core, one jetty, original non_pin=1 only')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.timeout == 0:
        parser.error('--timeout must be positive')
    cases = [(1, 1, 1)] if args.fault_only else matrix(args.single_only)
    if args.dry_run:
        terminal.emit(json.dumps({'cases': cases, 'peer_candidate': args.peer_candidate,
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
    before = capture(['dmesg', '--ctime'])
    (output / 'dmesg-before.txt').write_text(before)
    terminal.emit('DIAG_REPORT_BEGIN max_terminal_lines=1000; paste this run output, no directory upload needed')
    terminal.section('manifest', json.dumps(metadata, default=str, indent=2).splitlines(), 45)
    terminal.section('environment', inventory.splitlines(), 55)
    results = []
    for cores, jettys, non_pin in cases:
        name = f'c{cores}-j{jettys}-nonpin{non_pin}'
        case_env = env | {'KVAF_DIAG_CORES': str(cores), 'KVAF_DIAG_JETTYS': str(jettys),
                          'KVAF_DIAG_PUT_NON_PIN': str(non_pin)}
        terminal.emit(f'DIAG_RUN case={name} timeout_seconds={args.timeout}; output appears at completion; Ctrl+C stops the child and prints captured evidence.')
        code, timed_out = run_case([sys.executable, str(ROOT / 'tests/python/diagnose_remote_cpu_case.py')],
                                  case_env, output / (name + '.log'), args.timeout)
        content = (output / (name + '.log')).read_text(errors='replace')
        instrumentation = '[KVAF_DIAG] FIXED_PAIR' in content
        cleanup = 'DIAG_CLEANUP_OK' in content and code != 130 and not timed_out
        registration = 'HBM_REGISTER implementation=batch_put_private_v2' in content
        device_capture = bool(re.search(r'DEVICE_CAPTURE_V3 .*kernel_marker=0x44494133', content))
        results.append({'case': name, 'exit': code, 'timeout': timed_out,
                        'instrumentation': instrumentation, 'cleanup': cleanup, 'private_put_registration': registration,
                        'device_capture_v3': device_capture})
        (output / 'summary.json').write_text(json.dumps(results, indent=2))
        terminal.emit(f'DIAG_RESULT {results[-1]}')
        print_case_report(terminal, name, content)
        if ('PEER_LINK_EID_PAIR_RMA_PROBE_BEGIN' in content and not registration):
            terminal.emit('DIAG_REGISTRATION_MISSING: rebuild NPU libraries; Put registration evidence is absent.')
            break
        if 'PUT_COMPLETION' in content and not device_capture:
            terminal.emit('DIAG_DEVICE_CAPTURE_MISSING: rebuild BOTH Put host library and AIV kernel; stop mixed-version run.')
            break
        if timed_out or not cleanup or not instrumentation:
            terminal.emit('DIAG_STOP: incomplete cleanup or missing rebuilt instrumentation; inspect evidence before rerunning.')
            break
    after = capture(['dmesg', '--ctime'])
    (output / 'dmesg-after.txt').write_text(after)
    delta = kernel_delta(before, after)
    terminal.section('new kernel fault messages', delta or ['No new matching messages; check dmesg access below.'], 60)
    terminal.section('dmesg access', after.splitlines()[:4], 4)
    terminal.emit('DIAG_SUMMARY ' + json.dumps(results))
    terminal.emit(f'DIAG_LOCAL_ARCHIVE {output} (optional; terminal evidence above is ready to paste)')
    terminal.emit('DIAG_REPORT_END')
    if any(r['exit'] == 130 for r in results):
        return 130
    return 0 if len(results) == len(cases) and all(r['exit'] == 0 and r['instrumentation'] and r['cleanup']
                                                    and r['private_put_registration'] for r in results) else 1


if __name__ == '__main__':
    terminal = TerminalReport()
    try:
        raise SystemExit(main(terminal))
    except Exception:
        terminal.emit(traceback.format_exc())
        raise SystemExit(1)
