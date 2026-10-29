import contextlib
import csv
import io
import os
import json
from pathlib import Path
import tempfile
import sys
import time
import unittest
from unittest.mock import patch

import peer_eid_profile as report
import run_peer_eid_sweep as sweep
import test_npu_batch_get_perf as perf

EIDS = ['1' * 32, '2' * 32]


def profile_log():
    # Deliberately rounded-to-zero ms: summaries must use the precise cycles.
    return '\n'.join(
        f'PERF_RESULT operation=batch_get_kernel_{stage} mean_ms=0.000 p50_ms=0.000 '
        f'p90_ms=0.000 p99_ms=0.000 mean_cycles={base + 10} p50_cycles={base} '
        f'p90_cycles={base + 20} p99_cycles={base + 30}'
        for stage, base in [('total', 400), ('post', 100), ('completion_wait', 200)])


def sample(serial=False):
    return [{"jettys": [dict(eid_path=i % 2, requests=128, completion_count=1,
                             first_doorbell_cycle=10 + (40 if serial and i % 2 else i),
                             last_doorbell_cycle=10 + (40 if serial and i % 2 else i),
                             first_completion_cycle=30 + (40 if serial and i % 2 else i),
                             last_completion_cycle=30 + (40 if serial and i % 2 else i))
                         for i in range(4)]}]


class PeerEidTests(unittest.TestCase):
    def args(self, *extra):
        with patch.dict(os.environ, {}, clear=True):
            return sweep.parse_args(['build-npu', '--peer-eids', ','.join(EIDS),
                                     '--npu-eid', '3' * 32, '--output-dir', 'logs', *extra])

    def test_command_runs_both_eids_in_one_benchmark(self):
        args = self.args('--cores', '1', '--jettys', '4')
        parsed = perf.parse_args(sweep.benchmark_args(args))
        self.assertEqual(parsed.peer_eids, EIDS)
        self.assertEqual(parsed.npu_eids, ['3' * 32])
        self.assertEqual(parsed.aiv_cores, 1)
        self.assertEqual(parsed.jettys_per_core, 4)
        self.assertTrue(parsed.kernel_profile)
        self.assertEqual(parsed.max_inflight_total, 2048)
        self.assertIsNone(perf.validate_args(parsed))

    def test_default_budget_scales_to_activate_all_routes(self):
        args = self.args('--cores', '4', '--jettys', '4')
        self.assertEqual(args.max_inflight_total, 4096)
        self.assertIsNone(perf.validate_args(perf.parse_args(sweep.benchmark_args(args))))

    def test_miss_count_and_clock_are_forwarded(self):
        args = self.args('--miss-count', '2048', '--kernel-profile-clock-mhz', '800')
        parsed = perf.parse_args(sweep.benchmark_args(args))
        self.assertEqual(parsed.request_length, 32768)
        self.assertEqual(parsed.miss_count, 2048)
        self.assertEqual(parsed.kernel_profile_clock_mhz, 800)
        with patch.dict(os.environ, {'KVAF_BATCH_GET_KERNEL_CLOCK_MHZ': '900'}):
            parsed = sweep.parse_args(['build-npu', '--peer-eids', ','.join(EIDS),
                                      '--npu-eid', '3' * 32, '--output-dir', 'logs'])
            self.assertEqual(parsed.kernel_profile_clock_mhz, 900)

    def test_rejects_serializing_or_idle_shapes(self):
        for extra in [('--request-length', '128'), ('--max-inflight-total', '2'),
                      ('--cores', '0'), ('--iterations', '0'), ('--jettys', '1'),
                      ('--peer-eids', EIDS[0]), ('--peer-eids', ','.join([EIDS[0]] * 2))]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.args(*extra)

    def test_profile_detects_overlap_and_aggregates_both_paths(self):
        rows = report.summarize([sample()], EIDS, 1, 4, 656)
        self.assertEqual(len(rows), 2)
        self.assertEqual([r['requests'] for r in rows], [256, 256])
        self.assertEqual([r['bytes'] for r in rows], [256 * 656] * 2)
        self.assertTrue(all(r['first_wave_observed_overlap_cycles'] > 0 for r in rows))

    def test_serial_evidence_fails_but_preserves_csv(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            path = Path(folder) / 'samples.csv'
            with self.assertRaisesRegex(ValueError, 'not observed'):
                report.report([sample(serial=True)], EIDS, 1, 4, 656, path)
            with path.open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_idle_wrong_route_and_missing_profiles_fail(self):
        for key, value in [('requests', 0), ('completion_count', 0), ('eid_path', 1)]:
            profiles = sample()
            profiles[0]['jettys'][0][key] = value
            with self.assertRaises(ValueError):
                report.summarize([profiles], EIDS, 1, 4, 656)
        with self.assertRaises(ValueError):
            report.summarize([], EIDS, 1, 4, 656)

    def test_core_clocks_are_never_compared(self):
        profiles = sample() + sample()
        for lane in profiles[1]['jettys']:
            for key in ('first_doorbell_cycle', 'last_doorbell_cycle',
                        'first_completion_cycle', 'last_completion_cycle'):
                lane[key] += 100000
        rows = report.summarize([profiles], EIDS, 2, 4, 656)
        self.assertEqual(rows[0]['first_wave_observed_overlap_cycles'],
                         rows[2]['first_wave_observed_overlap_cycles'])

    def test_launcher_one_child_environment_logs_and_failure_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'bin').mkdir()
            (root / 'python/kvaf').mkdir(parents=True)
            for name in ('get', 'put'):
                (root / f'bin/kvaf_execute_batch_{name}_kernels.o').touch()
            output = root / 'logs'
            args = [str(root), '--peer-eids', ','.join(EIDS), '--npu-eid', '3' * 32,
                    '--output-dir', str(output)]

            def child(command, env, stream, timeout):
                parsed = perf.parse_args(command[3:])
                self.assertEqual(parsed.peer_eids, EIDS)
                self.assertEqual(env['KVAF_RUN_NPU_TESTS'], '1')
                self.assertEqual(env['DEVICE_ID'], '7')
                self.assertTrue(env['PYTHONPATH'].startswith(str(root / 'python')))
                stream.write('native and Python log\n')
                return 9

            with patch.dict(os.environ, {'KVAF_CPU_SERVICE_ADDR': '127.0.0.1:29000',
                                         'DEVICE_ID': '7'}, clear=True), \
                 patch.object(sweep, 'run_logged', side_effect=child) as run, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(sweep.main(args), 9)
                run.assert_called_once()
                text = (output / 'batch_get.log').read_text()
                self.assertIn('native and Python log\n', text)
                self.assertIn('BATCH_GET_MISS_SWEEP_FAILED', text)
                self.assertNotIn('BATCH_GET_MISS_SWEEP_SUMMARY', text)
                self.assertFalse((output / 'batch_get_miss_sweep_summary.csv').exists())
                self.assertEqual(json.loads((output / 'config.json').read_text())['execution'], 'single_kernel')
                self.assertEqual(sweep.main(args), 2)  # preserve old results
                run.assert_called_once()

    def test_summary_matches_original_csv_and_cycle_conversion(self):
        args = self.args('--miss-count', '2048', '--kernel-profile-clock-mhz', '800')
        lines = []
        with tempfile.TemporaryDirectory() as folder:
            sweep.write_summary(args, profile_log(), Path(folder), lines.append)
            with (Path(folder) / 'batch_get_miss_sweep_summary.csv').open() as stream:
                reader = csv.DictReader(stream)
                row = next(reader)
                reference = (Path(__file__).parents[1] / 'run_remote_cpu_batch_get_miss_count_perf.sh').read_text()
                self.assertIn("'" + ','.join(reader.fieldnames) + "'", reference)
            self.assertEqual(row['p50_us'], '0.500')
            self.assertEqual(row['post_p50_us'], '0.125')
            self.assertEqual(row['completion_wait_p50_us'], '0.250')
            self.assertEqual(row['p50_GBps'], '2686.976')  # both peers' TOTAL bytes, decimal GB/s
            self.assertEqual(row['miss_count'], '2048')
            self.assertEqual(row['selected_ub_device'], '')
            for stage in ('kernel_total', 'kernel_post', 'kernel_completion_wait'):
                self.assertIn(f'BATCH_GET_MISS_SWEEP_STAGE_SUMMARY stage={stage}', lines)
            self.assertIn('p50_GB/s', '\n'.join(lines))

    def test_successful_launcher_shows_and_saves_complete_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'bin').mkdir()
            (root / 'python/kvaf').mkdir(parents=True)
            for name in ('get', 'put'):
                (root / f'bin/kvaf_execute_batch_{name}_kernels.o').touch()
            output = root / 'logs'
            args = [str(root), '--peer-eids', ','.join(EIDS), '--npu-eid', '3' * 32,
                    '--output-dir', str(output), '--miss-count', '2048']

            def child(command, env, stream, timeout):
                row = dict.fromkeys(perf.jetty_profile.FIELDS, '0')
                perf.jetty_profile.write_csv(output / 'jettys.csv', [row])
                text = ('BATCH_GET_PERF_PHASE_BEGIN phase=initialize\n' + profile_log() +
                        '\nBATCH_GET_JETTY_CSV_WRITTEN path=jettys.csv rows=1\n'
                        'PEER_EID_CONCURRENCY samples=100 execution=single_kernel\n')
                stream.write(text)
                stream.flush()
                print(text, end='')
                return 0

            console = io.StringIO()
            with patch.dict(os.environ, {'KVAF_CPU_SERVICE_ADDR': 'localhost:29000'}, clear=True), \
                 patch.object(sweep, 'run_logged', side_effect=child), contextlib.redirect_stdout(console):
                self.assertEqual(sweep.main(args), 0)
            text = console.getvalue()
            self.assertEqual(text, (output / 'batch_get.log').read_text())
            self.assertEqual(text.count('BATCH_GET_MISS_SWEEP_STAGE_SUMMARY'), 3)
            self.assertEqual(text.count('PEER_EID_CONCURRENCY'), 1)
            self.assertIn('BATCH_GET_JETTY_SWEEP_CSV_WRITTEN', text)
            self.assertTrue((output / 'batch_get_miss_sweep_summary.csv').is_file())
            self.assertEqual((output / 'batch_get_jetty_summary.csv').read_text(),
                             (output / 'jettys.csv').read_text())

    def test_missing_or_invalid_summary_does_not_create_csv(self):
        for text in ('', profile_log().replace('p50_cycles=400', 'p50_cycles=nan'),
                     profile_log().replace('p50_cycles=400', 'p50_cycles=0')):
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    sweep.write_summary(self.args(), text, Path(folder), lambda line: None)
                self.assertFalse((Path(folder) / 'batch_get_miss_sweep_summary.csv').exists())

    def test_tee_outputs_before_child_exits_and_keeps_stderr(self):
        with tempfile.TemporaryDirectory() as folder:
            ack = Path(folder) / 'ack'
            log_path = Path(folder) / 'run.log'

            class LiveOutput(io.StringIO):
                def write(self, text):
                    count = super().write(text)
                    if 'READY' in text:
                        # The child cannot finish until it sees this acknowledgement.
                        if 'READY' in log_path.read_text():
                            ack.touch()
                    return count

            source = ('import pathlib, sys, time\n'
                      'print("READY", flush=True)\n'
                      'while not pathlib.Path(sys.argv[1]).exists(): time.sleep(0.01)\n'
                      'print("CHILD_STDERR", file=sys.stderr, flush=True)\n'
                      'print("DONE", flush=True)\n')
            console = LiveOutput()
            with log_path.open('w') as log, contextlib.redirect_stdout(console):
                code = sweep.run_logged([sys.executable, '-u', '-c', source, str(ack)],
                                        os.environ.copy(), log, 10)
            self.assertEqual(code, 0)
            self.assertEqual(console.getvalue(), log_path.read_text())
            self.assertIn('CHILD_STDERR', console.getvalue())
            self.assertIn('DONE', console.getvalue())

    def test_silent_child_times_out_and_nonzero_exit_is_preserved(self):
        with contextlib.redirect_stdout(io.StringIO()):
            begin = time.monotonic()
            self.assertEqual(sweep.run_logged(
                [sys.executable, '-c', 'import time; time.sleep(60)'],
                os.environ.copy(), io.StringIO(), 0.5), 124)
            self.assertLess(time.monotonic() - begin, 10)
            self.assertEqual(sweep.run_logged(
                [sys.executable, '-c', 'raise SystemExit(7)'],
                os.environ.copy(), io.StringIO(), 10), 7)

    def test_missing_hardware_is_failure_for_peer_benchmark(self):
        from types import SimpleNamespace
        with patch.object(perf, 'detect_torch_backend', return_value=SimpleNamespace(
                name='cpu', reason='unit test')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(perf.main(sweep.benchmark_args(self.args())), 2)


if __name__ == '__main__':
    unittest.main()
