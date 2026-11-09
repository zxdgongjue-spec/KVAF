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


def fake_build(folder):
    root = Path(folder)
    (root / 'bin').mkdir()
    (root / 'python/kvaf').mkdir(parents=True)
    for name in ('get', 'put'):
        (root / f'bin/kvaf_execute_batch_{name}_kernels.o').touch()
    return root


def fake_sweep_benchmark(command, env, stream, timeout):
    args = perf.parse_args(command[3:])
    row = dict.fromkeys(perf.jetty_profile.FIELDS, '0')
    row.update(token_bytes=args.token_bytes, miss_count=args.miss_count,
               aiv_cores=args.aiv_cores, jettys_per_core=args.jettys_per_core)
    perf.jetty_profile.write_csv(args.jetty_csv, [row])
    Path(args.peer_csv).write_text('peer_eid\n' + '\n'.join(args.peer_eids) + '\n')
    if len(args.peer_eids) == 1:
        marker = 'PEER_EID_BASELINE'
    elif args.jettys_per_core < len(args.peer_eids):
        marker = 'PEER_EID_MULTI_CORE_TRAFFIC'
    else:
        marker = 'PEER_EID_CONCURRENCY'
    stream.write(profile_log().replace('p50_cycles=400', f'p50_cycles={args.miss_count}') +
                 f'\nBATCH_GET_JETTY_CSV_WRITTEN csv\n{marker} execution=single_kernel\n')
    return 0


def sample(serial=False, peer_count=2, jettys=4, paths=None):
    lanes = []
    for i in range(jettys):
        offset = 40 if serial and i % peer_count == peer_count - 1 else i
        lanes.append(dict(eid_path=paths[i] if paths is not None else i % peer_count,
                          requests=128, completion_count=1,
                          first_doorbell_cycle=10 + offset, last_doorbell_cycle=10 + offset,
                          first_completion_cycle=30 + offset, last_completion_cycle=30 + offset))
    return [{"jettys": lanes}]


class PeerEidTests(unittest.TestCase):
    def args(self, *extra):
        with patch.dict(os.environ, {}, clear=True):
            _, (run,) = sweep.parse_args(['build-npu', '--peer-eids', ','.join(EIDS),
                                          '--npu-eid', '3' * 32, '--output-dir', 'logs', *extra])
            return run

    def test_command_runs_all_eids_in_one_benchmark(self):
        for count in (2, 3, 4, 8, 64):
            with self.subTest(peer_count=count):
                peers = [f'{i + 1:032x}' for i in range(count)]
                jettys = max(4, count)
                args = self.args('--peer-eids', ','.join(peers), '--cores', '1',
                                 '--jettys', str(jettys))
                parsed = perf.parse_args(sweep.benchmark_args(args))
                self.assertEqual(parsed.peer_eids, peers)
                self.assertEqual(parsed.npu_eids, ['3' * 32])
                self.assertEqual(parsed.aiv_cores, 1)
                self.assertEqual(parsed.jettys_per_core, jettys)
                self.assertTrue(parsed.kernel_profile)
                self.assertEqual(parsed.max_inflight_total, max(2048, jettys * 256))
                self.assertIsNone(perf.validate_args(parsed))

    def test_default_budget_scales_to_activate_all_routes(self):
        args = self.args('--cores', '4', '--jettys', '4')
        self.assertEqual(args.max_inflight_total, 4096)
        self.assertIsNone(perf.validate_args(perf.parse_args(sweep.benchmark_args(args))))

    def test_single_peer_uses_original_shape_limits_and_budget(self):
        for extra in ([], ['--jettys', '1'], ['--miss-count', '1'], ['--miss-count', '0'],
                      ['--miss-count', '512', '--max-inflight-total', '2'],
                      ['--cores', '4', '--jettys', '4', '--miss-count', '1']):
            with self.subTest(extra=extra):
                args = self.args('--peer-eids', EIDS[0], *extra)
                parsed = perf.parse_args(sweep.benchmark_args(args))
                self.assertEqual(parsed.peer_eids, EIDS[:1])
                self.assertEqual(parsed.npu_eids, ['3' * 32])
                self.assertIsNone(perf.validate_args(parsed))
                if '--max-inflight-total' not in extra:
                    self.assertEqual(parsed.max_inflight_total, 2048)

    def test_miss_count_and_clock_are_forwarded(self):
        args = self.args('--miss-count', '2048', '--kernel-profile-clock-mhz', '800')
        parsed = perf.parse_args(sweep.benchmark_args(args))
        self.assertEqual(parsed.request_length, 32768)
        self.assertEqual(parsed.miss_count, 2048)
        self.assertEqual(parsed.kernel_profile_clock_mhz, 800)
        with patch.dict(os.environ, {'KVAF_BATCH_GET_KERNEL_CLOCK_MHZ': '900'}):
            parsed, _ = sweep.parse_args(['build-npu', '--peer-eids', ','.join(EIDS),
                                          '--npu-eid', '3' * 32, '--output-dir', 'logs'])
            self.assertEqual(parsed.kernel_profile_clock_mhz, 900)

    def test_single_jetty_selects_one_peer_before_validation_and_budget(self):
        for selected in EIDS:
            with self.subTest(selected=selected), \
                 patch('npu_eids.random.choice', return_value=selected) as choose:
                args = self.args('--cores', '1', '--jettys', '1', '--miss-count', '0')
                parsed = perf.parse_args(sweep.benchmark_args(args))
                choose.assert_called_once_with(EIDS)
                self.assertEqual(args.requested_peer_eids, EIDS)
                self.assertEqual(args.peer_eids, [selected])
                self.assertEqual(parsed.peer_eids, [selected])
                self.assertEqual(args.mode, 'single_eid_baseline')
                self.assertEqual(args.max_inflight_total, 2048)
                self.assertIsNone(perf.validate_args(parsed))

    def test_multiple_cores_one_jetty_keep_all_peers(self):
        for cores in (2, 4, 16):
            with self.subTest(cores=cores), patch('npu_eids.random.choice') as choose:
                args = self.args('--cores', str(cores), '--jettys', '1')
                parsed = perf.parse_args(['--peer-eids', ','.join(EIDS),
                                          '--npu-eids', '3' * 32, '--aiv-cores', str(cores),
                                          '--jettys-per-core', '1',
                                          '--max-inflight-total', str(args.max_inflight_total)])
                choose.assert_not_called()
                self.assertEqual(args.peer_eids, EIDS)
                self.assertEqual(parsed.peer_eids, EIDS)
                self.assertEqual(args.mode, 'dual_eid_concurrent')
                self.assertEqual(args.max_inflight_total, max(2048, cores * 256))
                self.assertIsNone(perf.validate_args(parsed))

    def test_direct_benchmark_selects_and_logs_one_peer(self):
        from types import SimpleNamespace
        argv = ['--peer-eids', ','.join(EIDS), '--npu-eids', '3' * 32,
                '--aiv-cores', '1', '--jettys-per-core', '1', '--miss-count', '0']
        with patch.dict(os.environ, {}, clear=True), \
             patch('npu_eids.random.choice', return_value=EIDS[1]) as choose:
            args = perf.parse_args(argv)
            choose.assert_called_once_with(EIDS)
            self.assertEqual(args.requested_peer_eids, EIDS)
            self.assertEqual(args.peer_eids, EIDS[1:])
            self.assertIsNone(perf.validate_args(args))
            console = io.StringIO()
            choose.reset_mock()
            with patch.object(perf, 'detect_torch_backend', return_value=SimpleNamespace(
                    name='cpu', reason='unit test')), contextlib.redirect_stdout(console):
                self.assertEqual(perf.main(argv), 2)
            choose.assert_called_once_with(EIDS)
            self.assertIn(f'selected_peer_eid={EIDS[1]}', console.getvalue())
            self.assertIn('mode=single_eid_baseline', console.getvalue())
            self.assertNotIn('invalid BatchGet', console.getvalue())

    def test_rejects_serializing_or_idle_shapes(self):
        for extra in [('--request-length', '128'), ('--max-inflight-total', '2'),
                      ('--cores', '0'), ('--iterations', '0'), ('--jettys', '0'),
                      ('--jettys', '4,0'), ('--cores', '1,1'), ('--miss-count', '2048,'),
                      ('--peer-eids', ''),
                      ('--peer-eids', ','.join(EIDS + ['4' * 32]), '--jettys', '2'),
                      ('--peer-eids', ','.join(EIDS + ['4' * 32]), '--cores', '2', '--jettys', '1'),
                      ('--cores', '2', '--jettys', '1', '--miss-count', '0'),
                      ('--peer-eids', ','.join(f'{i + 1:032x}' for i in range(5))),
                      ('--npu-eid', ''), ('--npu-eid', ','.join(EIDS)),
                      ('--peer-eids', ','.join([EIDS[0]] * 2))]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.args(*extra)

    def test_profile_detects_overlap_and_aggregates_all_paths(self):
        for peers, jettys, requests in ((EIDS, 4, [256, 256]),
                                       (EIDS + ['4' * 32], 4, [256, 128, 128]),
                                       (EIDS + ['4' * 32, '5' * 32], 4, [128] * 4)):
            with self.subTest(peer_count=len(peers)):
                rows = report.summarize([sample(peer_count=len(peers), jettys=jettys)],
                                        peers, 1, jettys, 656)
                self.assertEqual([r['peer_eid'] for r in rows], peers)
                self.assertEqual([r['requests'] for r in rows], requests)
                self.assertEqual([r['bytes'] for r in rows], [n * 656 for n in requests])
                self.assertTrue(all(r['first_wave_observed_overlap_cycles'] > 0 for r in rows))

    def test_single_peer_allows_idle_lanes_and_cores_without_overlap(self):
        for requests in (0, 1, 128):
            with self.subTest(requests=requests), tempfile.TemporaryDirectory() as folder:
                profiles = sample() + sample()
                for core, profile in enumerate(profiles):
                    for slot, lane in enumerate(profile['jettys']):
                        lane['eid_path'] = 0
                        if core == 0 and slot == 0 and requests:
                            lane['requests'] = requests
                        else:
                            for key in ('requests', 'completion_count', 'first_doorbell_cycle',
                                        'last_doorbell_cycle', 'first_completion_cycle',
                                        'last_completion_cycle'):
                                lane[key] = 0
                rows = report.summarize([profiles], EIDS[:1], 2, 4, 656)
                self.assertEqual([row['requests'] for row in rows], [requests, 0])
                self.assertEqual([row['bytes'] for row in rows], [requests * 656, 0])
                self.assertTrue(all(row['first_wave_observed_overlap_cycles'] == '' for row in rows))
                console = io.StringIO()
                with contextlib.redirect_stdout(console):
                    report.report([profiles], EIDS[:1], 2, 4, 656, Path(folder) / 'peer.csv')
                self.assertIn('PEER_EID_BASELINE', console.getvalue())
                self.assertIn('concurrency_check=not_applicable', console.getvalue())
                self.assertNotIn('PEER_EID_CONCURRENCY', console.getvalue())
                self.assertEqual(console.getvalue().count('PEER_EID_TRAFFIC'), 1)

    def test_single_peer_keeps_route_and_completion_validation(self):
        profiles = sample()
        for lane in profiles[0]['jettys']:
            lane['eid_path'] = 0
        profiles[0]['jettys'][0]['completion_count'] = 0
        with self.assertRaisesRegex(ValueError, 'no completed traffic'):
            report.summarize([profiles], EIDS[:1], 1, 4, 656)
        profiles[0]['jettys'][0]['eid_path'] = 1
        with self.assertRaisesRegex(ValueError, 'route'):
            report.summarize([profiles], EIDS[:1], 1, 4, 656)

    def test_serial_evidence_fails_but_preserves_csv(self):
        for peers in (EIDS, EIDS + ['4' * 32], EIDS + ['4' * 32, '5' * 32]):
            with self.subTest(peer_count=len(peers)), tempfile.TemporaryDirectory() as folder, \
                 contextlib.redirect_stdout(io.StringIO()):
                path = Path(folder) / 'samples.csv'
                with self.assertRaisesRegex(ValueError, 'not observed'):
                    report.report([sample(serial=True, peer_count=len(peers))],
                                  peers, 1, 4, 656, path)
                with path.open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), len(peers))

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

    def test_cross_core_peers_have_traffic_without_comparing_clocks(self):
        profiles = sample(jettys=1, paths=[0]) + sample(jettys=1, paths=[1])
        for key in ('first_doorbell_cycle', 'last_doorbell_cycle',
                    'first_completion_cycle', 'last_completion_cycle'):
            profiles[1]['jettys'][0][key] += 100000
        rows = report.summarize([profiles], EIDS, 2, 1, 656)
        self.assertEqual([(r['core'], r['peer_path'], r['requests']) for r in rows],
                         [(0, 0, 128), (1, 1, 128)])
        self.assertTrue(all(r['first_wave_observed_overlap_cycles'] == '' for r in rows))
        console = io.StringIO()
        with contextlib.redirect_stdout(console):
            report.report([profiles], EIDS, 2, 1, 656, '')
        self.assertIn('PEER_EID_MULTI_CORE_TRAFFIC', console.getvalue())
        self.assertIn('cross_core_overlap=not_measured', console.getvalue())
        self.assertNotIn('PEER_EID_CONCURRENCY', console.getvalue())
        self.assertEqual(console.getvalue().count('PEER_EID_TRAFFIC'), 2)
        profiles[1]['jettys'][0]['eid_path'] = 0
        with self.assertRaisesRegex(ValueError, 'route'):
            report.summarize([profiles], EIDS, 2, 1, 656)

    def test_cross_core_distribution_keeps_local_overlap_and_coverage_checks(self):
        peers = EIDS + ['4' * 32]
        profiles = sample(jettys=2, paths=[0, 2]) + sample(jettys=2, paths=[1, 0])
        rows = report.summarize([profiles], peers, 2, 2, 656)
        self.assertEqual({(r['core'], r['peer_path']) for r in rows},
                         {(0, 0), (0, 2), (1, 0), (1, 1)})
        self.assertTrue(all(r['first_wave_observed_overlap_cycles'] > 0 for r in rows))
        for key in ('first_doorbell_cycle', 'last_doorbell_cycle',
                    'first_completion_cycle', 'last_completion_cycle'):
            profiles[0]['jettys'][1][key] += 40
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'not observed'):
            report.report([profiles], peers, 2, 2, 656, '')
        profiles = sample(jettys=1, paths=[0]) + sample(jettys=1, paths=[1])
        with self.assertRaisesRegex(ValueError, 'no traffic'):
            report.summarize([profiles], peers, 2, 1, 656)
        profiles[1]['jettys'][0]['completion_count'] = 0
        with self.assertRaisesRegex(ValueError, 'no completed traffic'):
            report.summarize([profiles], EIDS, 2, 1, 656)

    def test_launcher_one_child_environment_logs_and_failure_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = fake_build(folder)
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
            row = sweep.summary_row(args, profile_log())
            sweep.write_summary([row], args.kernel_profile_clock_mhz, Path(folder), lines.append)
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

    def test_sweep_four_axes_in_order_with_separate_outputs_and_combined_csvs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = fake_build(folder)
            output = root / 'logs'
            args = [str(root), '--peer-eids', ','.join(EIDS), '--npu-eid', '3' * 32,
                    '--output-dir', str(output), '--token-bytes', '656,1024', '--miss-count', '2048,4096',
                    '--cores', '1,4', '--jettys', '2,4']
            expected = [(t, m, c, j) for t in (656, 1024) for m in (2048, 4096)
                        for c in (1, 4) for j in (2, 4)]
            with patch.dict(os.environ, {'KVAF_CPU_SERVICE_ADDR': 'localhost:29000'}, clear=True), \
                 patch.object(sweep, 'run_logged', side_effect=fake_sweep_benchmark) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(sweep.main(args), 0)
            self.assertEqual(run.call_count, len(expected))
            for (t, m, c, j), call in zip(expected, run.call_args_list):
                parsed = perf.parse_args(call.args[0][3:])
                self.assertEqual((parsed.token_bytes, parsed.miss_count, parsed.aiv_cores,
                                  parsed.jettys_per_core), (t, m, c, j))
                case = output / f'token_bytes_{t}_miss_count_{m}_aiv_cores_{c}_jettys_per_core_{j}'
                config = json.loads((case / 'config.json').read_text())
                self.assertEqual(config['max_inflight_total'], max(2048, c * j * 256))
                self.assertEqual(config['peer_eids'], EIDS)
                self.assertTrue((case / 'peer_samples.csv').is_file())
                self.assertIn(f'p50_cycles={m}', (case / 'batch_get.log').read_text())
            for filename in ('batch_get_miss_sweep_summary.csv', 'batch_get_jetty_summary.csv'):
                with (output / filename).open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([tuple(int(row[k]) for k in sweep.SHAPE_COLUMNS) for row in rows], expected)
                if filename == 'batch_get_miss_sweep_summary.csv':
                    self.assertEqual([r['p50_us'] for r in rows], [f'{m / 1000:.3f}' for _, m, _, _ in expected])
            log = (output / 'batch_get.log').read_text()
            self.assertIn('jetty_counts=2 combinations=16', log)
            self.assertIn('BATCH_GET_MISS_SWEEP_SUMMARY combinations=16', log)
            self.assertEqual(log.count('BATCH_GET_MISS_SWEEP_STAGE_SUMMARY'), 3)

    def test_random_baseline_and_mixed_sweep_keep_effective_eids_in_outputs(self):
        for cores, jettys in (('1', '1'), ('1', '1,2,4'), ('1,2', '1,2')):
            with self.subTest(cores=cores, jettys=jettys), tempfile.TemporaryDirectory() as folder:
                root = fake_build(folder)
                output = root / 'logs'
                argv = [str(root), '--peer-eids', ','.join(EIDS), '--npu-eid', '3' * 32,
                        '--output-dir', str(output), '--cores', cores, '--jettys', jettys]
                console = io.StringIO()
                with patch.dict(os.environ, {'KVAF_CPU_SERVICE_ADDR': 'localhost:29000'}, clear=True), \
                     patch('npu_eids.random.choice', return_value=EIDS[1]) as choose, \
                     patch.object(sweep, 'run_logged', side_effect=fake_sweep_benchmark) as run, \
                     contextlib.redirect_stdout(console):
                    self.assertEqual(sweep.main(argv), 0)
                    choose.assert_called_once_with(EIDS)
                self.assertEqual(run.call_count, len(cores.split(',')) * len(jettys.split(',')))
                for call in run.call_args_list:
                    parsed = perf.parse_args(call.args[0][3:])
                    single = parsed.aiv_cores * parsed.jettys_per_core == 1
                    peers = EIDS[1:] if single else EIDS
                    mode = 'single_eid_baseline' if single else 'dual_eid_concurrent'
                    case = Path(parsed.peer_csv).parent
                    config = json.loads((case / 'config.json').read_text())
                    self.assertEqual(config['requested_peer_eids'], EIDS)
                    self.assertEqual(config['peer_eids'], peers)
                    self.assertEqual(config['benchmark']['peer_eids'], peers)
                    self.assertEqual(config['mode'], mode)
                    self.assertEqual(parsed.peer_eids, peers)
                    with (case / 'peer_samples.csv').open() as stream:
                        self.assertEqual([row['peer_eid'] for row in csv.DictReader(stream)], peers)
                    log = (case / 'batch_get.log').read_text()
                    marker = ('PEER_EID_BASELINE' if single else
                              'PEER_EID_MULTI_CORE_TRAFFIC' if parsed.jettys_per_core < len(peers) else
                              'PEER_EID_CONCURRENCY')
                    self.assertIn(marker, log)
                    if single:
                        self.assertIn(f'selected_peer_eid={EIDS[1]}', log)
                        self.assertNotIn('PEER_EID_CONCURRENCY', log)
                mode = 'single_eid_baseline' if run.call_count == 1 else 'mixed'
                self.assertIn(f'mode={mode}', (output / 'batch_get.log').read_text())
                self.assertIn(f'selected_peer_eid={EIDS[1]}', console.getvalue())
                with (output / 'batch_get_miss_sweep_summary.csv').open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), run.call_count)

    def test_sweep_stops_at_failed_case_and_keeps_prior_logs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = fake_build(folder)
            output = root / 'logs'
            args = [str(root), '--peer-eids', ','.join(EIDS), '--npu-eid', '3' * 32,
                    '--output-dir', str(output), '--miss-count', '2048,4096,8192']

            def timeout_second(command, env, stream, timeout):
                if perf.parse_args(command[3:]).miss_count == 4096:
                    stream.write('timed out here\n')
                    return 124
                return fake_sweep_benchmark(command, env, stream, timeout)

            with patch.dict(os.environ, {'KVAF_CPU_SERVICE_ADDR': 'localhost:29000'}, clear=True), \
                 patch.object(sweep, 'run_logged', side_effect=timeout_second) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(sweep.main(args), 124)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(len(list(output.glob('*/batch_get.log'))), 2)
            self.assertEqual(len(list(output.glob('*/jettys.csv'))), 1)
            log = (output / 'batch_get.log').read_text()
            self.assertIn('BATCH_GET_MISS_SWEEP_FAILED token_bytes=656 miss_count=4096', log)
            self.assertNotIn('BATCH_GET_MISS_SWEEP_SUMMARY', log)
            self.assertFalse(any(output.glob('*summary.csv')))

    def test_shell_sweep_lists_and_explicit_option_precedence(self):
        env = dict(KVAF_BATCH_GET_SWEEP_TOKEN_BYTES='656 512', KVAF_BATCH_GET_SWEEP_MISS_COUNTS='2048 4096',
                   KVAF_BATCH_GET_SWEEP_AIV_CORES='1 2', KVAF_BATCH_GET_SWEEP_JETTYS_PER_CORE='2 4')
        with patch.dict(os.environ, env, clear=True):
            _, runs = sweep.parse_args(['build-npu', '--peer-eids', ','.join(EIDS), '--npu-eid', '3' * 32,
                                        '--output-dir', 'logs', '--cores', '4'])
        self.assertEqual([tuple(sweep.shape_fields(run).values()) for run in runs],
                         [(t, m, 4, j) for t in (656, 512) for m in (2048, 4096) for j in (2, 4)])

    def test_successful_launcher_shows_and_saves_complete_summary(self):
        for peers in (EIDS[:1], EIDS, EIDS + ['4' * 32], EIDS + ['4' * 32, '5' * 32]):
            marker = 'PEER_EID_BASELINE' if len(peers) == 1 else 'PEER_EID_CONCURRENCY'
            mode = {1: 'single_eid_baseline', 2: 'dual_eid_concurrent'}.get(
                len(peers), 'multi_eid_concurrent')
            with self.subTest(peer_count=len(peers)), tempfile.TemporaryDirectory() as folder:
                root = fake_build(folder)
                output = root / 'logs'
                args = [str(root), '--peer-eids', ','.join(peers), '--npu-eid', '3' * 32,
                        '--output-dir', str(output), '--miss-count', '2048']

                def child(command, env, stream, timeout):
                    parsed = perf.parse_args(command[3:])
                    self.assertEqual(parsed.peer_eids, peers)
                    row = dict.fromkeys(perf.jetty_profile.FIELDS, '0')
                    perf.jetty_profile.write_csv(output / 'jettys.csv', [row])
                    peer_console = io.StringIO()
                    with contextlib.redirect_stdout(peer_console):
                        report.report([sample(peer_count=len(peers))], peers, 1, 4, 656,
                                      output / 'peer_samples.csv')
                    text = ('BATCH_GET_PERF_PHASE_BEGIN phase=initialize\n' + profile_log() +
                            '\nBATCH_GET_JETTY_CSV_WRITTEN path=jettys.csv rows=1\n' +
                            peer_console.getvalue())
                    stream.write(text)
                    stream.flush()
                    print(text, end='')
                    return 0

                console = io.StringIO()
                with patch.dict(os.environ, {'KVAF_CPU_SERVICE_ADDR': 'localhost:29000'}, clear=True), \
                     patch.object(sweep, 'run_logged', side_effect=child) as run, \
                     contextlib.redirect_stdout(console):
                    self.assertEqual(sweep.main(args), 0)
                    run.assert_called_once()
                text = console.getvalue()
                self.assertEqual(text, (output / 'batch_get.log').read_text())
                self.assertEqual(text.count('BATCH_GET_MISS_SWEEP_STAGE_SUMMARY'), 3)
                self.assertEqual(text.count(marker), 1)
                self.assertIn(f'peer_count={len(peers)}', text)
                self.assertEqual(text.count('PEER_EID_CONCURRENCY'), int(len(peers) > 1))
                self.assertEqual(text.count('PEER_EID_TRAFFIC'), len(peers))
                self.assertIn(f'mode={mode}', text)
                config = json.loads((output / 'config.json').read_text())
                self.assertEqual(config['mode'], mode)
                self.assertEqual(config['peer_eids'], peers)
                with (output / 'peer_samples.csv').open() as stream:
                    self.assertEqual([row['peer_eid'] for row in csv.DictReader(stream)], peers)
                self.assertIn('BATCH_GET_JETTY_SWEEP_CSV_WRITTEN', text)
                self.assertTrue((output / 'batch_get_miss_sweep_summary.csv').is_file())
                self.assertEqual((output / 'batch_get_jetty_summary.csv').read_text(),
                                 (output / 'jettys.csv').read_text())

    def test_missing_or_invalid_profile_rejects_summary_row(self):
        for text in ('', profile_log().replace('p50_cycles=400', 'p50_cycles=nan'),
                     profile_log().replace('p50_cycles=400', 'p50_cycles=0')):
            with self.assertRaises(ValueError):
                sweep.summary_row(self.args(), text)

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
