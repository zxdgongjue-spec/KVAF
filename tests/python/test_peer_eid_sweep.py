import contextlib
import csv
import io
import os
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

import peer_eid_profile as report
import run_peer_eid_sweep as sweep
import test_npu_batch_get_perf as perf

EIDS = ['1' * 32, '2' * 32]


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

            def child(command, **kwargs):
                parsed = perf.parse_args(command[3:])
                self.assertEqual(parsed.peer_eids, EIDS)
                self.assertEqual(kwargs['env']['KVAF_RUN_NPU_TESTS'], '1')
                self.assertEqual(kwargs['env']['DEVICE_ID'], '7')
                self.assertTrue(kwargs['env']['PYTHONPATH'].startswith(str(root / 'python')))
                kwargs['stdout'].write('native and Python log\n')
                return subprocess.CompletedProcess(command, 9)

            with patch.dict(os.environ, {'KVAF_CPU_SERVICE_ADDR': '127.0.0.1:29000',
                                         'DEVICE_ID': '7'}, clear=True), \
                 patch.object(sweep.subprocess, 'run', side_effect=child) as run, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(sweep.main(args), 9)
                run.assert_called_once()
                self.assertEqual((output / 'batch_get.log').read_text(), 'native and Python log\n')
                self.assertEqual(json.loads((output / 'config.json').read_text())['execution'], 'single_kernel')
                self.assertEqual(sweep.main(args), 2)  # preserve old results
                run.assert_called_once()

    def test_missing_hardware_is_failure_for_peer_benchmark(self):
        from types import SimpleNamespace
        with patch.object(perf, 'detect_torch_backend', return_value=SimpleNamespace(
                name='cpu', reason='unit test')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(perf.main(sweep.benchmark_args(self.args())), 2)


if __name__ == '__main__':
    unittest.main()
