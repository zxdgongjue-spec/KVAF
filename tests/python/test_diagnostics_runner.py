import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

RUNNER = Path(__file__).resolve().parents[1] / 'diagnose_remote_cpu.py'
spec = importlib.util.spec_from_file_location('diagnostics_runner', RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class DiagnosticsRunnerTests(unittest.TestCase):
    def test_fault_only_preserves_original_mode(self):
        result = subprocess.run([sys.executable, str(RUNNER), '/missing-build', '--dry-run', '--fault-only'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['cases'], [[1, 1, 1]])

    def test_raw_fault_evidence_survives_verbose_output(self):
        content = ('[KVAF_DIAG] unrelated repeated\n' * 5000
                   + '[KVAF_AIV] DIRECT_AIV_WQE_EID_ENCODING ra_host_eid=abcd\n'
                   + '[KVAF_DIAG] DEVICE_CQE words=02020006,00000000\n'
                   + '[KVAF_DIAG] DEVICE_WQE words=b0360000\n'
                   + '[KVAF_DIAG] REQUEST_CHECK metadata_valid=1\n'
                   + '[KVAF_DIAG] CQ_AUX_VALUES type:value=39:0x00000001\n'
                   + '[KVAF_DIAG] unrelated repeated\n' * 5000)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            runner.print_case_report(runner.TerminalReport(), 'fault', content)
        for marker in ('DEVICE_CQE', 'DEVICE_WQE', 'REQUEST_CHECK', 'CQ_AUX_VALUES', 'ra_host_eid=abcd'):
            self.assertIn(marker, stream.getvalue())
        self.assertLess(len(stream.getvalue().splitlines()), 200)

    def test_old_kernel_stops_after_completion_without_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            result, summary = self.run_matrix(directory, [(1, False,
                '[KVAF_DIAG] FIXED_PAIR\n[KVAF_DIAG] PUT_COMPLETION result=8\nDIAG_CLEANUP_OK\n')])
            self.assertEqual(result, 1)
            self.assertEqual(len(summary), 1)
            self.assertFalse(summary[0]['device_capture_v3'])

    def test_comparisons_hold_layout_constant(self):
        self.assertEqual(runner.matrix(), [(1, 1, 1), (1, 1, 0), (32, 2, 1), (32, 2, 0)])
        self.assertEqual(runner.matrix(True), [(1, 1, 1), (1, 1, 0)])

    def test_dry_run_needs_no_hardware_and_preserves_candidate(self):
        result = subprocess.run([sys.executable, str(RUNNER), '/missing-build', '--dry-run',
                                 '--peer-candidate', '3', '--npu-candidate', '1'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['peer_candidate'], 3)
        self.assertEqual(json.loads(result.stdout)['npu_candidate'], 1)

    def test_invalid_candidate_and_timeout_rejected(self):
        for option, value in [('--peer-candidate', '-1'), ('--npu-candidate', '4294967296'),
                              ('--timeout', '0'), ('--timeout', 'abc')]:
            result = subprocess.run([sys.executable, str(RUNNER), '/missing-build', '--dry-run', option, value],
                                    capture_output=True)
            self.assertEqual(result.returncode, 2)

    def test_failed_child_keeps_exit_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'case.log'
            code, timeout = runner.run_case([sys.executable, '-c', 'print("failure evidence"); raise SystemExit(7)'],
                                           os.environ.copy(), log, 5)
            self.assertEqual((code, timeout), (7, False))
            self.assertIn('failure evidence', log.read_text())

    def run_matrix(self, directory, outcomes):
        build = Path(directory) / 'build'
        (build / 'bin').mkdir(parents=True)
        (build / 'python/kvaf').mkdir(parents=True)
        for path in (build / 'bin/kvaf_execute_batch_get_kernels.o',
                     build / 'bin/kvaf_execute_batch_put_kernels.o',
                     build / 'python/kvaf/_kvaf.fake.so'):
            path.write_bytes(b'fake test artifact')
        output = Path(directory) / 'logs'

        def fake_case(command, env, log, timeout):
            code, timed_out, text = outcomes.pop(0)
            log.write_text(text)
            return code, timed_out

        with mock.patch.object(sys, 'argv', [str(RUNNER), str(build), '--output', str(output)]), \
             mock.patch.dict(os.environ, {'KVAF_CPU_SERVICE_ADDR': 'test:1234'}), \
             mock.patch.object(runner, 'capture', return_value='test inventory'), \
             mock.patch.object(runner, 'run_case', side_effect=fake_case), \
             mock.patch('builtins.print'):
            result = runner.main()
        return result, json.loads((output / 'summary.json').read_text())

    def test_clean_failures_continue_comparisons_but_exit_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            result, summary = self.run_matrix(directory, [
                (1, False, '[KVAF_DIAG] FIXED_PAIR\nDIAG_CLEANUP_OK\n') for _ in range(4)])
            self.assertEqual(result, 1)
            self.assertEqual(len(summary), 4)

    def test_unconfirmed_cleanup_stops_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            result, summary = self.run_matrix(directory, [(124, True, '[KVAF_DIAG] FIXED_PAIR\n')])
            self.assertEqual(result, 1)
            self.assertEqual(len(summary), 1)
            self.assertFalse(summary[0]['cleanup'])

    def test_terminal_hard_limit_including_multiline_messages(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            terminal = runner.TerminalReport()
            terminal.emit('line\n' * 1500)
            terminal.emit('must not leak beyond limit')
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 1000)
        self.assertIn('DIAG_TERMINAL_LIMIT', lines[-1])

    def test_four_large_cases_fit_report_and_keep_faults(self):
        content = ('[KVAF_AIV] DIRECT_JETTY_READY repeated\n' * 5000
                   + '[KVAF_DIAG] HBM_REGISTER token=0x1100\n'
                   + '[KVAF_DIAG] SQ_SLOT0 words=1234\n'
                   + '[KVAF_AIV] BATCH_PUT_DEVICE_ERROR cq_status=2 cq_sub_status=2\n'
                   + 'Traceback (most recent call last):\nRuntimeError: probe failed\nDIAG_CLEANUP_OK\n')
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            terminal = runner.TerminalReport()
            terminal.section('manifest', ['manifest'] * 500, 45)
            terminal.section('environment', ['env'] * 5000, 55)
            for case in range(4):
                runner.print_case_report(terminal, str(case), content)
            terminal.section('kernel faults', ['fault'] * 5000, 60)
            terminal.emit('DIAG_REPORT_END')
        output = stream.getvalue()
        self.assertLessEqual(len(output.splitlines()), 1000)
        self.assertEqual(output.count('HBM_REGISTER'), 8)  # evidence + tail per case
        self.assertIn('cq_status=2 cq_sub_status=2', output)
        self.assertTrue(output.rstrip().endswith('DIAG_REPORT_END'))

    def test_kernel_delta_retains_repeated_new_faults(self):
        self.assertEqual(runner.kernel_delta('UMMU fault\n', 'UMMU fault\nUMMU fault\nunrelated\n'),
                         ['UMMU fault'])

    def test_iommu_registration_noise_does_not_hide_fault(self):
        after = ('platform ummu_vdev.15.auto: Adding to iommu group 16\n' * 1000
                 + 'UMMU local access fault token=0x1100\n')
        self.assertEqual(runner.kernel_delta('', after), ['UMMU local access fault token=0x1100'])

    def test_missing_private_registration_stops_invalid_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            result, summary = self.run_matrix(directory, [(1, False,
                '[KVAF_DIAG] FIXED_PAIR\nPEER_LINK_EID_PAIR_RMA_PROBE_BEGIN\nDIAG_CLEANUP_OK\n')])
            self.assertEqual(result, 1)
            self.assertEqual(len(summary), 1)
            self.assertFalse(summary[0]['private_put_registration'])

    def test_hung_child_is_bounded_and_marked(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'case.log'
            code, timeout = runner.run_case([sys.executable, '-c', 'import time; time.sleep(60)'],
                                           os.environ.copy(), log, 0.2)
            self.assertEqual((code, timeout), (124, True))
            self.assertIn('cleanup_unconfirmed=1', log.read_text())


if __name__ == '__main__':
    unittest.main()
