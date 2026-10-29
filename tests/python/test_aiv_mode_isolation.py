"""Independent entry points and binary selection, without importing torch/NPU."""
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import aiv_batch_get_test_common as common
import test_aiv_sequential_batch_get as suite


class ModeIsolationTests(unittest.TestCase):
    def test_selected_binary_only_and_legacy_selector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            gather = root / 'kvaf_gather_batch_get_kernels.o'
            scatter = root / 'kvaf_scatter_batch_get_kernels.o'
            put = root / 'put.o'
            gather.touch()
            put.touch()
            for selector in ('kvaf_execute_batch_get_kernels.o', 'kvaf_sequential_batch_get_kernels.o'):
                with patch.dict(os.environ, {'KVAF_BATCH_GET_KERNEL': str(root / selector),
                                             'KVAF_BATCH_PUT_KERNEL': str(put)}):
                    self.assertEqual(common.kernel_paths('gather_read'), (str(gather), str(put)))
                    with self.assertRaisesRegex(ValueError, 'kvaf_scatter_batch_get_kernels.o'):
                        common.kernel_paths('cpu_push_scatter')
            gather.unlink()
            scatter.touch()
            with patch.dict(os.environ, {'KVAF_BATCH_GET_KERNEL': str(scatter),
                                         'KVAF_BATCH_PUT_KERNEL': str(put)}):
                self.assertEqual(common.kernel_paths('cpu_push_scatter'), (str(scatter), str(put)))
                with self.assertRaisesRegex(ValueError, 'requires kvaf_gather'):
                    common.kernel_paths('gather_read')

    def test_fixed_entry_ignores_suite_mode_selector(self):
        for name, mode in (('scatter', 'cpu_push_scatter'), ('gather', 'gather_read')):
            with patch.dict(os.environ, {'KVAF_AIV_BATCH_GET_MODES': 'invalid'}), \
                 patch.object(common, 'run', return_value=0) as run:
                with self.assertRaises(SystemExit) as result:
                    runpy.run_path(str(Path(__file__).with_name(f'test_aiv_{name}_batch_get.py')),
                                   run_name='__main__')
                self.assertEqual(result.exception.code, 0)
                run.assert_called_once_with(mode)

    def test_suite_runs_next_process_after_failure(self):
        with patch.dict(os.environ, {'KVAF_RUN_NPU_TESTS': '1',
                                     'KVAF_AIV_BATCH_GET_MODES': 'cpu_push_scatter,gather_read'}), \
             patch.object(suite.subprocess, 'run', side_effect=[
                 subprocess.CompletedProcess([], 1), subprocess.CompletedProcess([], 0)]) as run:
            self.assertEqual(suite.main(), 1)
            self.assertEqual(run.call_count, 2)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual([Path(c[1]).name for c in commands],
                             ['test_aiv_scatter_batch_get.py', 'test_aiv_gather_batch_get.py'])
            self.assertTrue(all(c[0] == sys.executable for c in commands))

    def test_gather_ignores_scatter_probe(self):
        # Stop at the first hardware import. A scatter-only invalid probe must
        # not reject the gather test before it reaches its hardware setup.
        original_import = __import__
        def importing(name, *args, **kwargs):
            if name == 'backend':
                raise RuntimeError('reached hardware setup')
            return original_import(name, *args, **kwargs)
        with patch.dict(os.environ, {'KVAF_RUN_NPU_TESTS': '1', 'KVAF_AIV_SCATTER_PROBE': 'invalid'}), \
             patch.object(common, 'kernel_paths', return_value=('gather.o', 'put.o')), \
             patch('builtins.__import__', side_effect=importing):
            with self.assertRaisesRegex(RuntimeError, 'reached hardware setup'):
                common.run('gather_read')
            with self.assertRaisesRegex(ValueError, 'scatter probe'):
                common.run('cpu_push_scatter')


if __name__ == '__main__':
    unittest.main()
