"""Independent entry points and binary selection, without importing torch/NPU."""
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import aiv_batch_get_test_common as common


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

    def test_fixed_entry_selects_its_own_mode(self):
        for name, mode in (('scatter', 'cpu_push_scatter'), ('gather', 'gather_read')):
            with patch.dict(os.environ, {}), \
                 patch.object(common, 'run', return_value=0) as run:
                with self.assertRaises(SystemExit) as result:
                    runpy.run_path(str(Path(__file__).with_name(f'test_aiv_{name}_batch_get.py')),
                                   run_name='__main__')
                self.assertEqual(result.exception.code, 0)
                run.assert_called_once_with(mode)

    def test_scatter_temporarily_skips_payload_even_with_stale_probe(self):
        for stale_probe in ('', 'load', 'dma', 'copy'):
            def run(mode):
                self.assertEqual(mode, 'cpu_push_scatter')
                self.assertEqual(os.environ['KVAF_AIV_SCATTER_PROBE'], 'done')
                return 0
            with patch.dict(os.environ, {'KVAF_AIV_SCATTER_PROBE': stale_probe}), \
                 patch.object(common, 'run', side_effect=run):
                with self.assertRaises(SystemExit) as result:
                    runpy.run_path(str(Path(__file__).with_name('test_aiv_scatter_batch_get.py')),
                                   run_name='__main__')
                self.assertEqual(result.exception.code, 0)

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
