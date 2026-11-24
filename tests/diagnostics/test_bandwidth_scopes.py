import ast
from itertools import count
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from pathlib import Path
from bandwidth_scopes import instrument, summarize_sample


class ScopesTest(unittest.TestCase):
    def test_upstream_instrumentation(self):
        source = Path(__file__).resolve().parents[1]/'python/test_npu_batch_get_perf.py'
        compile(instrument(source.read_text()), str(source), 'exec')

    def test_fail_closed(self):
        with self.assertRaises(RuntimeError):
            instrument('')

    def test_current_eager_samples_remain_paired(self):
        source = Path(__file__).resolve().parents[1]/'python/test_npu_batch_get_perf.py'
        tree = ast.parse(instrument(source.read_text()))
        run_case = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'run_case')
        body = next(node.body for node in run_case.body if isinstance(node, ast.Try))
        start = next(i for i, node in enumerate(body)
                     if isinstance(node, ast.AnnAssign) and node.target.id == 'submit_samples')
        stop = next(i for i, node in enumerate(body)
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name) and node.value.func.id == 'print_measurements')
        samples = compile(ast.Module(body=body[start:stop + 1], type_ignores=[]), str(source), 'exec')

        for timing in ('async-drain', 'sync-call'):
            with self.subTest(timing=timing):
                case = self

                class Event:
                    def __init__(self, enable_timing):
                        case.assertTrue(enable_timing)
                        self.completed = False

                    def record(self):
                        pass

                    def synchronize(self):
                        self.completed = True

                    def elapsed_time(self, end):
                        case.assertTrue(end.completed)
                        return 0.25

                args = SimpleNamespace(iterations=3, warmup=0, get_timing=timing, sgl_payload_only=False,
                                       execution_mode='unified')
                runtime = SimpleNamespace(batch_get=Mock(), batch_get_kernel_profile=Mock())
                runtime.batch_get_kernel_profile.side_effect = lambda: [dict(sample=runtime.batch_get.call_count)]
                execution = SimpleNamespace(get_kwargs={}, complete_async_get=Mock())
                save, check = Mock(), Mock()
                environment = dict(
                    args=args, runtime=runtime, execution=execution,
                    torch=SimpleNamespace(npu=SimpleNamespace(Event=Event, synchronize=Mock())),
                    time=SimpleNamespace(perf_counter=lambda: next(clock)),
                    create_request_plan=lambda: (0, 0, 0), output=SimpleNamespace(zero_=Mock()),
                    device_table=object(), miss_count=object(), request_length=1,
                    miss_count_value=1, token_bytes=656, uses_aicpu=lambda _: False,
                    check_output=check, scope_save=save, print_samples=Mock(),
                    print_measurements=Mock(), report=None,
                )
                clock = count(0, 0.001)
                exec(samples, environment)
                save.assert_called_once()
                config, host, submit, events, profiles = save.call_args.args
                self.assertIs(config, args)
                self.assertEqual([len(values) for values in (host, submit, events, profiles)], [3] * 4)
                self.assertEqual(events, [250.0] * 3)  # Event ms converted to us.
                self.assertEqual(profiles, [[dict(sample=i)] for i in range(1, 4)])
                for value in host:
                    self.assertAlmostEqual(value, 2.0 if timing == 'async-drain' else 1.0)
                for value in submit:
                    self.assertAlmostEqual(value, 1.0)
                self.assertEqual(execution.complete_async_get.call_count, 3 if timing == 'async-drain' else 0)
                self.assertEqual(check.call_count, 3)
                for call in runtime.batch_get.call_args_list:
                    self.assertEqual(call.kwargs['synchronize'], timing == 'sync-call')

    def test_paired_scopes(self):
        p = dict(kernel_start_cycle=100, kernel_finish_cycle=61100,
                 first_doorbell_cycle=4100, completion_done_cycle=60100)
        result = summarize_sample([p], 1000, 1343488, 212, 141, 200)
        self.assertEqual(result['kernel_wall_us'], 61)
        self.assertEqual(result['transfer_window_us'], 56)
        self.assertAlmostEqual(result['GBps']['host'], 6.337207547, places=8)
        self.assertEqual(result['host_outside_kernel_us'], 151)


if __name__ == '__main__':
    unittest.main()
