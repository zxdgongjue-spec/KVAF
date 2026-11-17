"""Optional torch_npu CPU/CANN/device timeline collection for BatchGet."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path


def add_profiler_arguments(parser, *, output=True):
    parser.add_argument("--torch-npu-profiler", action=argparse.BooleanOptionalAction,
                        default=False, help="collect a visual CPU/NPU timeline (adds profiling overhead)")
    if output:
        parser.add_argument("--profiler-dir", type=Path,
                            help="timeline output root; each sweep shape gets its own subdirectory")
    parser.add_argument("--profiler-wait", type=int, default=0, help="measured rounds skipped before profiler warmup")
    parser.add_argument("--profiler-warmup", type=int, default=1, help="profiler warmup rounds (default: 1)")
    parser.add_argument("--profiler-active", type=int, default=3, help="rounds recorded once (default: 3)")
    parser.add_argument("--profiler-level", choices=("Level0", "Level1", "Level2"), default="Level1",
                        help="Level1 includes AscendCL APIs; Level2 also includes Runtime APIs")


def validate_profiler_args(args):
    if args.profiler_wait < 0 or args.profiler_warmup < 0 or args.profiler_active <= 0:
        return "profiler-wait/warmup must be non-negative and profiler-active must be positive"
    if args.torch_npu_profiler:
        if args.profiler_dir is None:
            return "torch-npu-profiler requires profiler-dir"
        if args.profiler_wait + args.profiler_warmup + args.profiler_active > args.iterations:
            return "iterations must cover profiler-wait + profiler-warmup + profiler-active"
    return None


class BatchGetProfiler:
    """No torch_npu import or trace annotations unless explicitly enabled.

    Capture and benchmark warmup are outside this context. One profiler step is
    one synchronized measurement round. Save raw data at the scheduled boundary,
    but defer parsing until all measured rounds/barriers have completed.
    """

    def __init__(self, args, torch):
        self.args, self.torch = args, torch
        self.profiler = None
        flow = os.environ.get("KVAF_BATCH_GET_FLOW_NAME", "standalone")
        self.worker = f"{flow}_device{os.environ.get('DEVICE_ID', '0')}_pid{os.getpid()}"

    def __enter__(self):
        if not self.args.torch_npu_profiler:
            return self
        import torch_npu.profiler as api
        from torch_npu.profiler.profiler import analyse

        self.analyse = analyse
        args = self.args
        stem = (f"token_bytes_{args.token_bytes}_miss_count_{args.miss_count}"
                f"_aiv_cores_{args.aiv_cores}_jettys_per_core_{args.jettys_per_core}")
        self.output = args.profiler_dir.resolve() / stem
        self.output.mkdir(parents=True, exist_ok=False)
        calls = args.graph_ring if args.execution_mode == "graph" else 1
        self.metadata = dict(worker=self.worker, device_id=os.environ.get("DEVICE_ID", "0"),
                             execution_mode=args.execution_mode, calls_per_round=calls,
                             wait=args.profiler_wait, warmup=args.profiler_warmup,
                             active=args.profiler_active, repeat=1, level=args.profiler_level,
                             first_active_round=args.profiler_wait + args.profiler_warmup,
                             performance_numbers_include_profiler_overhead=True)
        (self.output / "collection.json").write_text(json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8")
        self.profiler = api.profile(
            activities=[api.ProfilerActivity.CPU, api.ProfilerActivity.NPU],
            schedule=api.schedule(wait=args.profiler_wait, warmup=args.profiler_warmup,
                                  active=args.profiler_active, repeat=1),
            on_trace_ready=api.tensorboard_trace_handler(
                str(self.output), worker_name=self.worker, analyse_flag=False),
            record_shapes=False, profile_memory=False, with_stack=False,
            experimental_config=api._ExperimentalConfig(
                profiler_level=getattr(api.ProfilerLevel, args.profiler_level),
                aic_metrics=api.AiCMetrics.AiCoreNone,
                export_type=api.ExportType.Text, data_simplification=False),
        )
        self.profiler.__enter__()
        print(f"BATCH_GET_TORCH_NPU_PROFILER_BEGIN path={self.output} "
              f"wait={args.profiler_wait} warmup={args.profiler_warmup} active={args.profiler_active} "
              f"level={args.profiler_level} performance_numbers_include_profiler_overhead=1", flush=True)
        return self

    def range(self, phase, iteration):
        if self.profiler is None:
            return nullcontext()
        return self.torch.profiler.record_function(f"KVAF/{self.worker}/round{iteration}/{phase}")

    def step(self):
        if self.profiler is not None:
            self.profiler.step()

    def __exit__(self, error_type, error, traceback):
        if self.profiler is None:
            return
        self.profiler.__exit__(error_type, error, traceback)
        if error_type is not None:
            return  # Keep partial raw data; preserve the benchmark error.
        print(f"BATCH_GET_TORCH_NPU_PROFILER_ANALYSE path={self.output}", flush=True)
        # Bound parser CPU/memory use when eight workers finish together.
        self.analyse(str(self.output), max_process_number=1)
        traces = sorted(path for path in self.output.rglob("trace_view.json") if path.stat().st_size > 0)
        if not traces:
            raise RuntimeError(f"torch_npu.profiler produced no trace_view.json under {self.output}; "
                               "inspect raw data and profiler logs, and check the CANN/msprof installation")
        for path in traces:
            print(f"BATCH_GET_TORCH_NPU_TRACE_WRITTEN path={path}", flush=True)
