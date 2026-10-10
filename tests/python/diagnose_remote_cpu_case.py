#!/usr/bin/env python3
"""One fresh-process diagnostic case; invoked by diagnose_remote_cpu.py."""
import os
import faulthandler
from pathlib import Path
import traceback


def loaded_libraries():
    try:
        paths = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if '.so' in line and any(s in line.lower() for s in
                            ('ascend', 'libra.', 'hccp', 'urma', 'udma', 'ummu', 'kvaf', 'runtime'))})
        for path in paths:
            print(f'DIAG_LOADED_LIBRARY path={path}', flush=True)
    except OSError as exc:
        print(f'DIAG_MAPS_UNAVAILABLE {exc}', flush=True)


def main():
    faulthandler.enable(all_threads=True)
    import torch
    import torch_npu  # noqa: F401: registers the NPU backend
    import kvaf
    print(f'DIAG_PACKAGES torch={torch.__version__} torch_npu={torch_npu.__version__} '
          f'kvaf={kvaf.__file__}', flush=True)
    device_id = int(os.environ.get('DEVICE_ID', '0'))
    torch.npu.set_device(device_id)
    device = f'npu:{device_id}'
    runtime = kvaf.NpuRuntime(
        os.environ['KVAF_BATCH_GET_KERNEL'], batch_get_token_bytes=64,
        batch_get_tokens_per_block=1, device_id=device_id,
        npu_core_count=int(os.environ['KVAF_DIAG_CORES']),
        batch_put_kernel_object_path=os.environ['KVAF_BATCH_PUT_KERNEL'],
        jettys_per_core=int(os.environ['KVAF_DIAG_JETTYS']))
    table = None
    phase = 'link'
    success = False
    cleanup_ok = False
    try:
        runtime.link_to(os.environ['KVAF_CPU_SERVICE_ADDR'])
        loaded_libraries()
        phase = 'initialize_probe'
        print(f'DIAG_PHASE_BEGIN phase={phase} ddr_bytes=262144 slot_size=64', flush=True)
        runtime.initialize(4096 * 64, 64)
        print(f'DIAG_PHASE_PASS phase={phase}', flush=True)
        # Exercise torch-owned HBM too, after the aclrtMalloc-owned startup probe.
        rows = int(os.environ['KVAF_DIAG_CORES']) * (32 + int(os.environ['KVAF_DIAG_JETTYS']) - 1)
        keys = list(range(1, rows + 1))
        source_cpu = (torch.arange(rows * 64, dtype=torch.int64).reshape(rows, 64) * 17
                      + torch.arange(rows, dtype=torch.int64).reshape(rows, 1) * 131).remainder(256).to(torch.uint8)
        source = source_cpu.to(device).contiguous()
        torch.npu.synchronize()
        phase = 'torch_hbm_put'
        print(f'DIAG_PHASE_BEGIN phase={phase} rows={rows}', flush=True)
        runtime.batch_put(keys, source, [i * 64 for i in range(rows)], [64] * rows)
        print(f'DIAG_PHASE_PASS phase={phase}', flush=True)
        phase = 'readback'
        print(f'DIAG_PHASE_BEGIN phase={phase}', flush=True)
        table = runtime.prepare_access_table(keys)
        device_table = runtime.access_table_view(table, device='npu')
        indices = torch.arange(rows, dtype=torch.int32, device=device)
        offsets = torch.zeros(rows, dtype=torch.int32, device=device)
        miss_count = torch.tensor([rows], dtype=torch.int32, device=device)
        output = torch.zeros((rows, 64), dtype=torch.uint8, device=device)
        runtime.batch_get(device_table, indices, offsets, miss_count, indices, output, synchronize=True)
        if not torch.equal(output.cpu(), source_cpu):
            raise AssertionError('CPU DDR readback differs from source')
        print(f'DIAG_PHASE_PASS phase={phase}', flush=True)
        runtime.release_access_table(table)
        table = None
        runtime.batch_free(keys)
        success = True
    except Exception:
        print(f'DIAG_FAILURE phase={phase}', flush=True)
        traceback.print_exc()
    finally:
        # Record actual mappings while the RA DSOs are still present.
        loaded_libraries()
        try:
            torch.npu.synchronize()
            if table is not None and runtime.initialized:
                runtime.release_access_table(table)
            runtime.shutdown()
            cleanup_ok = True
            print('DIAG_CLEANUP_OK', flush=True)
        except Exception:
            print('DIAG_CLEANUP_FAILED', flush=True)
            traceback.print_exc()
    print(f'DIAG_CASE_RESULT success={int(success)} cleanup_ok={int(cleanup_ok)}', flush=True)
    return 0 if success and cleanup_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
