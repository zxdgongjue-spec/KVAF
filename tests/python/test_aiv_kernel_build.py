#!/usr/bin/env python3
"""Exercise the real kernel CMake graph with a CANN SDK-symbol tool double.

Each compiled module defines g_tilingKey; the linker rejects duplicate globals.
This verifies packaging/dependencies only, not CANN code generation.
"""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile


TOOL = r'''
import json
from pathlib import Path
import sys

args = sys.argv[1:]
output_index = args.index('-o') + 1
output = Path(args[output_index])
if Path(sys.argv[0]).name == 'ccec':
    source = next(Path(a) for a in args if a.endswith('.asc'))
    symbols = ['g_tilingKey', source.stem]
else:
    symbols = []
    for i, arg in enumerate(args):
        if i != output_index and arg.endswith('.o'):
            for symbol in json.loads(Path(arg).read_text()):
                if symbol in symbols:
                    sys.exit('duplicate symbol: ' + symbol)
                symbols.append(symbol)
output.write_text(json.dumps(symbols))
'''


def run(args):
    result = subprocess.run([str(a) for a in args], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


def main():
    cmake = sys.argv[1]
    repo = Path(__file__).resolve().parents[2]
    demo = repo / 'access_operator/demo/aiv_batch_get'
    with tempfile.TemporaryDirectory(prefix='kvaf-kernel-build-') as directory:
        root = Path(directory)
        shutil.copytree(demo / 'kernel', root / 'kernel')
        sdk = root / 'sdk'
        (sdk / 'include/simt_api').mkdir(parents=True)
        (sdk / 'include/kernel_operator.h').touch()
        (sdk / 'include/simt_api/device_atomic_functions.h').touch()
        for name in ('ccec', 'ld.lld'):
            tool = root / name
            tool.write_text(f'#!{sys.executable}\n' + TOOL)
            tool.chmod(0o755)
        (root / 'CMakeLists.txt').write_text(f'''
cmake_minimum_required(VERSION 3.18)
project(kernel_build_check NONE)
set(ASCEND_CANN_PACKAGE_PATH "{sdk.as_posix()}")
set(KVAF_KERNEL_OPERATOR_DIR "{sdk.as_posix()}/include")
set(KVAF_SIMT_API_DIR "{sdk.as_posix()}/include")
set(KVAF_AIV_DEMO_INCLUDE_DIR "{demo.as_posix()}/include")
set(KVAF_AIV_DEMO_BIN_DIR "${{CMAKE_BINARY_DIR}}/bin")
set(KVAF_CCEC "{root.as_posix()}/ccec")
set(KVAF_LLD "{root.as_posix()}/ld.lld")
add_subdirectory(kernel)
''')
        build = root / 'build'
        run([cmake, '-S', root, '-B', build])
        command = [cmake, '--build', build, '--target', 'kvaf_execute_batch_get_kernel', '-j4']
        run(command)
        for stem, symbol in (('kvaf_execute_batch_get', 'execute_batch_get_aiv'),
                             ('kvaf_sequential_batch_get', 'execute_sequential_batch_get_aiv')):
            product = build / 'kernel' / f'{stem}_kernels.o'
            installed = build / 'bin' / product.name
            assert product.read_bytes() == installed.read_bytes()
            assert json.loads(product.read_text()) == ['g_tilingKey', symbol]
        direct = build / 'kernel/kvaf_execute_batch_get_kernels.o'
        old_mtime = direct.stat().st_mtime_ns
        # A missing deployed companion must be restored by an incremental build.
        companion = build / 'bin/kvaf_sequential_batch_get_kernels.o'
        companion.unlink()
        run(command)
        assert companion.is_file()
        assert direct.stat().st_mtime_ns == old_mtime
    print('AIV kernel build packaging: PASS')


if __name__ == '__main__':
    main()
