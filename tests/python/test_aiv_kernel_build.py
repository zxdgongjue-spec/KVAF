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

from collect_aiv_scatter_debug import collect, compile_recipe

TOOL = r'''
import json
from pathlib import Path
import sys

args = sys.argv[1:]
if args == ['--version']:
    print('CANN tool double, not real hardware')
    sys.exit(0)
output_index = args.index('-o') + 1
output = Path(args[output_index])
if Path(sys.argv[0]).name == 'ccec':
    source = next(Path(a) for a in args if a.endswith('.asc'))
    if '-E' in args:
        if '-dM' in args:
            output.write_text('#define __CCE_AICORE__ 3101\n#define __NPU_ARCH__ 3510\n')
        else:
            header = Path(sys.argv[0]).parent / 'sdk/include/kernel_operator_common_impl.h'
            output.write_text(f'# 1 "{header}"\n' + header.read_text())
        sys.exit(0)
    if '-S' in args:
        if Path(sys.argv[0]).with_name('fail-assembly').exists():
            sys.exit('synthetic assembly failure')
        output.write_text('KvafSequentialBatchGetAivKernelV1:\n  model_instruction\n')
        sys.exit(0)
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
    with tempfile.TemporaryDirectory(prefix='kvaf kernel build-') as directory:
        root = Path(directory)
        shutil.copytree(demo / 'kernel', root / 'kernel')
        sdk = root / 'sdk'
        (sdk / 'include/simt_api').mkdir(parents=True)
        (sdk / 'include/kernel_operator.h').touch()
        (sdk / 'include/kernel_operator_common_impl.h').write_text(
            'inline void InitSocState() { /* tool double */ }\n')
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
        originals = [companion, build / 'kernel/kvaf_sequential_batch_get_kernels.o',
                     build / 'kernel/kvaf_sequential_batch_get.tmp.o']
        before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in originals]
        report, complete = collect(build, build / 'bin/kvaf_execute_batch_get_kernels.o')
        assert complete
        text = (report / 'report.txt').read_text()
        assert 'INIT IMPLEMENTATION:' in text and 'InitSocState' in text
        assert '__NPU_ARCH__ 3510' in text and 'model_instruction' in text
        manifest = json.loads((report / 'manifest.json').read_text())
        assert '--cce-aicore-arch=dav-c310-vec' in manifest['compile_command']
        assert '-O2' in manifest['compile_command']
        assert all(path.read_bytes() == data and path.stat().st_mtime_ns == stamp
                   for path, (data, stamp) in zip(originals, before))
        assert (report / 'original/configured/kvaf_sequential_batch_get_kernels.o').read_bytes() == before[0][0]
        (root / 'fail-assembly').touch()
        partial, complete = collect(build)
        assert not complete and partial != report
        text = (partial / 'report.txt').read_text()
        assert 'COLLECTION_PARTIAL' in text and 'synthetic assembly failure' in text
        assert all(path.read_bytes() == data and path.stat().st_mtime_ns == stamp
                   for path, (data, stamp) in zip(originals, before))
        # Never guess between multiple compile recipes.
        makefile = next(build.rglob('build.make'))
        recipe = next(line for line in makefile.read_text().splitlines()
                      if line.startswith('\t') and 'execute_sequential_batch_get_aiv.asc' in line)
        makefile.write_text(makefile.read_text() + '\n' + recipe + '\n')
        try:
            compile_recipe(build)
        except ValueError as error:
            assert 'found 2' in str(error)
        else:
            raise AssertionError('ambiguous compiler recipe was accepted')
    print('AIV kernel build packaging: PASS')


if __name__ == '__main__':
    main()
