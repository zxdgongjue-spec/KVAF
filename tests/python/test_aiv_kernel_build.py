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
from unittest.mock import patch

from collect_aiv_scatter_debug import analyze_saved, collect, compile_recipe, init_definitions

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
            output.write_text('#define __CCE_AICORE__ 310\n#define __NPU_ARCH__ 3510\n'
                              '#define ASCEND_HEADER_GUARD\n')
        else:
            sdk = Path(sys.argv[0]).parent / 'sdk/include'
            names = ['kernel_operator_common_impl.h', 'kernel_utils_example.h']
            if Path(sys.argv[0]).with_name('wrapper-only').exists():
                names.pop()
            output.write_text(''.join(f'# 1 "{sdk / name}"\n' + (sdk / name).read_text()
                                      for name in names))
        sys.exit(0)
    if '-S' in args:
        sys.exit("error: unsupported option '-S' on device side")
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
            'inline void InitSocState();\n'
            'inline void InitSocState() { AscendCUtils::InitSocStateImpl(); }\n')
        (sdk / 'include/kernel_utils_example.h').write_text(
            'struct AscendCUtils {\n'
            'static inline void InitSocStateImpl() {\n'
            '  if (true) { reset_state(); } /* nested braces */\n'
            '}\n};\n')
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
        assert 'DEFINITION: InitSocStateImpl' in text and 'reset_state();' in text
        assert '__NPU_ARCH__ 3510' in text and 'ASCEND_HEADER_GUARD' not in text
        assert 'kernel_utils_example.h:2' in text
        assert list((report / 'headers').glob('*_kernel_utils_example.h'))
        manifest = json.loads((report / 'manifest.json').read_text())
        assert '--cce-aicore-arch=dav-c310-vec' in manifest['compile_command']
        assert '-O2' in manifest['compile_command']
        assert 'assembly' not in manifest['steps']
        assert all(path.read_bytes() == data and path.stat().st_mtime_ns == stamp
                   for path, (data, stamp) in zip(originals, before))
        assert (report / 'original/configured/kvaf_sequential_batch_get_kernels.o').read_bytes() == before[0][0]
        # Wrapper/declaration presence alone must not claim complete evidence.
        (root / 'wrapper-only').touch()
        partial, complete = collect(build)
        assert not complete and partial != report
        text = (partial / 'report.txt').read_text()
        assert 'COLLECTION_PARTIAL' in text and 'InitSocStateImpl DEFINITION NOT FOUND' in text
        assert all(path.read_bytes() == data and path.stat().st_mtime_ns == stamp
                   for path, (data, stamp) in zip(originals, before))
        # Re-analyze the original .ii without invoking tools or reading changed
        # SDK headers. Keep the old report, binary snapshots and manifest intact.
        (sdk / 'include/kernel_utils_example.h').write_text('changed SDK\n')
        saved = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in report.rglob('*') if p.is_file()}
        with patch('collect_aiv_scatter_debug.subprocess.run', side_effect=AssertionError('compiler invoked')):
            analysis, complete = analyze_saved(report)
            assert complete and 'reset_state();' in analysis.read_text()
            assert 'changed SDK' not in analysis.read_text()
            other, complete = analyze_saved(report)
            assert complete and other != analysis
            _, complete = analyze_saved(partial)
            assert not complete
        assert all(p.read_bytes() == data and p.stat().st_mtime_ns == stamp
                   for p, (data, stamp) in saved.items())
        # A previous collector's failed assembly step remains visible, while
        # extracting existing source requires neither build.make nor a compiler.
        manifest['steps']['assembly'] = 1
        (report / 'manifest.json').write_text(json.dumps(manifest))
        run([sys.executable, Path(__file__).with_name('collect_aiv_scatter_debug.py'), '--analyze', report])
        assert all(p.read_bytes() == data and p.stat().st_mtime_ns == stamp
                   for p, (data, stamp) in zip(originals, before))
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
    # Ignore declarations/calls and braces in comments or string literals.
    definitions = init_definitions('# 40 "utils.h"\nvoid InitSocStateImpl();\n'
                                   'void Other() { InitSocStateImpl(); }\n'
                                   'void InitSocStateImpl() {\n'
                                   '  const char *s = "}"; /* { */\n'
                                   '  if (true) { reset_state(); }\n}\n')
    assert len(definitions) == 1 and definitions[0]['line'] == 42
    assert definitions[0]['text'].endswith('reset_state(); }\n}')
    print('AIV kernel build packaging and saved compiler evidence: PASS')


if __name__ == '__main__':
    main()
