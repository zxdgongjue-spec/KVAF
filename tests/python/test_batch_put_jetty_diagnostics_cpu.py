"""Optional public RA Jetty query against a real dynamically loaded mock.

This checks query ABI, optional-symbol behavior, and failure isolation on CPU;
it makes no claim about the target device's Jetty state or CQ completion.
"""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def query_program(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    root = tmp_path_factory.mktemp("put-jetty-query")
    source = root / "query.cc"
    source.write_text(r'''
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include "direct_hccp/direct_hccp_jetty_diagnostics.h"
int main(int argc, char** argv) {
  assert(argc == 3);
  void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  assert(library != nullptr);
  const unsigned int mode = static_cast<unsigned int>(std::strtoul(argv[2], nullptr, 10));
  errno = EDOM;
  kvaf::direct_hccp::LogBatchPutJettyState(
      mode == 7U ? nullptr : library,
      mode == 6U ? nullptr : reinterpret_cast<void*>(static_cast<std::uintptr_t>(mode)),
      "device", 3U, 1U);
  assert(errno == EDOM);
  dlclose(library);
}
''')
    mock = root / "mock.cc"
    mock.write_text(r'''
#include <cassert>
#include <cerrno>
#include <cstdint>
struct JettyAttr {
  std::uint32_t mask;
  std::uint32_t rx_threshold;
  std::int32_t state;
  std::uint32_t reserved[2];
};
#ifndef NO_QUERY
extern "C" int RaCtxQpQueryBatch(void* handles[], JettyAttr attributes[], unsigned int* count) {
  assert(handles != nullptr && attributes != nullptr && count != nullptr && *count == 1U);
  assert(attributes[0].mask == 0U && attributes[0].state == 0);
  const auto mode = reinterpret_cast<std::uintptr_t>(handles[0]);
  errno = EIO;
  if (mode == 4U) return -17;
  if (mode == 5U) { *count = 0U; return 0; }
  attributes[0].mask = 2U;
  attributes[0].state = mode == 8U ? 42 : mode == 9U ? 3 : static_cast<std::int32_t>(mode - 1U);
  return 0;
}
#else
extern "C" int AnUnrelatedSymbol() { return 0; }
#endif
''')
    executable = root / "query"
    flags = [] if sys.platform == "darwin" else ["-ldl"]
    subprocess.run([
        compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
        "-I", str(ROOT / "access_operator/include"), str(source),
        "-o", str(executable), *flags,
    ], check=True, capture_output=True, text=True)
    libraries = {}
    for name, definitions in (("present", []), ("absent", ["-DNO_QUERY"])):
        library = root / f"{name}.so"
        subprocess.run([
            compiler, "-std=c++17", "-shared", "-fPIC", *definitions,
            str(mock), "-o", str(library),
        ], check=True, capture_output=True, text=True)
        libraries[name] = library
    return executable, libraries


@pytest.mark.parametrize("mode,expected", [
    (1, "mask=0x2 state=0 state_name=reset"),
    (2, "mask=0x2 state=1 state_name=ready"),
    (3, "mask=0x2 state=2 state_name=suspended"),
    (4, "api_ret=-17 returned=1 state_name=unavailable"),
    (5, "api_ret=0 returned=0 state_name=unavailable"),
    (6, "availability=missing_handle api_ret=not_called"),
    (7, "availability=symbol_unavailable api_ret=not_called"),
    (8, "mask=0x2 state=42 state_name=unknown"),
    (9, "mask=0x2 state=3 state_name=error"),
])
def test_query_result_and_failure_isolation(query_program, mode, expected):
    executable, libraries = query_program
    result = subprocess.run([str(executable), str(libraries["present"]), str(mode)],
                            check=True, capture_output=True, text=True)
    assert result.stdout == ""
    assert result.stderr.count("BATCH_PUT_JETTY_STATE") == 1
    assert "core=3 jetty=1 role=device api=RaCtxQpQueryBatch" in result.stderr
    assert expected in result.stderr


def test_missing_optional_symbol_is_nonfatal(query_program):
    executable, libraries = query_program
    result = subprocess.run([str(executable), str(libraries["absent"]), "2"],
                            check=True, capture_output=True, text=True)
    assert "availability=symbol_unavailable api_ret=not_called" in result.stderr
    assert "state_name=ready" not in result.stderr
