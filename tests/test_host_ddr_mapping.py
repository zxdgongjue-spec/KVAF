#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).parents[1]
MAPPING_SOURCE = ROOT / "access_operator" / "a5" / "src" / "host_ddr_mapping.cpp"


FAKE_HAL_SOURCE = r"""
#include <stdint.h>
#include <stdlib.h>

static int g_fail_unregister;
static int g_fail_free;
static int g_alloc_failure_with_pointer;
static int g_register_failure_mode;
static int g_alloc_calls;
static int g_free_calls;
static int g_register_calls;
static int g_unregister_calls;

void kvaf_fake_reset(void) {
  g_fail_unregister = 0;
  g_fail_free = 0;
  g_alloc_failure_with_pointer = 0;
  g_register_failure_mode = 0;
  g_alloc_calls = 0;
  g_free_calls = 0;
  g_register_calls = 0;
  g_unregister_calls = 0;
}

void kvaf_fake_set_fail_unregister(int value) { g_fail_unregister = value; }
void kvaf_fake_set_fail_free(int value) { g_fail_free = value; }
void kvaf_fake_set_alloc_failure_with_pointer(int value) {
  g_alloc_failure_with_pointer = value;
}
void kvaf_fake_set_register_failure_mode(int value) {
  g_register_failure_mode = value;
}
int kvaf_fake_alloc_calls(void) { return g_alloc_calls; }
int kvaf_fake_free_calls(void) { return g_free_calls; }
int kvaf_fake_register_calls(void) { return g_register_calls; }
int kvaf_fake_unregister_calls(void) { return g_unregister_calls; }

int halMemAlloc(void **address, uint64_t bytes, uint64_t flags) {
  (void)flags;
  ++g_alloc_calls;
  *address = malloc((size_t)bytes);
  return g_alloc_failure_with_pointer ? 17 : 0;
}

int halMemFree(void *address) {
  ++g_free_calls;
  if (g_fail_free) {
    return 23;
  }
  free(address);
  return 0;
}

int halHostRegister(void *host_address, uint64_t bytes, uint32_t map_type,
                    uint32_t device_id, void **device_address) {
  (void)bytes;
  (void)map_type;
  (void)device_id;
  ++g_register_calls;
  if (g_register_failure_mode == 2) {
    *device_address = NULL;
    return 31;
  }
  *device_address = (void *)((uintptr_t)host_address + 0x1000U);
  return g_register_failure_mode == 1 ? 31 : 0;
}

int halHostUnregisterEx(void *host_address, uint32_t device_id,
                        uint32_t map_type) {
  (void)host_address;
  (void)device_id;
  (void)map_type;
  ++g_unregister_calls;
  return g_fail_unregister ? 41 : 0;
}
"""


HARNESS_SOURCE = r"""
#include "kvaf/access_operator/a5/host_ddr_mapping.h"

#include <dlfcn.h>

#include <cstdint>
#include <iostream>
#include <string>

using kvaf::Address;
using kvaf::StatusCode;
using kvaf::access::a5::HostDdrDeviceMapping;

namespace {

template <typename Function>
Function Symbol(void* library, const char* name) {
  auto* symbol = dlsym(library, name);
  if (symbol == nullptr) {
    std::cerr << "missing fake HAL symbol: " << name << '\n';
    std::exit(2);
  }
  return reinterpret_cast<Function>(symbol);
}

void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "CHECK FAILED: " << message << '\n';
    std::exit(3);
  }
}

}  // namespace

int main() {
  void* library = dlopen("libascend_hal.so", RTLD_NOW | RTLD_LOCAL);
  Check(library != nullptr, dlerror());

  const auto reset = Symbol<void (*)()>(library, "kvaf_fake_reset");
  const auto set_fail_unregister =
      Symbol<void (*)(int)>(library, "kvaf_fake_set_fail_unregister");
  const auto set_fail_free =
      Symbol<void (*)(int)>(library, "kvaf_fake_set_fail_free");
  const auto set_alloc_failure = Symbol<void (*)(int)>(
      library, "kvaf_fake_set_alloc_failure_with_pointer");
  const auto set_register_failure = Symbol<void (*)(int)>(
      library, "kvaf_fake_set_register_failure_mode");
  const auto alloc_calls =
      Symbol<int (*)()>(library, "kvaf_fake_alloc_calls");
  const auto free_calls =
      Symbol<int (*)()>(library, "kvaf_fake_free_calls");
  const auto register_calls =
      Symbol<int (*)()>(library, "kvaf_fake_register_calls");
  const auto unregister_calls =
      Symbol<int (*)()>(library, "kvaf_fake_unregister_calls");

  // A failed unregister retains the original allocation and mapping.  A new
  // Register must not overwrite that state and accidentally free a borrowed
  // address when Close is retried.
  reset();
  {
    HostDdrDeviceMapping mapping;
    Check(mapping.AllocateAndRegister(4096, 2).ok(), "initial allocation");
    const Address original_host = mapping.host_address();
    const Address original_device = mapping.device_address();
    set_fail_unregister(1);
    Check(!mapping.Close().ok(), "unregister failure must be visible");
    Check(mapping.host_address() == original_host,
          "failed Close must retain host allocation");
    Check(mapping.device_address() == original_device,
          "failed Close must retain device mapping");
    const auto rejected = mapping.Register(0x12345000U, 1024, 3);
    Check(!rejected.ok() && rejected.code() == StatusCode::kInvalidArgument,
          "new Register must reject retained state");
    Check(register_calls() == 1, "rejected Register must not reach HAL");
    Check(alloc_calls() == 1, "rejected Register must not allocate");

    set_fail_unregister(0);
    Check(mapping.Close().ok(), "unregister retry");
    Check(unregister_calls() == 2, "unregister must be retried");
    Check(free_calls() == 1, "original owned allocation must be freed once");
    Check(mapping.host_address() == 0 && mapping.device_address() == 0 &&
              mapping.bytes() == 0,
          "successful Close must clear public addresses");

    Check(mapping.Register(0x12345000U, 1024, 3).ok(),
          "object must be reusable after successful Close");
    Check(mapping.Close().ok(), "borrowed mapping close");
    Check(free_calls() == 1, "borrowed address must never be freed");
  }

  // Some HAL failures still return an allocation.  If freeing that partial
  // result fails, preserve it and reject all new work until Close succeeds.
  reset();
  {
    HostDdrDeviceMapping mapping;
    set_alloc_failure(1);
    set_fail_free(1);
    const auto allocation_failure = mapping.AllocateAndRegister(4096, 1);
    Check(!allocation_failure.ok(), "allocation failure must be visible");
    Check(allocation_failure.message().find("retained for Close retry") !=
              std::string::npos,
          "partial cleanup failure must be reported");
    Check(mapping.host_address() != 0,
          "failed free must retain the returned allocation");
    Check(alloc_calls() == 1 && free_calls() == 1,
          "partial allocation cleanup must be attempted once");
    Check(!mapping.AllocateAndRegister(8192, 1).ok(),
          "new allocation must reject retained partial state");
    Check(alloc_calls() == 1,
          "rejected allocation must not overwrite retained address");

    set_fail_free(0);
    Check(mapping.Close().ok(), "partial allocation free retry");
    Check(free_calls() == 2, "free must be retried");
  }

  // An error return carrying a device address may represent a live mapping.
  // Keep the borrowed address until unregister is positively confirmed.
  reset();
  {
    HostDdrDeviceMapping mapping;
    set_register_failure(1);
    set_fail_unregister(1);
    const auto registration_failure =
        mapping.Register(0x45678000U, 2048, 4);
    Check(!registration_failure.ok(), "registration failure must be visible");
    Check(mapping.host_address() == 0x45678000U &&
              mapping.device_address() != 0,
          "possible mapping must remain available for Close retry");
    Check(!mapping.AllocateAndRegister(4096, 4).ok(),
          "retained borrowed mapping must block owned allocation");
    Check(alloc_calls() == 0, "blocked allocation must not reach HAL");

    set_fail_unregister(0);
    Check(mapping.Close().ok(), "possible mapping unregister retry");
    Check(free_calls() == 0, "borrowed mapping must not be freed");
  }

  // A clean registration failure without a device address leaves no retained
  // mapping and the same object can be used again.
  reset();
  {
    HostDdrDeviceMapping mapping;
    set_register_failure(2);
    Check(!mapping.Register(0x789ab000U, 1024, 5).ok(),
          "clean registration failure");
    set_register_failure(0);
    Check(mapping.Register(0x789ab000U, 1024, 5).ok(),
          "reuse after clean registration failure");
    Check(mapping.Close().ok(), "final close");
    Check(mapping.Close().ok(), "Close must be idempotent");
  }

  Check(dlclose(library) == 0, "close fake HAL control handle");
  std::cout << "HOST_DDR_MAPPING_TEST_OK\n";
  return 0;
}
"""


class HostDdrMappingTests(unittest.TestCase):
    def test_retained_state_and_ownership(self) -> None:
        cc = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
        cxx = (
            shutil.which("g++")
            or shutil.which("clang++")
            or shutil.which("c++")
        )
        if cc is None or cxx is None:
            self.skipTest("host C/C++ compiler is unavailable")

        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            fake_source = build / "fake_ascend_hal.c"
            harness_source = build / "host_ddr_mapping_test.cpp"
            fake_library = build / "libascend_hal.so"
            executable = build / "host_ddr_mapping_test"
            fake_source.write_text(
                textwrap.dedent(FAKE_HAL_SOURCE), encoding="utf-8"
            )
            harness_source.write_text(
                textwrap.dedent(HARNESS_SOURCE), encoding="utf-8"
            )

            if sys.platform == "darwin":
                library_command = [
                    cc,
                    "-dynamiclib",
                    "-fPIC",
                    str(fake_source),
                    "-o",
                    str(fake_library),
                ]
                dynamic_path_variable = "DYLD_LIBRARY_PATH"
                dl_library: list[str] = []
            else:
                library_command = [
                    cc,
                    "-shared",
                    "-fPIC",
                    str(fake_source),
                    "-o",
                    str(fake_library),
                ]
                dynamic_path_variable = "LD_LIBRARY_PATH"
                dl_library = ["-ldl"]

            subprocess.run(library_command, check=True, capture_output=True)
            subprocess.run(
                [
                    cxx,
                    "-std=c++17",
                    "-I",
                    str(ROOT / "include"),
                    "-I",
                    str(ROOT / "access_operator" / "a5" / "include"),
                    str(MAPPING_SOURCE),
                    str(harness_source),
                    *dl_library,
                    "-o",
                    str(executable),
                ],
                check=True,
                capture_output=True,
            )

            environment = os.environ.copy()
            old_path = environment.get(dynamic_path_variable)
            environment[dynamic_path_variable] = (
                str(build)
                if not old_path
                else f"{build}{os.pathsep}{old_path}"
            )
            result = subprocess.run(
                [str(executable)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                cwd=build,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "HOST_DDR_MAPPING_TEST_OK\n")


if __name__ == "__main__":
    unittest.main()
