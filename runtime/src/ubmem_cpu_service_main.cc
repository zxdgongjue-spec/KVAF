#include "ubmem_runtime.h"
#include "runtime_service.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
  kvaf::RuntimeServiceOptions options;
  options.port = 29000;
  int numa = -1;
  std::string library = "libascend_hal.so";
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      std::printf("Usage: %s [--listen ADDRESS] [--port PORT] [--numa-node N] [--hal-library PATH]\n", argv[0]);
      return 0;
    }
    if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); return 2; }
    const char* value = argv[++i];
    if (arg == "--listen") options.address = value;
    else if (arg == "--hal-library") library = value;
    else if (arg == "--port" || arg == "--numa-node") {
      char* end = nullptr;
      errno = 0;
      const long n = std::strtol(value, &end, 10);
      if (errno || end == value || *end || n < 0 ||
          (arg == "--port" ? (n == 0 || n > 65535) : n > 2147483647L)) {
        std::fprintf(stderr, "invalid %s\n", arg.c_str()); return 2;
      }
      if (arg == "--port") options.port = static_cast<std::uint16_t>(n);
      else numa = static_cast<int>(n);
    } else { std::fprintf(stderr, "unknown option %s\n", arg.c_str()); return 2; }
  }
  kvaf::ubmem::CpuRuntime runtime;
  auto status = runtime.Initialize(numa, library);
  if (status.ok()) status = kvaf::RuntimeService(options).Serve(runtime);
  std::fprintf(stderr, "[KVAF_UBMEM] %s\n", status.message().c_str());
  return 1;
}
