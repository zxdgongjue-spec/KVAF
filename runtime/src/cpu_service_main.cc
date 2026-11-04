#include "cpu_runtime.h"
#include "runtime_service.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

namespace {

void PrintUsage(const char* program) {
  std::fprintf(stderr,
               "Usage: %s [--listen ADDRESS] [--port PORT] [--ub-device NAME] [--eid-index N] [--gather-threads N] [--gather-mode adaptive|active|passive] [--gather-affinity on|off] [--memory-page-mode default|thp]\n"
               "Default listener is 127.0.0.1:28999. Without --ub-device/--eid-index, all UB EID "
               "candidates are discovered automatically. Gather threads default to auto (up to 32), "
               "mode to adaptive (one polling coordinator, parked helpers), affinity to on. "
               "Default memory page mode is thp. The service requires only liburma and /dev/uburma.\n",
               program);
}

bool ParseUnsigned(const char* text, std::uint32_t maximum, std::uint32_t* value) {
  if (text == nullptr || value == nullptr || text[0] == '\0') return false;
  char* end = nullptr;
  errno = 0;
  const unsigned long parsed = std::strtoul(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed > maximum) return false;
  *value = static_cast<std::uint32_t>(parsed);
  return true;
}

bool ParseMemoryPageMode(const std::string& text, kvaf::CpuMemoryPageMode* mode) {
  if (mode == nullptr) return false;
  if (text == "default") {
    *mode = kvaf::CpuMemoryPageMode::kDefault;
    return true;
  }
  if (text == "thp") {
    *mode = kvaf::CpuMemoryPageMode::kTransparentHugePage;
    return true;
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  kvaf::RuntimeServiceOptions service_options;
  kvaf::CpuRuntimeOptions cpu_options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--help") {
      PrintUsage(argv[0]);
      return 0;
    }
    if ((argument == "--listen" || argument == "--port" || argument == "--ub-device" ||
         argument == "--eid-index" || argument == "--gather-threads" ||
         argument == "--gather-mode" || argument == "--gather-affinity" || argument == "--memory-page-mode") &&
        index + 1 >= argc) {
      PrintUsage(argv[0]);
      return 2;
    }
    if (argument == "--listen") {
      service_options.address = argv[++index];
    } else if (argument == "--port") {
      std::uint32_t parsed = 0U;
      if (!ParseUnsigned(argv[++index], std::numeric_limits<std::uint16_t>::max(), &parsed) ||
          parsed == 0U) {
        std::fprintf(stderr, "invalid --port\n");
        return 2;
      }
      service_options.port = static_cast<std::uint16_t>(parsed);
    } else if (argument == "--ub-device") {
      cpu_options.ub_device = argv[++index];
    } else if (argument == "--eid-index") {
      if (!ParseUnsigned(argv[++index], std::numeric_limits<std::uint32_t>::max(),
                         &cpu_options.eid_index)) {
        std::fprintf(stderr, "invalid --eid-index\n");
        return 2;
      }
      cpu_options.has_eid_index = true;
    } else if (argument == "--gather-threads") {
      if (!ParseUnsigned(argv[++index], 64U, &cpu_options.gather_threads) ||
          cpu_options.gather_threads == 0U) {
        std::fprintf(stderr, "invalid --gather-threads (expected 1..64)\n");
        return 2;
      }
    } else if (argument == "--gather-affinity") {
      const std::string value(argv[++index]);
      if (value != "on" && value != "off") {
        std::fprintf(stderr, "invalid --gather-affinity (expected on|off)\n");
        return 2;
      }
      cpu_options.gather_affinity = value == "on";
    } else if (argument == "--gather-mode") {
      const std::string value(argv[++index]);
      if (value != "adaptive" && value != "active" && value != "passive") {
        std::fprintf(stderr, "invalid --gather-mode (expected adaptive|active|passive)\n");
        return 2;
      }
      cpu_options.gather_mode = value == "adaptive" ? kvaf::CpuGatherMode::kAdaptive
          : value == "active" ? kvaf::CpuGatherMode::kActive
                              : kvaf::CpuGatherMode::kPassive;

    } else if (argument == "--memory-page-mode") {
      if (!ParseMemoryPageMode(argv[++index], &cpu_options.memory_page_mode)) {
        std::fprintf(stderr, "invalid --memory-page-mode; expected default or thp\n");
        return 2;
      }
    } else {
      PrintUsage(argv[0]);
      return 2;
    }
  }

  kvaf::CpuRuntime runtime;
  const kvaf::Status initialize = runtime.Initialize(cpu_options);
  if (!initialize.ok()) {
    std::fprintf(stderr, "[KVAF_CPU] %s\n", initialize.message().c_str());
    return 1;
  }
  const kvaf::Status serve = kvaf::RuntimeService(service_options).Serve(runtime);
  std::fprintf(stderr, "[KVAF_CPU] %s\n", serve.message().c_str());
  return 1;
}
