#include "npu_runtime.h"
#include "runtime_service.h"
#include <acl/acl.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
std::uint32_t Number(const std::string& text) {
  std::size_t end = 0U;
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("expected an unsigned integer: " + text);
  const auto value = std::stoull(text, &end);
  if (end != text.size() || value > UINT32_MAX) throw std::invalid_argument("integer out of range");
  return static_cast<std::uint32_t>(value);
}
}
int main(int argc, char** argv) {
  kvaf::NpuHbmServiceOptions npu;
  kvaf::RuntimeServiceOptions tcp;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--help") {
        std::puts("Usage: kvaf_npu_service [--listen ADDRESS] [--port PORT] [--device-id N] "
                  "[--npu-eid HEX (repeatable)] [--hbm-allocator huge|normal] "
                  "[--max-sessions N] [--max-channels N]");
        return 0;
      }
      if (i + 1 == argc) throw std::invalid_argument("missing value for " + arg);
      const std::string value = argv[++i];
      if (arg == "--listen") tcp.address = value;
      else if (arg == "--device-id") npu.device_id = Number(value);
      else if (arg == "--port") {
        const auto port = Number(value);
        if (port == 0U || port > 65535U) throw std::invalid_argument("port must be in [1,65535]");
        tcp.port = static_cast<std::uint16_t>(port);
      } else if (arg == "--npu-eid") npu.npu_eids.push_back(value);
      else if (arg == "--max-sessions") npu.max_sessions = Number(value);
      else if (arg == "--max-channels") npu.max_channels_per_session = Number(value);
      else if (arg == "--hbm-allocator") {
        if (value != "normal" && value != "huge") throw std::invalid_argument("allocator must be huge or normal");
        npu.huge_first = value == "huge";
      } else throw std::invalid_argument("unknown argument: " + arg);
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what()); return 2;
  }
  if (aclInit(nullptr) != ACL_SUCCESS) {
    std::fprintf(stderr, "aclInit failed\n"); return 1;
  }
  int result = 1;
  {
    kvaf::NpuRuntime runtime(npu);
    kvaf::Status status = runtime.InitializeHbmService();
    if (status.ok()) status = kvaf::RuntimeService(tcp).Serve(runtime);
    if (!status.ok()) std::fprintf(stderr, "[KVAF_NPU] %s\n", status.message().c_str());
    result = status.ok() ? 0 : 1;
  }
  (void)aclrtResetDevice(npu.device_id);
  (void)aclFinalize();
  return result;
}
