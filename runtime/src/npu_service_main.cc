#include "npu_runtime.h"
#include "runtime_service.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
  kvaf::RuntimeServiceOptions service_options;
  kvaf::NpuServiceOptions npu_options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if ((arg == "--listen" || arg == "--port" || arg == "--device-id" || arg == "--eid-index" ||
         arg == "--hbm-allocator") && i + 1 >= argc) {
      std::fprintf(stderr, "missing value for %s\n", arg.c_str());
      return 2;
    }
    if (arg == "--listen") service_options.address = argv[++i];
    else if (arg == "--port") service_options.port = static_cast<std::uint16_t>(std::strtoul(argv[++i], nullptr, 10));
    else if (arg == "--device-id") { npu_options.device_id = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10)); npu_options.has_device_id = true; }
    else if (arg == "--eid-index") { npu_options.eid_index = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10)); npu_options.has_eid_index = true; }
    else if (arg == "--hbm-allocator") {
      const std::string allocator(argv[++i]);
      if (allocator != "normal" && allocator != "huge") {
        std::fprintf(stderr, "--hbm-allocator must be normal or huge\n");
        return 2;
      }
      npu_options.hbm_huge_first = allocator == "huge";
    }
    else if (arg == "--help") { std::fprintf(stderr, "Usage: %s [--listen ADDRESS] [--port PORT] [--device-id N] [--eid-index N] [--hbm-allocator normal|huge (default: huge)]\n", argv[0]); return 0; }
    else { std::fprintf(stderr, "unknown argument: %s\n", arg.c_str()); return 2; }
  }
  kvaf::NpuRuntime service(npu_options);
  const kvaf::Status init = service.InitializeHbmService();
  if (!init.ok()) { std::fprintf(stderr, "[KVAF_NPU] %s\n", init.message().c_str()); return 1; }
  const kvaf::Status status = kvaf::RuntimeService(service_options).Serve(service);
  std::fprintf(stderr, "[KVAF_NPU] %s\n", status.message().c_str());
  return 1;
}
