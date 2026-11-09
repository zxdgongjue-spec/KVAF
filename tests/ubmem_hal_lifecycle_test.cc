#include "cpu_runtime.h"
#include "execution_cpu_backend.h"

#include <dlfcn.h>
#include <sys/socket.h>
#include <unistd.h>
#include <array>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string>
#include <vector>

#define CHECK(c) do { if (!(c)) { \
  std::cerr << "line " << __LINE__ << ": " #c "\n"; std::exit(1); \
} } while (false)

using namespace kvaf;

void CheckReleased(const std::string& path, unsigned allocations) {
  // Inspect only after pool/backend destruction. Holding a separate dlopen
  // reference during destruction would mask the original premature unload.
  void* library = dlopen(path.c_str(), RTLD_NOW | RTLD_NOLOAD);
  CHECK(library);
  auto snapshot = reinterpret_cast<void (*)(unsigned*)>(dlsym(library, "kvafTestHalCalls"));
  CHECK(snapshot);
  std::array<unsigned, 7> calls{};
  snapshot(calls.data());
  CHECK(dlclose(library) == 0);
  for (std::size_t i = 0; i < 6; ++i) CHECK(calls[i] == allocations);
  CHECK(calls[6] == 1);
}

void ExecutionPools(const CpuRuntimeOptions& options) {
  // Repeat with no live pool between allocations, then recreate the service.
  for (int service_cycle = 0; service_cycle < 2; ++service_cycle) {
    auto service = CreateExecutionService(options);
    for (int pool_cycle = 0; pool_cycle < 3; ++pool_cycle) {
      execution::Request request;
      request.op = execution::Op::CreateRegion;
      request.client = 1;
      request.request = pool_cycle + 1;
      request.epoch = service->Handle(execution::Request{}).epoch;
      request.mode = TransportMode::Ubmem;
      request.logical_bytes = 4096;
      request.auxiliary_bytes = 64;
      request.slot_size = 64;
      const auto response = service->Handle(request);
      if (!response.status.ok()) std::cerr << response.status.message() << '\n';
      CHECK(response.status.ok());
      CHECK(response.binding.fabric_mapping_bytes == 2U * 1024 * 1024);
      request.op = execution::Op::ReleaseRegion;
      request.region = response.region;
      CHECK(service->Handle(request).status.ok());
      CHECK(service->Handle(request).status.ok());  // No double release.
    }
    CHECK(service->Shutdown().ok());
  }
}

void LegacyPools(const CpuRuntimeOptions& options) {
  for (int cycle = 0; cycle < 3; ++cycle) {
    CpuRuntime runtime;
    CHECK(runtime.Initialize(options).ok());
    int sockets[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    peer::AllocateUbmemRegionRequest request{};
    request.header = peer::MakeHeader(peer::Command::kAllocateUbmemRegion, sizeof(request));
    request.ddr_bytes = 4096;
    request.slot_size = 64;
    CHECK(send(sockets[0], &request, sizeof(request), 0) == sizeof(request));
    const auto status = runtime.ServeConnection(sockets[1]);
    if (!status.ok()) std::cerr << status.message() << '\n';
    CHECK(status.ok());
    peer::AllocateUbmemRegionResponse response{};
    CHECK(recv(sockets[0], &response, sizeof(response), MSG_WAITALL) == sizeof(response));
    CHECK(response.result == static_cast<int>(peer::Result::kOk));
    CHECK(runtime.ReleaseLink(response.session_id).ok());
    CHECK(runtime.Shutdown().ok());
    CHECK(runtime.Shutdown().ok());
    close(sockets[0]);
    close(sockets[1]);
  }
}

int main(int argc, char** argv) {
  CHECK(argc == 3);
  CHECK(unsetenv("KVAF_TEST_HOST_SVM_OPEN") == 0);
  CpuRuntimeOptions options;
  options.hal_library = argv[1];
  const std::string scenario = argv[2];
  if (scenario == "concurrent") {
    std::promise<void> start;
    const auto gate = start.get_future().share();
    std::vector<std::future<void>> workers;
    for (int i = 0; i < 4; ++i)
      workers.emplace_back(std::async(std::launch::async, [&] {
        gate.wait();
        ExecutionPools(options);
      }));
    start.set_value();
    for (auto& worker : workers) worker.get();
    CheckReleased(options.hal_library, 24);
  } else {
    CHECK(scenario == "execution" || scenario == "legacy");
    // Both entry points must independently preserve HAL and share it with the
    // other entry point; each CTest invocation starts in a fresh process.
    if (scenario == "execution") {
      ExecutionPools(options);
      CheckReleased(options.hal_library, 6);
      LegacyPools(options);
    } else {
      LegacyPools(options);
      CheckReleased(options.hal_library, 3);
      ExecutionPools(options);
    }
    CheckReleased(options.hal_library, 9);
  }
  std::cout << "HAL/SVM reused; all pool mappings and allocations released\n";
}
