#include "ubmem_runtime.h"
#include "ubmem/device_mapping.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <thread>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

extern "C" int KvafFakeAclReservations();
extern "C" int KvafFakeAclHandles();

int main(int argc, char** argv) {
  CHECK(argc == 2);
  void* library = dlopen(argv[1], RTLD_NOW);
  CHECK(library != nullptr);
  auto fail = reinterpret_cast<void(*)(const char*)>(dlsym(library, "KvafFakeHalFail"));
  auto handles = reinterpret_cast<int(*)()>(dlsym(library, "KvafFakeHalHandles"));
  auto addresses = reinterpret_cast<int(*)()>(dlsym(library, "KvafFakeHalAddresses"));
  auto maps = reinterpret_cast<int(*)()>(dlsym(library, "KvafFakeHalMaps"));
  CHECK(fail && handles && addresses && maps);
  auto hal = std::make_shared<kvaf::ubmem::Hal>();
  CHECK(hal->Load(argv[1]).ok());
  for (const char* stage : {"create", "reserve", "map", "export", "translate", "attribute"}) {
    fail(stage);
    { kvaf::ubmem::Mapping mapping(hal); kvaf::ubmem::FabricHandle descriptor{};
      CHECK(!mapping.AllocateHost(kvaf::ubmem::kPageBytes, -1, &descriptor).ok()); }
    CHECK(handles() == 0 && addresses() == 0 && maps() == 0);
  }
  kvaf::ubmem::CpuRuntime provider;
  CHECK(provider.Initialize(-1, argv[1]).ok());
  const int listener = socket(AF_INET, SOCK_STREAM, 0);
  CHECK(listener >= 0);
  sockaddr_in address{}; address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  CHECK(bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
  socklen_t length = sizeof(address);
  CHECK(getsockname(listener, reinterpret_cast<sockaddr*>(&address), &length) == 0);
  CHECK(listen(listener, 8) == 0);
  std::thread server([&] {
    for (int i = 0; i < 8; ++i) {
      int connection = accept(listener, nullptr, nullptr); CHECK(connection >= 0);
      (void)provider.ServeConnection(connection); close(connection);
    }
  });
  kvaf::peer::PeerLink link("127.0.0.1:" + std::to_string(ntohs(address.sin_port)));
  kvaf::peer::PeerRegion region{};
  CHECK(link.AllocateUbmemRegion(kvaf::ubmem::kPageBytes, 4096, &region).ok());
  for (const char* stage : {"import", "reserve", "map"}) {
    fail(stage);
    {
      kvaf::ubmem::DeviceMapping failed;
      CHECK(!failed.Import(region.fabric_handle, region.fabric_mapping_bytes).ok());
    }
    CHECK(KvafFakeAclHandles() == 0 && KvafFakeAclReservations() == 0 && handles() == 1 && addresses() == 1 && maps() == 1);
  }
  {
    kvaf::ubmem::DeviceMapping imported;
    CHECK(imported.Import(region.fabric_handle, region.fabric_mapping_bytes).ok());
    CHECK(KvafFakeAclReservations() == 1);
    CHECK(link.RetainUbmemMapping(region.base).ok());
    CHECK(link.ReleaseRegion(region.base).code() == kvaf::StatusCode::kBusy);
    CHECK(link.UbmemRange(region.base, 0, 656, 29, true).ok());
    auto* data = reinterpret_cast<std::uint8_t*>(imported.address());
    for (unsigned i = 0; i < 656; ++i) CHECK(data[i] == static_cast<std::uint8_t>((i * 31) ^ (i >> 8) ^ (i >> 16) ^ 29));
    for (unsigned i = 0; i < 656; ++i) data[i] = static_cast<std::uint8_t>((i * 31) ^ (i >> 8) ^ (i >> 16) ^ 71);
    CHECK(link.UbmemRange(region.base, 0, 656, 71, false).ok());
    data[0] ^= 1;
    CHECK(!link.UbmemRange(region.base, 0, 656, 71, false).ok());
    fail("unmap"); CHECK(!imported.Close().ok());
    CHECK(handles() == 2 && maps() == 2 && KvafFakeAclReservations() == 1);
    fail("release"); CHECK(!imported.Close().ok());
    CHECK(handles() == 2 && maps() == 1 && KvafFakeAclReservations() == 1 && KvafFakeAclHandles() == 1);
    fail("free"); CHECK(!imported.Close().ok());
    CHECK(handles() == 1 && maps() == 1 && KvafFakeAclReservations() == 1 && KvafFakeAclHandles() == 0);
    CHECK(imported.Close().ok());
    CHECK(KvafFakeAclReservations() == 0);
    link.ReleaseUbmemMapping(region.base);
  }
  CHECK(link.ReleaseRegion(region.base).ok());
  CHECK(link.AllocateUbmemRegion(4096, 4096, &region).ok());
  CHECK(region.length == 4096 && region.fabric_mapping_bytes == kvaf::ubmem::kPageBytes);
  CHECK(link.ReleaseRegion(region.base).ok());
  fail("create");
  const auto error = link.AllocateUbmemRegion(4096, 4096, &region);
  CHECK(!error.ok() && error.message().find("halMemCreate") != std::string::npos);
  server.join(); close(listener);
  CHECK(provider.Shutdown().ok());
  CHECK(handles() == 0 && addresses() == 0 && maps() == 0);
  dlclose(library);
  std::puts("UBmem HAL export / ACL import rollback, aliasing, CPU oracle, and control protocol passed");
}
