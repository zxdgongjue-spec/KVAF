#include "ubmem/device_mapping.h"
#include <cstdlib>
#include <future>
#include <iostream>
#include <atomic>
#define CHECK(c)                                        \
  do {                                                  \
    if (!(c)) {                                         \
      std::cerr << "line " << __LINE__ << ": " #c "\n"; \
      std::abort();                                     \
    }                                                   \
  } while (false)
namespace {
std::atomic<int> imports{0}, unmaps{0}, freed{0};
bool fail_unmap = false;
bool fail_import = false;
}  // namespace
extern "C" {
const char* aclGetRecentErrMsg() { return "fake mapping failure"; }
aclError aclrtReserveMemAddress(void** address, size_t, size_t, void*, uint64_t) {
  *address = std::malloc(1);
  return 0;
}
aclError aclrtReleaseMemAddress(void* address) {
  std::free(address);
  return 0;
}
aclError aclrtMemImportFromShareableHandleV2(void*, aclrtMemSharedHandleType, uint64_t,
                                             aclrtDrvMemHandle* handle) {
  ++imports;
  if (fail_import) return 1;
  *handle = reinterpret_cast<aclrtDrvMemHandle>(std::malloc(1));
  return 0;
}
aclError aclrtMapMem(void*, size_t, size_t, aclrtDrvMemHandle, uint64_t) { return 0; }
aclError aclrtUnmapMem(void*) {
  if (fail_unmap) return 1;
  ++unmaps;
  return 0;
}
aclError aclrtFreePhysical(aclrtDrvMemHandle handle) {
  ++freed;
  std::free(reinterpret_cast<void*>(handle));
  return 0;
}
}
int main() {
  using namespace kvaf::ubmem;
  DeviceMappingCache cache;
  FabricHandle handle{};
  handle[0] = 1;
  std::shared_ptr<DeviceMapping> put, get, other;
  auto p =
      std::async(std::launch::async, [&] { return cache.Acquire(0, handle, kPageBytes, &put); });
  auto g =
      std::async(std::launch::async, [&] { return cache.Acquire(0, handle, kPageBytes, &get); });
  CHECK(p.get().ok() && g.get().ok() && put == get && imports == 1);
  CHECK(cache.Release(&put).ok() && !put && unmaps == 0);
  fail_unmap = true;
  CHECK(!cache.Release(&get).ok() && get && freed == 0);
  CHECK(!cache.Acquire(0, handle, kPageBytes, &other).ok());
  CHECK(cache.Release(&other).ok());
  fail_unmap = false;
  CHECK(cache.Release(&get).ok() && !get && unmaps == 1 && freed == 1);
  CHECK(cache.Acquire(0, handle, kPageBytes, &put).ok());
  CHECK(cache.Acquire(1, handle, kPageBytes, &other).ok() && put != other && imports == 3);
  CHECK(cache.Release(&put).ok() && cache.Release(&other).ok());
  fail_import = true;
  CHECK(!cache.Acquire(0, handle, kPageBytes, &get).ok() && get);
  CHECK(cache.Release(&get).ok() && !get);
  fail_import = false;
  CHECK(cache.Acquire(0, handle, kPageBytes, &get).ok());
  CHECK(cache.Release(&get).ok());
  std::cout << "shared Fabric mapping, device isolation and failed-unmap retention passed\n";
}
