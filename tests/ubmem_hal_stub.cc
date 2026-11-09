#include "ubmem/hal.h"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <mutex>

namespace {
std::mutex mutex;
bool initialized = false;
// create, release, reserve, free_address, map, unmap, master initialization.
std::array<unsigned, 7> calls{};
}

// The environment models kernel process state, which survives dlclose. The
// library's initialized flag models HAL userspace state, which does not.
extern "C" int halMemCreate(void** handle, std::size_t,
                            const kvaf::ubmem::MemoryProperty*, std::uint64_t) {
  std::lock_guard<std::mutex> lock(mutex);
  if (!initialized) {
    if (std::getenv("KVAF_TEST_HOST_SVM_OPEN")) {
      errno = EEXIST;
      return 38;
    }
    if (setenv("KVAF_TEST_HOST_SVM_OPEN", "1", 1)) return 1;
    initialized = true;
    ++calls[6];
  }
  *handle = std::malloc(1);
  if (!*handle) return 1;
  ++calls[0];
  return 0;
}
extern "C" int halMemRelease(void* handle) {
  std::lock_guard<std::mutex> lock(mutex);
  std::free(handle);
  ++calls[1];
  return 0;
}
extern "C" int halMemAddressReserve(void** address, std::size_t bytes,
                                    std::size_t, void*, std::uint64_t) {
  std::lock_guard<std::mutex> lock(mutex);
  *address = std::malloc(bytes);
  if (!*address) return 1;
  ++calls[2];
  return 0;
}
extern "C" int halMemAddressFree(void* address) {
  std::lock_guard<std::mutex> lock(mutex);
  std::free(address);
  ++calls[3];
  return 0;
}
extern "C" int halMemMap(void*, std::size_t, std::size_t, void*, std::uint64_t) {
  std::lock_guard<std::mutex> lock(mutex);
  ++calls[4];
  return 0;
}
extern "C" int halMemUnmap(void*) {
  std::lock_guard<std::mutex> lock(mutex);
  ++calls[5];
  return 0;
}
extern "C" int halMemExportToShareableHandleV2(
    void*, int, std::uint64_t, kvaf::ubmem::FabricHandle* handle) {
  handle->fill(1);
  return 0;
}
extern "C" int halMemTransShareableHandle(
    int, kvaf::ubmem::FabricHandle*, std::uint32_t* server, std::uint64_t* handle) {
  *server = 0;
  *handle = 1;
  return 0;
}
extern "C" int halMemShareHandleSetAttribute(
    std::uint64_t, int, kvaf::ubmem::Hal::ShareAttribute) {
  return 0;
}
extern "C" void kvafTestHalCalls(unsigned* out) {
  std::lock_guard<std::mutex> lock(mutex);
  for (std::size_t i = 0; i < calls.size(); ++i) out[i] = calls[i];
}
