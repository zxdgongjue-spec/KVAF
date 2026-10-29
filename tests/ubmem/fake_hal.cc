// Host-only test double. Shared file mappings model aliasing and ownership;
// this library is never linked into either hardware executable.
#include "ubmem/hal.h"

#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

namespace {
struct Handle { int fd; std::size_t size; };
std::unordered_map<void*, std::size_t> addresses;
std::string fail;
int handles = 0;
int maps = 0;
bool Fail(const char* stage) { if (fail != stage) return false; fail.clear(); return true; }
}
extern "C" {
void KvafFakeHalFail(const char* stage) { fail = stage; }
int KvafFakeHalHandles() { return handles; }
int KvafFakeHalAddresses() { return static_cast<int>(addresses.size()); }
int KvafFakeHalMaps() { return maps; }

int halMemCreate(void** out, std::size_t bytes, const kvaf::ubmem::MemoryProperty* prop, std::uint64_t) {
  if (Fail("create")) return 901;
  if (prop->memory_type != 3 || (prop->side != 0 && prop->side != 2)) return 902;
  char path[] = "/tmp/kvaf-fake-hal-XXXXXX";
  const int fd = mkstemp(path);
  if (fd < 0) return 903;
  unlink(path);
  if (ftruncate(fd, bytes) != 0) { close(fd); return 904; }
  *out = new Handle{fd, bytes};
  ++handles;
  return 0;
}
int halMemRelease(void* ptr) {
  if (Fail("release")) return 905;
  auto* handle = static_cast<Handle*>(ptr);
  close(handle->fd); delete handle; --handles;
  return 0;
}
int halMemAddressReserve(void** out, std::size_t bytes, std::size_t, void*, std::uint64_t) {
  if (Fail("reserve")) return 906;
  *out = mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (*out == MAP_FAILED) return 907;
  addresses.emplace(*out, bytes);
  return 0;
}
int halMemAddressFree(void* ptr) {
  if (Fail("free")) return 908;
  const auto found = addresses.find(ptr);
  if (found == addresses.end()) return 909;
  munmap(ptr, found->second); addresses.erase(found);
  return 0;
}
int halMemMap(void* address, std::size_t bytes, std::size_t, void* ptr, std::uint64_t) {
  if (Fail("map")) return 910;
  auto* handle = static_cast<Handle*>(ptr);
  if (bytes != handle->size || !addresses.count(address)) return 911;
  if (mmap(address, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, handle->fd, 0) == MAP_FAILED) return 912;
  ++maps;
  return 0;
}
int halMemUnmap(void*) {
  if (Fail("unmap")) return 913;
  --maps;
  return 0;
}
int halMemExportToShareableHandleV2(void* ptr, int type, std::uint64_t, kvaf::ubmem::FabricHandle* out) {
  if (Fail("export")) return 914;
  if (type != 1) return 915;
  std::memcpy(out->data(), ptr, sizeof(Handle));
  return 0;
}
int halMemImportFromShareableHandleV2(int type, kvaf::ubmem::FabricHandle* in, std::uint32_t, void** out) {
  if (Fail("import")) return 916;
  if (type != 1) return 917;
  Handle source{}; std::memcpy(&source, in->data(), sizeof(source));
  const int fd = dup(source.fd);
  if (fd < 0) return 918;
  *out = new Handle{fd, source.size}; ++handles;
  return 0;
}
int halMemTransShareableHandle(int, kvaf::ubmem::FabricHandle*, std::uint32_t* server, std::uint64_t* id) {
  if (Fail("translate")) return 919;
  *server = 1; *id = 1; return 0;
}
int halMemShareHandleSetAttribute(std::uint64_t, int, kvaf::ubmem::Hal::ShareAttribute) {
  return Fail("attribute") ? 920 : 0;
}
}
