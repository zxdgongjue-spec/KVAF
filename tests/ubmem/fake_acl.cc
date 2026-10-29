#include "ubmem/device_mapping.h"

#include <unordered_set>

extern "C" {
int halMemAddressReserve(void**, std::size_t, std::size_t, void*, std::uint64_t);
int halMemAddressFree(void*);
int halMemImportFromShareableHandleV2(int, kvaf::ubmem::FabricHandle*, std::uint32_t, void**);
int halMemMap(void*, std::size_t, std::size_t, void*, std::uint64_t);
int halMemUnmap(void*);
int halMemRelease(void*);
}

namespace {
std::unordered_set<void*> reservations, handles, mapped;
}

extern "C" {
int KvafFakeAclReservations() { return static_cast<int>(reservations.size()); }
int KvafFakeAclHandles() { return static_cast<int>(handles.size()); }
const char* aclGetRecentErrMsg() { return "fake ACL VMM failure"; }

aclError aclrtReserveMemAddress(void** out, std::size_t bytes, std::size_t alignment, void* hint, std::uint64_t flags) {
  if (flags != 1 || hint != nullptr || alignment != 0) return 921;
  const int ret = halMemAddressReserve(out, bytes, alignment, hint, flags);
  if (ret == 0) reservations.insert(*out);
  return ret;
}
aclError aclrtReleaseMemAddress(void* ptr) {
  if (!reservations.count(ptr) || mapped.count(ptr)) return 922;
  const int ret = halMemAddressFree(ptr);
  if (ret == 0) reservations.erase(ptr);
  return ret;
}
aclError aclrtMemImportFromShareableHandleV2(void* descriptor, aclrtMemSharedHandleType type,
                                           std::uint64_t flags, aclrtDrvMemHandle* out) {
  if (type != ACL_MEM_SHARE_HANDLE_TYPE_FABRIC || flags != 0) return 923;
  const int ret = halMemImportFromShareableHandleV2(1, static_cast<kvaf::ubmem::FabricHandle*>(descriptor), 0, out);
  if (ret == 0) handles.insert(*out);
  return ret;
}
aclError aclrtMapMem(void* ptr, std::size_t bytes, std::size_t offset, aclrtDrvMemHandle handle, std::uint64_t flags) {
  if (!reservations.count(ptr) || !handles.count(handle) || offset != 0 || flags != 0) return 924;
  const int ret = halMemMap(ptr, bytes, offset, handle, flags);
  if (ret == 0) mapped.insert(ptr);
  return ret;
}
aclError aclrtUnmapMem(void* ptr) {
  if (!mapped.count(ptr)) return 925;
  const int ret = halMemUnmap(ptr);
  if (ret == 0) mapped.erase(ptr);
  return ret;
}
aclError aclrtFreePhysical(aclrtDrvMemHandle handle) {
  if (!handles.count(handle)) return 926;
  const int ret = halMemRelease(handle);
  if (ret == 0) handles.erase(handle);
  return ret;
}
}
