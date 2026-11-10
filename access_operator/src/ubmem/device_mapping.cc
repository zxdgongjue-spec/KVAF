#include "ubmem/device_mapping.h"

#include <cstdio>
#include <limits>

namespace kvaf::ubmem {

Status Acl(const char* operation, aclError code) {
  if (code == ACL_SUCCESS) return Status::OK();
  // Capture the failing task's details before cleanup ACL calls replace them.
  const char* detail = aclGetRecentErrMsg();
  if (detail && *detail) std::fprintf(stderr, "[KVAF_UBMEM] %s detail: %s\n", operation, detail);
  return HalError(operation, code);
}

DeviceMapping::~DeviceMapping() { (void)Close(); }

Status DeviceMapping::Import(const FabricHandle& descriptor, std::uint64_t bytes) {
  if (address_ || handle_ || bytes == 0 || bytes % kPageBytes != 0 ||
      bytes > std::numeric_limits<std::size_t>::max()) {
    return Status::InvalidArgument("invalid Fabric import size or mapping state");
  }
  auto status = Acl("aclrtReserveMemAddress",
      aclrtReserveMemAddress(&address_, bytes, 0, nullptr, 1));
  if (!status.ok()) return status;
  FabricHandle copy = descriptor;
  // ACL chooses the current device and performs the Runtime import checks.
  // Its FABRIC enum is 2; HAL's identically named handle type is 1.
  status = Acl("aclrtMemImportFromShareableHandleV2(FABRIC)",
      aclrtMemImportFromShareableHandleV2(copy.data(), ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, 0, &handle_));
  if (!status.ok()) return status;
  status = Acl("aclrtMapMem", aclrtMapMem(address_, bytes, 0, handle_, 0));
  if (status.ok()) mapped_ = true;
  return status;
}

Status DeviceMapping::Close() {
  if (mapped_) {
    auto status = Acl("aclrtUnmapMem", aclrtUnmapMem(address_));
    if (!status.ok()) return status;
    mapped_ = false;
  }
  if (handle_) {
    auto status = Acl("aclrtFreePhysical", aclrtFreePhysical(handle_));
    if (!status.ok()) return status;
    handle_ = nullptr;
  }
  if (address_) {
    auto status = Acl("aclrtReleaseMemAddress", aclrtReleaseMemAddress(address_));
    if (!status.ok()) return status;
    address_ = nullptr;
  }
  return Status::OK();
}

}  // namespace kvaf::ubmem
