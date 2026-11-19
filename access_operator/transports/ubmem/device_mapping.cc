#include "ubmem/device_mapping.h"
#include <acl/acl.h>

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

Status DeviceMappingCache::Acquire(std::uint32_t device, const FabricHandle& handle,
                                   std::uint64_t bytes, std::shared_ptr<DeviceMapping>* out) {
  if (!out || *out) return Status::InvalidArgument("mapping output is occupied");
  std::lock_guard<std::mutex> lock(mutex_);
  auto [it, created] = entries_.try_emplace({device, bytes, handle});
  auto& e = it->second;
  if (created) {
    e.mapping = std::make_shared<DeviceMapping>();
    e.failure = e.mapping->Import(handle, bytes);
  }
  // Even failed imports retain their partial handles until the last owner closes.
  ++e.users;
  *out = e.mapping;
  return e.failure;
}
Status DeviceMappingCache::Release(std::shared_ptr<DeviceMapping>* owner) {
  if (!owner || !*owner) return Status::OK();
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = entries_.begin(); it != entries_.end(); ++it) {
    auto& e = it->second;
    if (e.mapping != *owner) continue;
    if (e.users == 1) {
      auto s = e.mapping->Close();
      if (!s.ok()) {
        e.failure = s;
        return s;
      }
      entries_.erase(it);
    } else
      --e.users;
    owner->reset();
    return Status::OK();
  }
  return Status::InvalidState("mapping lease is not owned by this Runtime");
}
}  // namespace kvaf::ubmem
