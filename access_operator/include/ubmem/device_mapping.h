#pragma once

#include <acl/acl_rt.h>
#include <map>
#include <mutex>
#include <tuple>
#include "ubmem/hal.h"

namespace kvaf::ubmem {

Status Acl(const char* operation, aclError code);

// NPU-side lifecycle used by SHMEM's AclMemSegmentDevice for Host DDR.
// The caller selects the ACL device and drains kernels before Close().
class DeviceMapping {
 public:
  DeviceMapping() = default;
  ~DeviceMapping();
  DeviceMapping(const DeviceMapping&) = delete;
  DeviceMapping& operator=(const DeviceMapping&) = delete;

  Status Import(const FabricHandle& descriptor, std::uint64_t bytes);
  Status Close();
  std::uint64_t address() const { return reinterpret_cast<std::uintptr_t>(address_); }

 private:
  void* address_{};
  aclrtDrvMemHandle handle_{};
  bool mapped_{};
};

// Runtime-scoped cache: one device mapping per exported allocation/device.
// Reference changes and final unmap are atomic under the index lock.
class DeviceMappingCache {
 public:
  Status Acquire(std::uint32_t device, const FabricHandle&, std::uint64_t bytes,
                 std::shared_ptr<DeviceMapping>*);
  Status Release(std::shared_ptr<DeviceMapping>*);

 private:
  using Key = std::tuple<std::uint32_t, std::uint64_t, FabricHandle>;
  struct Entry {
    std::shared_ptr<DeviceMapping> mapping;
    std::size_t users = 0;
    Status failure;
  };
  std::mutex mutex_;
  std::map<Key, Entry> entries_;
};
}  // namespace kvaf::ubmem
