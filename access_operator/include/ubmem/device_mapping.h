#pragma once

#include <acl/acl_rt.h>
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

}  // namespace kvaf::ubmem
