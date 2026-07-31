#pragma once

#include <cstdint>
#include <memory>

#include "kvaf/status.h"
#include "kvaf/types.h"

namespace kvaf::access::a5 {

// Pins an existing Host DDR allocation into one NPU address space and owns
// the matching halHostRegister(..., HOST_MEM_MAP_DEV, ...) lifetime.  This
// component is deliberately independent of HCOMM and liburma.
class HostDdrDeviceMapping {
 public:
  HostDdrDeviceMapping();
  ~HostDdrDeviceMapping();

  HostDdrDeviceMapping(const HostDdrDeviceMapping&) = delete;
  HostDdrDeviceMapping& operator=(const HostDdrDeviceMapping&) = delete;

  Status AllocateAndRegister(std::uint64_t bytes,
                             std::uint32_t logical_device_id);
  Status Register(Address host_address, std::uint64_t bytes,
                  std::uint32_t logical_device_id);
  Address host_address() const;
  Address device_address() const;
  std::uint64_t bytes() const;
  Status Close();

 private:
  struct Impl;
  Status LoadHal(bool require_allocator);
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf::access::a5
