#pragma once

#include <cstdint>
#include <memory>

#include "kvaf/status.h"
#include "kvaf/types.h"

namespace kvaf::access::a5 {

// Allocates or pins Host DDR into one NPU address space and owns the matching
// halHostRegister(..., HOST_MEM_MAP_DEV, ...) lifetime.  This component is
// deliberately independent of HCOMM and liburma.
//
// Close is retryable: if it returns an error, the object retains every HAL
// handle and any owned backing that might still be device-visible.  The caller
// must keep the object alive and retry Close; Register/AllocateAndRegister are
// rejected until teardown has completed successfully.
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
  // True once halHostUnregisterEx has proved that the NPU no longer has a
  // mapping. Close may still fail afterward while freeing local Host memory
  // or unloading HAL; those failures do not imply live DMA reachability.
  bool device_mapping_removed() const;
  Status Close();

 private:
  struct Impl;
  Status LoadHal(bool require_allocator);
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf::access::a5
