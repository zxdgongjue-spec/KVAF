#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kvaf/status.h"
#include "kvaf/types.h"

namespace kvaf::access::a5 {

enum class EndpointLocation : std::uint32_t {
  kHost = 0,
  kDevice,
};

enum class MemoryLocation : std::uint32_t {
  kHostDdr = 0,
  kDeviceHbm,
};

enum class ChannelEngine : std::uint32_t {
  kCpu = 0,
  kAiv,
};

enum class ChannelRole : std::uint32_t {
  kClient = 0,
  kServer,
};

struct EndpointSpec {
  std::array<std::uint8_t, 16> eid{};
  EndpointLocation location{EndpointLocation::kDevice};
  std::uint32_t physical_device_id{0};
  std::uint32_t super_device_id{0};
  std::uint32_t server_index{0};
  std::uint32_t super_pod_index{0};
};

struct ImportedMemoryView {
  Address address{0};
  std::uint64_t bytes{0};
  MemoryLocation location{MemoryLocation::kHostDdr};
};

// Pins an existing Host DDR allocation into one NPU address space and owns
// the matching halHostRegister(..., HOST_MEM_MAP_DEV, ...) lifetime.
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

// Thin ownership wrapper around the HCOMM resource API. QP/CQ/token details
// intentionally remain opaque and never enter KVAF's stable Runtime ABI.
class HcommUrmaResources {
 public:
  HcommUrmaResources();
  ~HcommUrmaResources();

  HcommUrmaResources(const HcommUrmaResources&) = delete;
  HcommUrmaResources& operator=(const HcommUrmaResources&) = delete;

  Status OpenEndpoint(const EndpointSpec& local);
  Status RegisterMemory(const std::string& tag, Address address,
                        std::uint64_t bytes, MemoryLocation location,
                        std::uint32_t* registration_id);
  Status ExportMemory(std::uint32_t registration_id,
                      std::vector<std::uint8_t>* descriptor) const;
  Status ImportMemory(const std::vector<std::uint8_t>& descriptor,
                      ImportedMemoryView* view);
  Status CreateChannel(const EndpointSpec& peer, ChannelEngine engine,
                       ChannelRole role);
  Status WaitChannelReady(std::uint32_t timeout_ms);

  std::uint64_t channel() const;
  Status Close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf::access::a5
