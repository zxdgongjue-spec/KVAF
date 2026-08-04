#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kvaf/access_operator/aiv_simt_urma_h2d.h"
#include "kvaf/access_operator/a5/host_ddr_mapping.h"
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

// Ownership wrapper around the HCOMM Host resource API. HCOMM creates and
// exchanges the Endpoint, MR, EID, TP, Jetty, SQ and CQ resources. Once the
// AIV channel is READY, SnapshotDirectChannel copies only the versioned queue
// and token metadata required by KVAF's direct device data plane.
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
                      MemoryLocation expected_location,
                      ImportedMemoryView* view);

  // Pure descriptor construction/ABI validation. This does not initialize
  // ACL or call HCOMM, so callers can reject incompatible installed headers
  // before allocating or registering resource-owning memory.
  static Status ValidateChannelDescriptor(const EndpointSpec& peer,
                                          ChannelRole role);
  Status CreateChannel(const EndpointSpec& peer, ChannelEngine engine,
                       ChannelRole role,
                       const std::string& channel_name = {});
  Status PollChannelReady(bool* ready);
  Status WaitChannelReady(std::uint32_t timeout_ms);
  Status SnapshotDirectChannel(Address local_address,
                               std::uint64_t local_bytes,
                               Address remote_address,
                               std::uint64_t remote_bytes,
                               AivDirectUrmaChannel* direct_channel) const;

  // CANN HCOMM v3 adds HcommChannelDesc.channelName. Older installed public
  // headers support only anonymous pairing; callers must make that downgrade
  // explicit and provide their own endpoint-exclusivity proof.
  bool SupportsChannelName() const;
  std::uint64_t channel() const;
  // Retryable reverse teardown.  Channel destruction must succeed before any
  // lower resource is touched; after that, independent imports/registrations
  // are drained best-effort and failed handles remain owned for a later retry.
  Status Close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf::access::a5
