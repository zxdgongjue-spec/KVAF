#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "kvaf/access_operator/aiv_simt_urma_h2d.h"
#include "kvaf/status.h"
#include "kvaf/types.h"

namespace kvaf::access::a5 {

enum class DirectUrmaExecutionDomain : std::uint32_t {
  kHostProcess = 0,
  kAicpu,
};

enum class DirectUrmaRole : std::uint32_t {
  kServer = 0,
  kClient,
};

struct DirectUrmaOpenSpec {
  DirectUrmaExecutionDomain execution_domain{
      DirectUrmaExecutionDomain::kAicpu};
  DirectUrmaRole role{DirectUrmaRole::kClient};
  std::string provider_library_path;
  std::string provider_soname{"libkvaf_urma_provider_device.so"};
  std::string urma_device;
  std::string local_eid;
  std::string peer_eid;
  std::uint32_t eid_index{0};
  std::uint32_t control_port{21115};
  std::uint32_t control_timeout_seconds{120};
  Address local_address{0};
  std::uint64_t local_allocation_bytes{0};
  std::uint64_t segment_payload_bytes{0};
};

struct DirectUrmaMemoryView {
  Address address{0};
  std::uint64_t bytes{0};
};

// Direct liburma/UDMA control plane. The Host endpoint calls the provider
// shared library in-process. The Device endpoint launches the same provider
// through AICPU so its provider-created SQ/CQ/doorbells are device-visible.
// No HCOMM handle or HCOMM resource ABI crosses this class.
class DirectUrmaResources {
 public:
  DirectUrmaResources();
  ~DirectUrmaResources();

  DirectUrmaResources(const DirectUrmaResources&) = delete;
  DirectUrmaResources& operator=(const DirectUrmaResources&) = delete;

  Status Open(const DirectUrmaOpenSpec& spec);
  DirectUrmaMemoryView local_memory() const;
  DirectUrmaMemoryView remote_memory() const;
  Status SnapshotDirectChannel(
      Address local_address, std::uint64_t local_bytes,
      Address remote_address, std::uint64_t remote_bytes,
      AivDirectUrmaChannel* direct_channel) const;
  // True once no provider/queue can access the caller's backing memory. This
  // can become true even when Close still reports a local control-buffer,
  // stream-handle, or dlclose cleanup error.
  bool transport_teardown_complete() const;
  Status Close();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf::access::a5
