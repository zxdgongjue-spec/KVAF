#ifndef KVAF_RUNTIME_CPU_URMA_TRANSPORT_H_
#define KVAF_RUNTIME_CPU_URMA_TRANSPORT_H_

#include <cstddef>
#include <cstdint>
#include <array>
#include <memory>
#include <string>

#include "status.h"
#include "peer_link.h"

namespace kvaf {

// A small CPU-only URMA data-plane client.  It deliberately keeps the
// control-plane dependency at PeerLink and exposes only byte transfers, so it
// can later be wrapped by TransportRuntime without changing the URMA setup.
struct CpuUrmaTransportOptions {
  std::string peer_address;  // host:port of kvaf_cpu_service
  std::string ub_device;     // empty: first usable UB device
  bool has_eid_index{false};
  std::uint32_t eid_index{0};
  // Optional EID selected by the caller's device-discovery logic. When set,
  // Initialize only accepts a URMA device exposing this exact EID. H2H
  // callers leave it unset; the HBM probe fills it from the selected ACL
  // device's CLOS/pair-info records.
  bool has_preferred_eid{false};
  std::array<std::uint8_t, 16> preferred_eid{};
  // Remote CPU service candidate selected by the caller's retry loop. H2H
  // and single-candidate deployments use the default value zero.
  std::uint32_t peer_candidate_id{0};
  std::uint64_t region_bytes{4096};
  // When true, peer_address points to kvaf_npu_service and the remote region
  // is device HBM. The CPU still owns and registers only its local host
  // buffer; no HBM registration is attempted through liburma.
  bool remote_hbm{false};
  std::uint32_t remote_device_id{0};
  std::uint32_t remote_eid_index{0};
  // Number of independent local Jettys used by batch transfers. Each Jetty
  // owns its JFC/JFR/SQ and establishes its own CTP to the remote target.
  std::uint32_t jetty_count{1};
  // Number of persistent CPU workers used by batch transfers. A worker may
  // own multiple Jettys; each Jetty still has exactly one producer.
  std::uint32_t worker_count{1};
  // When true, each worker owns its Jettys end-to-end: prepare, post, and CQ
  // polling. When false, workers only prepare WRs and the caller owns URMA IO.
  bool aiv_thread_mode{false};
  bool has_external_region{false};
  peer::PeerRegion external_region{};
};

class CpuUrmaTransport final {
 public:
  CpuUrmaTransport();
  ~CpuUrmaTransport();

  CpuUrmaTransport(const CpuUrmaTransport&) = delete;
  CpuUrmaTransport& operator=(const CpuUrmaTransport&) = delete;

  Status Initialize(const CpuUrmaTransportOptions& options);
  // Uses the caller's host buffer directly as the URMA SGE. The segment is
  // registered and cached on first use, so no bounce-buffer copy is made.
  Status Write(std::uint64_t remote_offset, const void* source, std::size_t bytes);
  struct HostSge {
    void* data;
    std::size_t bytes;
  };
  Status WriteSgl(std::uint64_t remote_offset, const HostSge* sources, std::size_t count);
  // Register one page-aligned host region once. Direct transfers from any
  // sub-range of this region then reuse the same URMA segment.
  Status RegisterDirectHostMemory(void* base, std::size_t bytes);
  // Attempts a direct CPU URMA WRITE from device memory (for example an
  // aclrtMalloc HBM pointer), without a KVAF host-side memcpy. The pointer
  // must remain valid until this call returns. Failure means the active
  // liburma/provider cannot register or transmit from that device-memory
  // address. The API does not claim that a provider never uses internal
  // staging; that is a provider/driver implementation detail.
  Status WriteDeviceMemory(std::uint64_t remote_offset, const void* device_source,
                           std::size_t bytes);
  Status Read(std::uint64_t remote_offset, void* destination, std::size_t bytes);
  // Posts one WR per HostSge as a linked list and waits only for the final
  // completion. Jettys are sharded across persistent workers. Each worker
  // posts all of its Jetty lists before polling their final completions.
  Status WriteBatch(std::uint64_t remote_offset, const HostSge* sources, std::size_t count);
  Status ReadBatch(std::uint64_t remote_offset, const HostSge* destinations, std::size_t count);
  Status ReadSgl(std::uint64_t remote_offset, const HostSge* destinations, std::size_t count);
  Status Shutdown();

  bool initialized() const;
  std::uint64_t region_bytes() const;
  std::uint32_t max_sge() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf

#endif  // KVAF_RUNTIME_CPU_URMA_TRANSPORT_H_
