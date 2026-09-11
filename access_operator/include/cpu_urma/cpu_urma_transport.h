#ifndef KVAF_ACCESS_OPERATOR_CPU_URMA_TRANSPORT_H_
#define KVAF_ACCESS_OPERATOR_CPU_URMA_TRANSPORT_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "status.h"
#include "peer_link.h"

namespace kvaf {

// CPU-side URMA data plane shared by the production AccessOperator and its
// hardware benchmarks. PeerLink supplies the control-plane channel; this
// class owns only the local URMA resources and synchronous byte transfers.
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
  // Streaming wave size and number of WRs submitted per
  // urma_post_jetty_send_wr call. Zero prepares and submits one full linked
  // list per Jetty.
  std::uint32_t wqes_per_doorbell{0};
  // WRITE only: values > 1 enable one contiguous local SGE -> multiple
  // remote SGEs. Requests above device capacity are split across WRs.
  std::uint32_t remote_sgl_size{1};
  bool has_external_region{false};
  peer::PeerRegion external_region{};
};

class CpuUrmaTransport final {
 public:
  struct BatchTiming {
    // Preparation/submission time of the worker with the longest combined
    // prepare+post duration; concurrent workers are not summed.
    std::uint64_t prepare_ns{0};
    std::uint64_t post_ns{0};
    std::uint64_t poll_ns{0};
    // Wall time after acquiring the submission lock through final completion.
    // Includes validation, workspace/segment lookup and gaps between chunks.
    std::uint64_t total_ns{0};
    std::uint64_t wr_count{0};
  };

  CpuUrmaTransport();
  ~CpuUrmaTransport();

  CpuUrmaTransport(const CpuUrmaTransport&) = delete;
  CpuUrmaTransport& operator=(const CpuUrmaTransport&) = delete;

  Status Initialize(const CpuUrmaTransportOptions& options);
  // Verify all Jettys with small WRITEs and completion before selecting a link.
  Status ValidateLink();
  struct HostSge {
    void* data;
    std::size_t bytes;
  };
  // One local SGE per WR. Four fixed workers each own a Jetty and prepare
  // and submit its batches independently; completion is polled after joining.
  Status WriteSglBatch(std::uint64_t remote_offset, const HostSge* sources,
                       std::size_t count);
  // CPU BatchPut path: transfer packed CPU DRAM rows into independent HBM
  // destinations. This is the CPU-initiated equivalent of AIV BatchGet READs.
  // With remote_sgl_size > 1, adjacent local rows share one source SGE and
  // a WR containing up to remote_sgl_size remote destinations. Otherwise,
  // each row uses one WR.
  // Queue capacity is applied after grouping, with shared completion fences.
  Status WriteBatch(const std::uint64_t* remote_offsets,
                    const HostSge* sources, std::size_t count);
  // Register one page-aligned host region once. SGL transfers from any
  // sub-range of this region then reuse the same URMA segment.
  Status RegisterDirectHostMemory(void* base, std::size_t bytes);
  Status ReadSglBatch(std::uint64_t remote_offset, const HostSge* destinations,
                      std::size_t count);
  // CPU BatchGet path: transfer independent HBM rows into CPU DRAM. This is
  // the CPU-initiated equivalent of AIV BatchPut WRITEs.
  Status ReadBatch(const std::uint64_t* remote_offsets,
                   const HostSge* destinations, std::size_t count);
  Status Shutdown();

  bool initialized() const;
  std::uint64_t region_bytes() const;
  BatchTiming last_batch_timing() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf

#endif  // KVAF_ACCESS_OPERATOR_CPU_URMA_TRANSPORT_H_
