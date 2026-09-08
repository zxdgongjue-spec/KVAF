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
  // Number of independent local Jettys used by batch transfers. Each Jetty
  // owns its JFC/JFR/SQ and establishes its own CTP to the remote target.
  std::uint32_t jetty_count{1};
  // Number of OpenMP threads used to prepare WR/SGE entries. URMA submission
  // and CQ polling stay on the calling thread.
  std::uint32_t omp_thread_count{1};
  // Streaming wave size and number of WRs submitted per
  // urma_post_jetty_send_wr call. Zero prepares and submits one full linked
  // list per Jetty.
  std::uint32_t wqes_per_doorbell{0};
  // Maximum local SGEs per production WR; only adjacent remote ranges merge.
  std::uint32_t sgl_size{1};
  bool has_external_region{false};
  peer::PeerRegion external_region{};
};

class CpuUrmaTransport final {
 public:
  struct BatchTiming {
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
  struct HostSge {
    void* data;
    std::size_t bytes;
  };
  // Builds one WR per SGL group, prepares the WQEs with OpenMP, distributes
  // them across all configured Jettys, posts the complete batch, then polls
  // one completion fence per Jetty.
  Status WriteSglBatch(std::uint64_t remote_offset, const HostSge* sources,
                       std::size_t count, std::size_t sgl_size);
  // CPU BatchPut path: transfer packed CPU DRAM rows into independent HBM
  // destinations. This is the CPU-initiated equivalent of AIV BatchGet READs.
  // Adjacent remote rows share a WR up to options.sgl_size; gaps start a new
  // WR. Queue capacity is applied after grouping, with shared completion fences.
  Status WriteBatch(const std::uint64_t* remote_offsets,
                    const HostSge* sources, std::size_t count);
  // Register one page-aligned host region once. SGL transfers from any
  // sub-range of this region then reuse the same URMA segment.
  Status RegisterDirectHostMemory(void* base, std::size_t bytes);
  Status ReadSglBatch(std::uint64_t remote_offset, const HostSge* destinations,
                      std::size_t count, std::size_t sgl_size);
  // CPU BatchGet path: transfer independent HBM rows into CPU DRAM. This is
  // the CPU-initiated equivalent of AIV BatchPut WRITEs.
  Status ReadBatch(const std::uint64_t* remote_offsets,
                   const HostSge* destinations, std::size_t count);
  Status Shutdown();

  bool initialized() const;
  std::uint64_t region_bytes() const;
  std::uint32_t max_sge() const;
  BatchTiming last_batch_timing() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf

#endif  // KVAF_ACCESS_OPERATOR_CPU_URMA_TRANSPORT_H_
