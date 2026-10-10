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
  // Number of WRs submitted per
  // urma_post_jetty_send_wr call. Zero prepares and submits one full linked
  // list per Jetty.
  std::uint32_t wqes_per_doorbell{0};
  // Legacy public name. WRITE gathers up to this many local source SGEs into
  // one contiguous remote HBM SGE. Every input row remains a distinct SGE.
  // Device max_jfs_sge, max_msg_size, or a remote gap may split the WR earlier.
  std::uint32_t remote_sgl_size{1};
  // Number of independent Jettys and their submission/CQ owners.
  std::uint32_t jetty_count{4};
  bool has_external_region{false};
  peer::PeerRegion external_region{};
  // Maximum CPU workers used by batch planning and descriptor preparation.
  // WRITE uses at most one worker per Jetty owner.
  std::uint32_t prepare_thread_count{4};
  // Diagnostic only: capture WRITE queue ownership/completion skew using
  // existing per-queue timestamps. Disabled for normal transports.
  bool profile_write_queues{false};
};

class CpuUrmaTransport final {
 public:
  struct BatchTiming {
    // Wall timeline for WR construction, hardware posting and CQ completion.
    // WRITE workers poll their assigned Jettys without a global post barrier:
    // post_ns ends at the last post, and poll_ns is the remaining tail until
    // the last CQE. Concurrent worker time is not summed, so phases add up.
    std::uint64_t prepare_ns{0};
    std::uint64_t post_ns{0};
    std::uint64_t poll_ns{0};
    // Wall time after acquiring the submission lock through final completion.
    // Includes validation, workspace/segment lookup and gaps between chunks.
    std::uint64_t total_ns{0};
    std::uint64_t wr_count{0};
    // Per-sub-batch validation, workspace sizing and host segment selection.
    std::uint64_t setup_ns{0};
    // Entire prepare/post call, including OpenMP scheduling and joining.
    std::uint64_t dispatch_ns{0};
    // Pre-validation for non-contiguous batch APIs. WRITE grouping/descriptor
    // construction is included in prepare_ns; plan may be zero for one wave.
    std::uint64_t plan_ns{0};
  };
  struct WriteQueueProfile {
    struct Queue {
      std::uint64_t waves{}, rows{}, wr_count{};
      std::uint64_t post_ns{}, post_to_cq_ns{}, cq_tail_ns{}, tail_waves{};
    };
    std::uint64_t jettys{}, waves{}, cq_spread_ns{};
    // Sums over serial waves, never sums parallel owners into wall time.
    // post_to_cq includes any delay entering polling; CQ timestamps are
    // software observations, not the instant DMA reaches remote memory.
    std::array<Queue, 16> queue{};
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
  // Legacy one-local-SGE API for a packed remote range. Preparation may span
  // more CPU workers than Jettys; after a barrier, each Jetty has one
  // submission and CQ owner.
  Status WriteSglBatch(std::uint64_t remote_offset, const HostSge* sources,
                       std::size_t count);
  // CPU BatchPut path: gather CPU DRAM rows into HBM destinations. This is the
  // CPU-initiated equivalent of AIV BatchGet READs.
  // With remote_sgl_size > 1, rows targeting adjacent HBM ranges may share one
  // WR even when their local CPU addresses are unrelated. The WR gathers one
  // or more local source SGEs into one contiguous remote destination SGE.
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
  // Last WriteBatch only; control ReadSglBatch/WriteSglBatch do not overwrite
  // this record. Read after the synchronous call with no concurrent caller.
  // Returns zeroes when profile_write_queues is disabled.
  WriteQueueProfile last_write_queue_profile() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf

#endif  // KVAF_ACCESS_OPERATOR_CPU_URMA_TRANSPORT_H_
