#ifndef KVAF_RUNTIME_PEER_LINK_H_
#define KVAF_RUNTIME_PEER_LINK_H_

#include <array>
#include <cstdint>
#include <string>
#include <mutex>
#include <unordered_map>
#include <memory>
#include "execution_config.h"

#include "peer_protocol.h"
#include "cpu_gather.h"
#include "status.h"

namespace kvaf::execution {
struct Request;
struct Response;
}  // namespace kvaf::execution
namespace kvaf::peer {

struct PeerRegion {
  uint64_t session_id = 0U;
  uint64_t base = 0U;
  uint64_t length = 0U;
  uint32_t token_id = 0U;
  uint32_t token_value = 0U;
  uint32_t segment_key_bytes = 0U;
  std::array<uint8_t, kEidBytes> eid{};
  std::array<uint8_t, kSegKeyBytes> segment_key{};
  std::array<uint8_t, 128> fabric_handle{};
  uint64_t fabric_mapping_bytes{};
};

struct PeerCandidateInfo {
  uint32_t candidate_id = 0U;
  uint32_t device_id = 0U;
  uint32_t eid_index = 0U;
  std::string device_name;
};

// Per-Runtime peer control-plane client. Its endpoint is explicit: Runtime
// receives it through LinkTo and never reads process environment.
class PeerLink : public std::enable_shared_from_this<PeerLink> {
 public:
  explicit PeerLink(std::string address);
  ~PeerLink() = default;
  Status AllocateCacheRegion(std::uint64_t logical_bytes, std::uint64_t slot_size,
                             std::uint64_t auxiliary_bytes, TransportMode, PeerRegion*);
  std::shared_ptr<PeerLink> ExecutionSession(const execution::Request& config);
  Status PrepareExecution();
  Status CloseExecution();
  Status ResolveExecutionControl();
  Status QueryExecution(execution::Response*) const;
  // Typed legacy commands are encapsulated in an idempotent session request.
  bool Control(const void*, std::size_t, void*, std::size_t, std::string*) const;

  const std::string& address() const { return address_; }
  bool has_region() const { return !regions_.empty(); }

  Status GetCandidateCount(uint32_t* candidate_count, uint32_t* device_count = nullptr,
                           bool* device_filter_applied = nullptr) const;
  Status GetCandidateInfo(uint32_t candidate_id, PeerCandidateInfo* info) const;
  Status AllocateRegion(uint64_t ddr_bytes, uint64_t slot_size,
                        uint32_t candidate_id, PeerRegion* region);
  Status AllocateUbmemRegion(uint64_t ddr_bytes, uint64_t slot_size, PeerRegion* region);
  Status RetainUbmemMapping(uint64_t base);
  void ReleaseUbmemMapping(uint64_t base);
  Status ReleaseRegion(uint64_t base);
  Status LookupRegion(uint64_t base, PeerRegion* region) const;
  Status EnableDirectGather(uint64_t region_base) const;
  Status EnableGatherLanes(uint64_t region_base, uint64_t control_address, uint32_t lanes) const;
  Status OpenCpuPushGet(uint64_t region_base, OpenCpuPushGetRequest request,
                        OpenCpuPushGetResponse* response) const;
  Status StartCpuPushGet(uint64_t region_base) const;
  Status StopCpuPushGet(uint64_t region_base) const;
  Status CreateChannel(
      uint64_t region_base, const uint8_t local_qp_key[kQpKeyBytes],
      uint32_t local_qp_key_bytes, const uint8_t local_eid[kEidBytes],
      uint32_t peer_token_value, uint64_t local_tp_handle, uint32_t local_psn,
      std::array<uint8_t, kQpKeyBytes>* peer_qp_key,
      uint32_t* peer_qp_key_bytes, uint64_t* peer_tp_handle,
      uint32_t* peer_psn);

  // Shared TP is negotiated explicitly; keep the original overload for callers
  // using the existing per-channel transport setup.
  Status CreateChannel(
      uint64_t region_base, const uint8_t local_qp_key[kQpKeyBytes],
      uint32_t local_qp_key_bytes, const uint8_t local_eid[kEidBytes],
      uint32_t peer_token_value, uint64_t local_tp_handle, uint32_t local_psn,
      std::array<uint8_t, kQpKeyBytes>* peer_qp_key,
      uint32_t* peer_qp_key_bytes, uint64_t* peer_tp_handle,
      uint32_t* peer_psn, bool share_tp);

 private:
  struct ExecutionState;
  std::shared_ptr<ExecutionState> execution_;
  std::shared_ptr<PeerLink> parent_;
  std::string address_;
  std::unordered_map<uint64_t, PeerRegion> regions_;
  mutable std::mutex mapping_mutex_;
  std::unordered_map<uint64_t, std::size_t> ubmem_mappings_;
};

}  // namespace kvaf::peer

#endif  // KVAF_RUNTIME_PEER_LINK_H_
