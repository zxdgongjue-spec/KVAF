#ifndef KVAF_RUNTIME_PEER_LINK_H_
#define KVAF_RUNTIME_PEER_LINK_H_

#include <array>
#include <cstdint>
#include <string>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <optional>
#include <vector>
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
  // Local descriptor identity; wire encoding remains unchanged.
  uint64_t epoch = 0U, region_id = 0U;
  bool has_memory_eid{false};
  std::array<uint8_t, kEidBytes> memory_eid{};
};

struct PeerCandidateInfo {
  uint32_t candidate_id = 0U;
  uint32_t device_id = 0U;
  uint32_t eid_index = 0U;
  std::string device_name;
  std::array<uint8_t, kEidBytes> eid{};
};

struct PeerCandidateTable {
  uint32_t count{}, device_count{};
  bool device_fixed{}, local{};
};

struct CpuFeSelection {
  PeerCandidateInfo primary;
  std::vector<PeerCandidateInfo> read_fes;
  bool explicit_devices{};
  uint32_t requested_count{};

  Status ForWorkers(uint64_t workers, std::vector<PeerCandidateInfo>* active) const;
};

// Per-Runtime peer control-plane client. Its endpoint is explicit: Runtime
// receives it through LinkTo and never reads process environment.
class PeerLink : public std::enable_shared_from_this<PeerLink> {
 public:
  explicit PeerLink(std::string address);
  virtual ~PeerLink() = default;
  Status AllocateCacheRegion(std::uint64_t logical_bytes, std::uint64_t slot_size,
                             std::uint64_t auxiliary_bytes, TransportMode, PeerRegion*);
  std::shared_ptr<PeerLink> ExecutionSession(const execution::Request& config,
                                             const PeerCandidateInfo* candidate = nullptr);
  Status PrepareExecution();
  Status PrepareCpuPullRegion(uint64_t base, uint32_t candidate, PeerRegion*);
  Status CloseExecution();
  Status ResolveExecutionControl();
  Status QueryExecution(execution::Response*) const;
  // Typed legacy commands are encapsulated in an idempotent session request.
  bool Control(const void*, std::size_t, void*, std::size_t, std::string*) const;

  const std::string& address() const { return address_; }
  bool has_region() const { return !regions_.empty(); }

  Status GetCandidateCount(uint32_t* candidate_count, uint32_t* device_count = nullptr,
                           bool* device_filter_applied = nullptr) const;
  Status GetCandidateInfo(uint32_t candidate_id, PeerCandidateInfo* info, bool with_eid = true) const;
  Status GetCacheCandidateCount(PeerCandidateTable* table) const;
  Status DiscoverCandidates(PeerCandidateTable* table, bool cache_scope = true);
  Status SelectCpuCandidate(const PeerCandidateTable&, const std::string& device,
                            const std::optional<std::array<uint8_t, kEidBytes>>& pair_cpu_eid,
                            PeerCandidateInfo* selected);
  Status SelectCpuFes(const PeerCandidateTable&, const std::string& devices,
                      const std::optional<std::array<uint8_t, kEidBytes>>& pair_cpu_eid,
                      bool single_cpu_fe, CpuFeSelection* selected) const;
  Status AllocateRegion(uint64_t ddr_bytes, uint64_t slot_size,
                        uint32_t candidate_id, PeerRegion* region);
  Status AllocateUbmemRegion(uint64_t ddr_bytes, uint64_t slot_size, PeerRegion* region);
  Status RetainUbmemMapping(uint64_t base);
  void ReleaseUbmemMapping(uint64_t base);
  Status ReleaseRegion(uint64_t base);
  virtual Status LookupRegion(uint64_t base, PeerRegion* region) const;
  Status PrepareGatherWorkspace(uint64_t region_base, uint64_t bytes, PeerRegion* workspace);
  Status PrepareGatherRoute(uint64_t workspace_base, uint32_t candidate, PeerRegion* route);
  Status PrepareGatherWindow(uint64_t region_base, uint64_t budget, uint32_t token_bytes,
                             uint32_t workers, PeerRegion* workspace, uint32_t* capacity);
  Status EnableDirectGather(uint64_t region_base) const;
  Status EnableGatherLanes(uint64_t region_base, uint64_t control_address, uint32_t lanes,
                           bool aiv = false) const;
  Status EnableGatherLanesWithCapabilities(uint64_t region_base, uint64_t control_address,
                                           uint32_t lanes, uint32_t& capabilities, bool aiv = false) const;
  Status StopGatherLanes(uint64_t region_base) const;
  Status OpenCpuPushMailbox(uint64_t region_base, OpenCpuPushMailboxRequest request,
                            OpenCpuPushMailboxResponse* response, PeerRegion* mailbox);
  Status StartCpuPushGet(uint64_t region_base) const;
  Status StopCpuPushGet(uint64_t region_base, uint64_t mailbox_base);
  Status OpenAivCpuPushGet(uint64_t region_base, OpenAivCpuPushGetRequest request,
                          OpenAivCpuPushGetResponse* response) const;
  Status OpenAivCpuPushGet(uint64_t region_base, OpenAivCpuPushJettyRequest request,
                          OpenAivCpuPushJettyResponse* response, bool cpu_gather = false) const;
  Status StartAivCpuPushGet(uint64_t region_base) const;
  Status StopAivCpuPushGet(uint64_t region_base) const;
  virtual Status CreateChannel(
      uint64_t region_base, const uint8_t local_qp_key[kQpKeyBytes],
      uint32_t local_qp_key_bytes, const uint8_t local_eid[kEidBytes],
      uint32_t peer_token_value, uint64_t local_tp_handle, uint32_t local_psn,
      std::array<uint8_t, kQpKeyBytes>* peer_qp_key,
      uint32_t* peer_qp_key_bytes, uint64_t* peer_tp_handle,
      uint32_t* peer_psn);

  // Shared TP is negotiated explicitly; keep the original overload for callers
  // using the existing per-channel transport setup.
  virtual Status CreateChannel(
      uint64_t region_base, const uint8_t local_qp_key[kQpKeyBytes],
      uint32_t local_qp_key_bytes, const uint8_t local_eid[kEidBytes],
      uint32_t peer_token_value, uint64_t local_tp_handle, uint32_t local_psn,
      std::array<uint8_t, kQpKeyBytes>* peer_qp_key,
      uint32_t* peer_qp_key_bytes, uint64_t* peer_tp_handle,
      uint32_t* peer_psn, bool share_tp);
  Status CreateChannel(
      uint64_t region_base, const uint8_t local_qp_key[kQpKeyBytes],
      uint32_t local_qp_key_bytes, const uint8_t local_eid[kEidBytes],
      uint32_t peer_token_value, uint64_t local_tp_handle, uint32_t local_psn,
      std::array<uint8_t, kQpKeyBytes>* peer_qp_key,
      uint32_t* peer_qp_key_bytes, uint64_t* peer_tp_handle,
      uint32_t* peer_psn, bool share_tp,
      uint32_t route_selector);

 private:
  struct ExecutionState;
  std::shared_ptr<ExecutionState> execution_;
  std::shared_ptr<PeerLink> parent_;
  std::string address_;
  std::optional<std::array<uint8_t, kEidBytes>> selected_cpu_eid_;
  std::unordered_map<uint64_t, PeerRegion> regions_;
  mutable std::mutex mapping_mutex_;
  std::unordered_map<uint64_t, std::size_t> ubmem_mappings_;
};

}  // namespace kvaf::peer

#endif  // KVAF_RUNTIME_PEER_LINK_H_
