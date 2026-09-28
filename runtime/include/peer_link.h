#ifndef KVAF_RUNTIME_PEER_LINK_H_
#define KVAF_RUNTIME_PEER_LINK_H_

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "peer_protocol.h"
#include "cpu_gather.h"
#include "status.h"

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
  uint32_t jetty_key_bytes = 0U;
  std::array<uint8_t, kQpKeyBytes> jetty_key{};
  bool device_memory = false;
};

struct GatherReady {
  std::uint64_t sequence{0U};
  std::uint64_t command_address{0U};
  std::uint64_t notify_address{0U};
  std::uint64_t ack_address{0U};
  std::uint32_t total_bytes{0U};
};

// Per-Runtime peer control-plane client. Its endpoint is explicit: Runtime
// receives it through LinkTo and never reads process environment.
class PeerLink {
 public:
  explicit PeerLink(std::string address);

  const std::string& address() const { return address_; }
  bool has_region() const { return !regions_.empty(); }

  Status GetCandidateCount(uint32_t* candidate_count) const;
  Status GetHbmEidCandidates(uint32_t device_id,
                            std::vector<std::array<uint8_t, kEidBytes>>* candidates) const;
  Status AllocateRegion(uint64_t ddr_bytes, uint64_t slot_size,
                        uint32_t candidate_id, PeerRegion* region);
  Status AllocateHbmRegion(uint64_t hbm_bytes, uint64_t slot_size,
                           uint32_t device_id, uint32_t eid_index,
                           PeerRegion* region);
  Status ReleaseRegion(uint64_t base);
  Status ReleaseHbmRegion(uint64_t session_id);
  Status LookupRegion(uint64_t base, PeerRegion* region) const;
  Status ConfigureGather(uint64_t region_base, uint64_t staging_address,
                         const std::vector<GatherSpan>& spans, GatherReady* ready) const;
  // Register a descriptor obtained by an out-of-band allocator (for example
  // RuntimeCore's NPU HBM allocator) so channel setup can use the same API.
  Status RememberRegion(const PeerRegion& region);
  Status CreateChannel(
      uint64_t region_base, const uint8_t local_qp_key[kQpKeyBytes],
      uint32_t local_qp_key_bytes, const uint8_t local_eid[kEidBytes],
      uint32_t peer_token_value, uint64_t local_tp_handle, uint32_t local_psn,
      std::array<uint8_t, kQpKeyBytes>* peer_qp_key,
      uint32_t* peer_qp_key_bytes, uint64_t* peer_tp_handle,
      uint32_t* peer_psn);

 private:
  std::string address_;
  std::unordered_map<uint64_t, PeerRegion> regions_;
};

}  // namespace kvaf::peer

#endif  // KVAF_RUNTIME_PEER_LINK_H_
