#ifndef KVAF_RUNTIME_PEER_PROTOCOL_H_
#define KVAF_RUNTIME_PEER_PROTOCOL_H_

#include <cstddef>
#include <cstdint>

// This protocol carries control-plane records between two Runtime peers. The
// first deployment is CPU DDR served to an Ascend NPU, but the records carry
// no role-specific wire semantics: data continues over UB.
namespace kvaf::peer {

constexpr uint32_t kProtocolMagic = 0x4b564350U;  // "KVCP"
// Version 5 adds NPU EID discovery. HBM allocation's eid_index now indexes
// the same CLOS/pair-info candidate list, rather than a provider-specific list.
constexpr uint16_t kProtocolVersion = 5U;
constexpr uint32_t kQpKeyBytes = 64U;
constexpr uint32_t kSegKeyBytes = 64U;
constexpr uint32_t kEidBytes = 16U;

enum class Command : uint16_t {
  kAllocateRegion = 1U,
  kCreateChannel = 2U,
  kReleaseRegion = 3U,
  kGetCandidateCount = 4U,
  // NPU service owns device HBM.  The response carries the provider-created
  // segment key and Jetty descriptor; the CPU peer imports both with liburma.
  kAllocateHbmRegion = 5U,
  kReleaseHbmRegion = 6U,
  kGetHbmEidCandidate = 7U,
};

enum class Result : int32_t {
  kOk = 0,
  kInvalidRequest = 1,
  kNotFound = 2,
  kResourceError = 3,
  kTransportError = 4,
};

#pragma pack(push, 1)
struct MessageHeader {
  uint32_t magic;
  uint16_t version;
  uint16_t command;
  uint32_t bytes;
};

struct AllocateRegionRequest {
  MessageHeader header;
  uint64_t ddr_bytes;
  uint64_t slot_size;
  uint32_t candidate_id;
  uint32_t reserved;
};

struct CandidateCountRequest {
  MessageHeader header;
};

struct CandidateCountResponse {
  MessageHeader header;
  int32_t result;
  uint32_t candidate_count;
};

struct AllocateRegionResponse {
  MessageHeader header;
  int32_t result;
  uint32_t segment_key_bytes;
  uint64_t session_id;
  uint64_t peer_base;
  uint64_t peer_length;
  uint32_t token_id;
  uint32_t token_value;
  uint8_t peer_eid[kEidBytes];
  uint8_t segment_key[kSegKeyBytes];
};

// HBM descriptors intentionally use fixed-size wire fields.  segment_key is
// copied verbatim from the provider's MemKey.value (never reconstructed), and
// jetty_key is the public UMDK Jetty id/mode record exported by the NPU side.
struct AllocateHbmRegionRequest {
  MessageHeader header;
  uint64_t hbm_bytes;
  uint64_t slot_size;
  uint32_t device_id;
  uint32_t eid_index;
};

struct AllocateHbmRegionResponse {
  MessageHeader header;
  int32_t result;
  uint32_t segment_key_bytes;
  uint64_t session_id;
  uint64_t hbm_bytes;
  uint64_t hbm_base;
  uint32_t token_id;
  uint32_t token_value;
  uint32_t jetty_key_bytes;
  uint32_t reserved;
  uint8_t peer_eid[kEidBytes];
  uint8_t segment_key[kSegKeyBytes];
  uint8_t jetty_key[kQpKeyBytes];
};

// Candidate indices are used by AllocateHbmRegionRequest.eid_index.
struct HbmEidCandidateRequest {
  MessageHeader header;
  uint32_t device_id;
  uint32_t candidate_index;
};

struct HbmEidCandidateResponse {
  MessageHeader header;
  int32_t result;
  uint32_t candidate_count;
  uint32_t candidate_index;
  uint8_t eid[kEidBytes];
};

struct ReleaseHbmRegionRequest {
  MessageHeader header;
  uint64_t session_id;
};

struct CreateChannelRequest {
  MessageHeader header;
  uint64_t session_id;
  uint64_t local_tp_handle;
  uint32_t local_qp_key_bytes;
  uint32_t peer_token_value;
  uint32_t local_psn;
  uint32_t reserved;
  uint8_t local_eid[kEidBytes];
  uint8_t local_qp_key[kQpKeyBytes];
};

struct CreateChannelResponse {
  MessageHeader header;
  int32_t result;
  uint32_t peer_qp_key_bytes;
  uint64_t peer_tp_handle;
  uint32_t peer_psn;
  uint32_t reserved;
  uint8_t peer_qp_key[kQpKeyBytes];
};

struct ReleaseRegionRequest {
  MessageHeader header;
  uint64_t session_id;
};

struct StatusResponse {
  MessageHeader header;
  int32_t result;
  uint32_t reserved;
};
#pragma pack(pop)

static_assert(sizeof(MessageHeader) == 12U, "wire header ABI changed");
static_assert(sizeof(AllocateRegionRequest) == 36U, "allocate request ABI changed");
static_assert(sizeof(CandidateCountRequest) == 12U, "candidate count request ABI changed");
static_assert(sizeof(CandidateCountResponse) == 20U, "candidate count response ABI changed");
static_assert(sizeof(CreateChannelRequest) == 124U, "channel request ABI changed");
static_assert(sizeof(CreateChannelResponse) == 100U, "channel response ABI changed");
static_assert(sizeof(AllocateHbmRegionRequest) == 36U, "HBM allocate request ABI changed");
static_assert(sizeof(AllocateHbmRegionResponse) == 204U, "HBM allocate response ABI changed");
static_assert(sizeof(ReleaseHbmRegionRequest) == 20U, "HBM release request ABI changed");
static_assert(sizeof(HbmEidCandidateRequest) == 20U, "HBM candidate request ABI changed");
static_assert(sizeof(HbmEidCandidateResponse) == 40U, "HBM candidate response ABI changed");

inline constexpr uint16_t ResponseCommand(Command command) {
  return static_cast<uint16_t>(command) | 0x8000U;
}

inline constexpr MessageHeader MakeHeader(Command command, uint32_t bytes) {
  return MessageHeader{kProtocolMagic, kProtocolVersion,
                       static_cast<uint16_t>(command), bytes};
}

inline bool IsValidHeader(const MessageHeader& header, Command command,
                          std::size_t expected_bytes) {
  return header.magic == kProtocolMagic &&
         header.version == kProtocolVersion &&
         header.command == static_cast<uint16_t>(command) &&
         header.bytes == expected_bytes;
}

inline bool IsValidResponseHeader(const MessageHeader& header, Command command,
                                  std::size_t expected_bytes) {
  return header.magic == kProtocolMagic &&
         header.version == kProtocolVersion &&
         header.command == ResponseCommand(command) &&
         header.bytes == expected_bytes;
}

}  // namespace kvaf::peer

#endif  // KVAF_RUNTIME_PEER_PROTOCOL_H_
