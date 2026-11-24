#ifndef KVAF_RUNTIME_PEER_PROTOCOL_H_
#define KVAF_RUNTIME_PEER_PROTOCOL_H_

#include <cstddef>
#include <cstdint>

#include "cpu_push_get_abi.h"

// This protocol carries control-plane records between two Runtime peers. The
// first deployment is CPU DDR served to an Ascend NPU, but the records carry
// no role-specific wire semantics: data continues over UB.
namespace kvaf::peer {

constexpr uint32_t kProtocolMagic = 0x4b564350U;  // "KVCP"
// Version 6 exposes candidate metadata before allocation so adaptive startup
// can group EIDs by K-side UB device and benchmark each device only once.
constexpr uint16_t kProtocolVersion = 6U;
constexpr uint16_t kAivCpuPushProtocolVersion = 8U;
constexpr uint16_t kAivCpuPushJettyProtocolVersion = 9U;
constexpr uint32_t kQpKeyBytes = 64U;
constexpr uint32_t kSegKeyBytes = 64U;
constexpr uint32_t kEidBytes = 16U;
constexpr uint32_t kMaxGatherSpans = 65536U;
constexpr uint32_t kChannelSharedTp = 1U;
constexpr uint32_t kDeviceNameBytes = 64U;

enum class Command : uint16_t {
  kAllocateRegion = 1U,
  kCreateChannel = 2U,
  kReleaseRegion = 3U,
  kGetCandidateCount = 4U,
  kEnableDirectGather = 9U,
  kEnableGatherLanes = 10U,
  kGetCandidateInfo = 5U,
  kOpenCpuPushGet = 11U,
  kStartCpuPushGet = 12U,
  kStopCpuPushGet = 13U,
  kPrepareGatherWorkspace = 14U,
  kGetCacheCandidateCount = 15U,  // Service-affinity view used by ordinary cache providers.
  kStopGatherLanes = 16U,
  kOpenAivCpuGatherPushGet = 17U,
  kEnableAivGatherLanes = 18U,
  kAllocateUbmemRegion = 6U,
  // Command IDs 7 and 8 belonged to removed test-only fill/check requests.
};

inline constexpr bool IsAivCpuPushCommand(Command command) {
  return command == Command::kOpenCpuPushGet || command == Command::kStartCpuPushGet ||
         command == Command::kStopCpuPushGet || command == Command::kOpenAivCpuGatherPushGet;
}

inline constexpr bool IsSupportedProtocolVersion(Command command, uint16_t version) {
  return (version == kProtocolVersion && command != Command::kOpenAivCpuGatherPushGet) ||
         (version == kAivCpuPushProtocolVersion && IsAivCpuPushCommand(command)) ||
         (version == kAivCpuPushJettyProtocolVersion &&
          (command == Command::kOpenCpuPushGet || command == Command::kOpenAivCpuGatherPushGet));
}

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
  uint32_t device_count;
  uint32_t device_filter_applied;
};

struct CandidateInfoRequest {
  MessageHeader header;
  uint32_t candidate_id;
};

// Extended readers request the raw CPU EID using the same v6 command.
struct CandidateEidRequest {
  CandidateInfoRequest info;
  uint32_t response_bytes;
};

struct CandidateInfoResponse {
  MessageHeader header;
  int32_t result;
  uint32_t candidate_id;
  uint32_t device_id;
  uint32_t eid_index;
  char device_name[kDeviceNameBytes];
};

struct CandidateEidResponse {
  CandidateInfoResponse info;
  uint8_t eid[kEidBytes];
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

// The workspace belongs to this session; its registration is independent of
// the cache allocation. Replies use AllocateRegionResponse with the same
// session_id and the workspace's registered base/length/key.
struct PrepareGatherWorkspaceRequest {
  MessageHeader header;
  uint64_t session_id;
  uint64_t bytes;
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

// UBmem allocation shares the service and session namespace with URMA.
// Existing URMA v6 records and entry points stay usable.
struct AllocateUbmemRegionRequest {
  MessageHeader header;
  uint64_t ddr_bytes;
  uint64_t slot_size;
};

struct AllocateUbmemRegionResponse {
  MessageHeader header;
  int32_t result;
  uint64_t session_id;
  uint64_t peer_base;
  uint64_t peer_length;
  uint64_t mapping_bytes;
  uint8_t fabric_handle[128];
  char error_message[256];
};

struct StatusResponse {
  MessageHeader header;
  int32_t result;
  uint32_t reserved;
};
struct EnableDirectGatherRequest {
  MessageHeader header;
  uint64_t session_id;
};
struct EnableGatherLanesRequest {
  MessageHeader header;
  uint64_t session_id;
  uint64_t control_address;
  uint32_t lanes;
};
// CPU-initiated transfers target Host-registered HBM. Setup completes before
// the separate start command permits the CPU worker to issue any DMA.
struct CpuPushChannelKey {
  uint32_t bytes;
  uint8_t key[kQpKeyBytes];
};
struct OpenCpuPushGetRequest {
  MessageHeader header;
  uint64_t session_id, source_bytes, hbm_base, hbm_bytes, npu_tp;
  uint32_t npu_psn, token_value, segment_bytes, token_bytes, workers;
  uint8_t eid[kEidBytes], segment[kSegKeyBytes];
  CpuPushChannelKey channels[4];
};
struct OpenCpuPushGetResponse {
  MessageHeader header;
  int32_t result;
  uint64_t cpu_tp;
  uint32_t cpu_psn;
  CpuPushChannelKey channels[4];
};
struct OpenAivCpuPushGetResponse {
  MessageHeader header;
  int32_t result;
  uint64_t cpu_tp;
  uint32_t cpu_psn;
  CpuPushChannelKey channels[4];
  uint64_t mailbox_base, mailbox_bytes;
  uint32_t mailbox_token_id, mailbox_segment_bytes, max_write_bytes;
  uint8_t mailbox_segment[kSegKeyBytes];
};
// The extended open records retain their legacy wire prefixes. START and
// STOP use the same AIV command version for every queue count.
struct OpenAivCpuPushJettyRequest {
  OpenCpuPushGetRequest base;
  uint32_t payload_jettys;
  CpuPushChannelKey extra_channels[cpu_push::kMaxJettyCount - cpu_push::kJettyCount];
  const CpuPushChannelKey& Channel(uint32_t q) const {
    return q < cpu_push::kJettyCount ? base.channels[q] : extra_channels[q - cpu_push::kJettyCount];
  }
  CpuPushChannelKey& Channel(uint32_t q) {
    return q < cpu_push::kJettyCount ? base.channels[q] : extra_channels[q - cpu_push::kJettyCount];
  }
};
struct OpenAivCpuPushJettyResponse {
  OpenAivCpuPushGetResponse base;
  CpuPushChannelKey extra_channels[cpu_push::kMaxJettyCount - cpu_push::kJettyCount];
  CpuPushChannelKey& Channel(uint32_t q) {
    return q < cpu_push::kJettyCount ? base.channels[q] : extra_channels[q - cpu_push::kJettyCount];
  }
};
struct CpuPushGetSessionRequest {
  MessageHeader header;
  uint64_t session_id;
};
#pragma pack(pop)

static_assert(sizeof(AllocateUbmemRegionRequest) == 28U, "UBmem allocation ABI changed");
static_assert(sizeof(AllocateUbmemRegionResponse) == 432U, "UBmem descriptor ABI changed");

static_assert(sizeof(MessageHeader) == 12U, "wire header ABI changed");
static_assert(sizeof(AllocateRegionRequest) == 36U, "allocate request ABI changed");
static_assert(sizeof(PrepareGatherWorkspaceRequest) == 28U, "Gather workspace request ABI changed");
static_assert(sizeof(CandidateCountRequest) == 12U, "candidate count request ABI changed");
static_assert(sizeof(CandidateCountResponse) == 28U, "candidate count response ABI changed");
static_assert(sizeof(CandidateInfoRequest) == 16U, "candidate info request ABI changed");
static_assert(sizeof(CandidateEidRequest) == 20U && sizeof(CandidateEidResponse) == 108U);
static_assert(sizeof(CandidateInfoResponse) == 92U, "candidate info response ABI changed");
static_assert(sizeof(CreateChannelRequest) == 124U, "channel request ABI changed");
static_assert(sizeof(CreateChannelResponse) == 100U, "channel response ABI changed");
static_assert(sizeof(EnableGatherLanesRequest) == 32U, "gather lanes request ABI changed");
static_assert(sizeof(OpenCpuPushGetRequest) == 424U, "CPU push setup ABI changed");
static_assert(sizeof(OpenCpuPushGetResponse) == 300U, "CPU push setup response ABI changed");
static_assert(sizeof(OpenAivCpuPushGetResponse) == 392U, "AIV CPU push setup response ABI changed");
static_assert(sizeof(OpenAivCpuPushJettyRequest) == 2332U && sizeof(OpenAivCpuPushJettyResponse) == 2296U,
              "AIV CPU push Jetty setup ABI changed");
static_assert(sizeof(CpuPushGetSessionRequest) == 20U, "CPU push session ABI changed");

inline constexpr uint16_t ResponseCommand(Command command) {
  return static_cast<uint16_t>(command) | 0x8000U;
}

inline constexpr MessageHeader MakeHeader(Command command, uint32_t bytes,
                                          uint16_t version = kProtocolVersion) {
  return MessageHeader{kProtocolMagic, version,
                       static_cast<uint16_t>(command), bytes};
}

inline bool IsValidHeader(const MessageHeader& header, Command command,
                          std::size_t expected_bytes, uint16_t version = kProtocolVersion) {
  return header.magic == kProtocolMagic &&
         header.version == version &&
         header.command == static_cast<uint16_t>(command) &&
         header.bytes == expected_bytes;
}

inline bool IsValidResponseHeader(const MessageHeader& header, Command command,
                                  std::size_t expected_bytes, uint16_t version = kProtocolVersion) {
  return header.magic == kProtocolMagic &&
         header.version == version &&
         header.command == ResponseCommand(command) &&
         header.bytes == expected_bytes;
}

}  // namespace kvaf::peer

#endif  // KVAF_RUNTIME_PEER_PROTOCOL_H_
