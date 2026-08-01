#pragma once

#include <cstdint>

#include "kvaf/access_operator/aiv_simt_urma_h2d.h"

namespace kvaf::access::a5 {

constexpr std::uint32_t kAicpuOwnerProbeAivResultMagic = 0x4b415650U;
constexpr std::uint32_t kAicpuOwnerProbeAivResultAbiVersion = 1U;

enum class AicpuOwnerProbeAivStatus : std::uint32_t {
  kEmpty = 0,
  kReleaseSubmitted = 1,
  kReadyTimedOut = 2,
  kOwnerTerminalBeforeReady = 3,
  kInvalidOwnerStatus = 4,
};

// Written only by the resource-free AIV probe and copied to Host after its
// stream has synchronized.  Keep one cache line so the probe can explicitly
// clean the complete result before the Host observes it.
struct alignas(64) AicpuOwnerProbeAivResult {
  std::uint32_t magic{kAicpuOwnerProbeAivResultMagic};
  std::uint32_t abi_version{kAicpuOwnerProbeAivResultAbiVersion};
  std::uint32_t struct_size{64U};
  std::uint32_t status{
      static_cast<std::uint32_t>(AicpuOwnerProbeAivStatus::kEmpty)};
  std::uint64_t session_id{0};
  std::uint64_t observed_sequence{0};
  std::uint32_t observed_owner_state{0};
  std::uint32_t poll_count{0};
  std::uint64_t reserved[3]{};
};

static_assert(sizeof(AicpuOwnerProbeAivResult) == 64,
              "AICPU owner AIV probe result must occupy one cache line");

constexpr std::uint32_t kAicpuEpochOwnerAivResultMagic = 0x4b455641U;
constexpr std::uint32_t kAicpuEpochOwnerAivResultAbiVersion = 1U;

enum class AicpuEpochOwnerAivStatus : std::uint32_t {
  kEmpty = 0,
  kReadyObserved = 1,
  kShutdownSubmitted = 2,
  kOwnerTerminal = 3,
  kQueueNotQuiesced = 4,
  kTimedOut = 5,
  kInvalidOwnerStatus = 6,
};

// One AIV-produced cache line used for both READY and shutdown observations.
// The four queue counters are zero for the READY phase and carry the exact
// shutdown proof for the SHUTDOWN_QUIESCED phase.
struct alignas(64) AicpuEpochOwnerAivResult {
  std::uint32_t magic{kAicpuEpochOwnerAivResultMagic};
  std::uint32_t abi_version{kAicpuEpochOwnerAivResultAbiVersion};
  std::uint32_t struct_size{64U};
  std::uint32_t status{
      static_cast<std::uint32_t>(AicpuEpochOwnerAivStatus::kEmpty)};
  std::uint64_t epoch{0};
  std::uint64_t session_id{0};
  std::uint32_t observed_owner_state{0};
  std::uint32_t poll_count{0};
  std::uint32_t observed_sq_head{0};
  std::uint32_t observed_sq_tail{0};
  std::uint32_t observed_cq_tail{0};
  std::uint32_t observed_wqe_count{0};
  std::uint64_t observed_sequence{0};
};

static_assert(sizeof(AicpuEpochOwnerAivResult) == 64,
              "AICPU epoch-owner AIV result must occupy one cache line");

// stream is an aclrtStream kept as void* so the stable header does not expose
// ACL runtime types.
void LaunchAivSimtUrmaH2D(
    void* stream, const DeviceBatchGetParams* device_params,
    DeviceCompletion* device_completion,
    AivDirectUrmaChannel* direct_channel);

// Resource-free concurrency/coherency probe. command and status point to the
// 64-byte provider ABI records shared with one bounded AICPU invocation.
// Keeping those internal C ABI types out of this launcher header avoids
// exposing the provider implementation to normal access-operator callers.
void LaunchAicpuOwnerConcurrencyProbe(
    void* stream, void* command, const void* status,
    AicpuOwnerProbeAivResult* result, std::uint64_t session_id,
    std::uint32_t max_ready_polls);

// Control-plane kernels for the resource-owning AICPU epoch.  These launches
// are asynchronous; the caller synchronizes the control stream and validates
// AicpuEpochOwnerAivResult before exposing the data plane or joining owner
// teardown.
void LaunchAivEpochOwnerWaitReady(
    void* stream, const void* owner_status, const void* open_result,
    AicpuEpochOwnerAivResult* aiv_result, std::uint64_t epoch,
    std::uint64_t session_id, std::uint32_t max_polls);

void LaunchAivEpochOwnerShutdown(
    void* stream, void* command, const void* owner_status,
    AicpuEpochOwnerAivResult* aiv_result,
    std::uint64_t packed_sq_head_address,
    std::uint64_t sq_tail_address, std::uint64_t cq_tail_address,
    std::uint64_t epoch, std::uint64_t session_id,
    std::uint32_t max_polls);

}  // namespace kvaf::access::a5
