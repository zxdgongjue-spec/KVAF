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

}  // namespace kvaf::access::a5
