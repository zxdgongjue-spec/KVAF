#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "kvaf/status.h"
#include "kvaf/types.h"

namespace kvaf::access {

inline constexpr std::uint32_t kAivUrmaAbiVersion = 1;
inline constexpr std::uint32_t kAivUrmaSimtLanes = 32;
inline constexpr std::uint32_t kAivUrmaMaxDescriptors = 32;
inline constexpr std::uint64_t kAivUrmaMaxTransferBytes =
    std::numeric_limits<std::uint32_t>::max();
inline constexpr std::uint32_t kNoFailedIndex =
    std::numeric_limits<std::uint32_t>::max();

enum class TransferState : std::uint32_t {
  kEmpty = 0,
  kReady,
  kSkipped,
  kSubmitted,
  kComplete,
  kFailed,
};

enum class DeviceBatchState : std::uint32_t {
  kEmpty = 0,
  kPlanning,
  kPlanned,
  kSubmitting,
  kComplete,
  kFailed,
};

enum class DeviceBatchStatus : std::uint32_t {
  kOk = 0,
  kAbiMismatch,
  kGenerationMismatch,
  kNullAddress,
  kCountOutOfRange,
  kIndexOutOfRange,
  kAccessTableEntryInvalid,
  kRemoteRangeInvalid,
  kOutputRangeInvalid,
  kDescriptorInvalid,
  kHcommInitFailed,
  kHcommReadFailed,
  kHcommCommitFailed,
  kHcommDrainFailed,
};

// Stable GM ABI shared by the CPU reference planner, the SIMT descriptor
// producer, and the scalar AIV URMA queue owner.
struct alignas(64) UrmaH2DDescriptor {
  std::uint32_t abi_version{kAivUrmaAbiVersion};
  std::uint32_t state{static_cast<std::uint32_t>(TransferState::kEmpty)};
  std::uint32_t logical_index{0};
  std::uint32_t flags{0};
  Address remote_addr{0};
  Address local_addr{0};
  std::uint64_t bytes{0};
  std::uint64_t generation{0};
  std::uint32_t status{static_cast<std::uint32_t>(DeviceBatchStatus::kOk)};
  std::uint32_t reserved0{0};
  std::uint64_t user_tag{0};
};

static_assert(sizeof(UrmaH2DDescriptor) == 64,
              "URMA H2D descriptor ABI must remain one cache line");
static_assert(alignof(UrmaH2DDescriptor) == 64,
              "URMA H2D descriptor ABI must remain cache-line aligned");

struct alignas(64) DeviceAccessTableView {
  // Device addresses of arrays: imported remote address, byte length, validity.
  std::uint64_t remote_addresses{0};
  std::uint64_t lengths{0};
  std::uint64_t valid{0};
  std::uint32_t capacity{0};
  std::uint32_t abi_version{kAivUrmaAbiVersion};
  std::uint64_t generation{0};
  std::uint64_t reserved[3]{};
};

static_assert(sizeof(DeviceAccessTableView) == 64,
              "AccessTable device ABI must remain 64 bytes");

struct alignas(64) DeviceBatchGetParams {
  DeviceAccessTableView table{};
  std::uint64_t indices{0};
  std::uint64_t mask{0};
  std::uint64_t output_offsets{0};
  std::uint64_t descriptors{0};
  std::uint64_t output_base{0};
  std::uint64_t output_bytes{0};
  std::uint32_t entry_count{0};
  std::uint32_t descriptor_capacity{0};
  std::uint64_t generation{0};
};

static_assert(sizeof(DeviceBatchGetParams) == 128,
              "BatchGet device ABI must remain 128 bytes");

struct alignas(64) DeviceCompletion {
  std::uint32_t abi_version{kAivUrmaAbiVersion};
  std::uint32_t state{static_cast<std::uint32_t>(DeviceBatchState::kEmpty)};
  std::uint32_t status{static_cast<std::uint32_t>(DeviceBatchStatus::kOk)};
  std::uint32_t planned_count{0};
  std::uint32_t submitted_count{0};
  std::uint32_t completed_count{0};
  std::uint32_t first_failed_index{kNoFailedIndex};
  std::uint32_t reserved0{0};
  std::uint64_t generation{0};
  std::uint64_t reserved[3]{};
};

static_assert(sizeof(DeviceCompletion) == 64,
              "Completion device ABI must remain 64 bytes");

// Host-side view used by PC-A unit tests and by host code preparing a captured
// device graph. output_offsets are byte offsets into one contiguous HBM output.
struct H2DPlanView {
  const Address* remote_addresses{nullptr};
  const std::uint64_t* lengths{nullptr};
  const std::uint8_t* valid{nullptr};
  std::uint32_t table_capacity{0};
  std::uint64_t table_generation{0};

  const std::uint32_t* indices{nullptr};
  const std::uint8_t* mask{nullptr};
  const std::uint64_t* output_offsets{nullptr};
  std::uint32_t entry_count{0};

  Address output_base{0};
  std::uint64_t output_bytes{0};
  std::uint64_t generation{0};

  UrmaH2DDescriptor* descriptors{nullptr};
  std::uint32_t descriptor_capacity{0};
  DeviceCompletion* completion{nullptr};
};

// Produces all descriptors or fails the whole batch before any descriptor may
// be submitted. A null mask means every request is active; a null valid array
// means every AccessTable row is valid.
Status BuildH2DTransferPlan(const H2DPlanView& view);

}  // namespace kvaf::access
