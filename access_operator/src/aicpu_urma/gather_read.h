#pragma once

#include <cstdint>

namespace kvaf::aicpu_urma {

constexpr std::uint64_t kGatherStatusMagic = 0x4b56414647415448ULL;
constexpr std::uint32_t kGatherScratchBytes = 64U;

struct GatherReadStatus {
  std::uint64_t magic;
  std::uint64_t result;
  std::uint64_t phase;
  std::uint64_t ack_seen;
  std::uint64_t sq_producer_index;
  std::uint64_t cq_consumer_index;
  std::uint64_t cqe_word;
  std::uint64_t sequence;
};

struct GatherReadArgs {
  std::uint64_t channel_entity;
  std::uint64_t queue_state;
  std::uint64_t core_status;
  std::uint64_t output;
  std::uint64_t remote_staging;
  std::uint64_t remote_command;
  std::uint64_t remote_notify;
  std::uint64_t remote_ack;
  std::uint64_t sequence;
  std::uint32_t bytes;
  std::uint32_t reserved;
};

static_assert(sizeof(GatherReadStatus) == 64U, "GatherRead status ABI changed");
static_assert(sizeof(GatherReadArgs) == 80U, "GatherRead argument ABI changed");

}  // namespace kvaf::aicpu_urma
