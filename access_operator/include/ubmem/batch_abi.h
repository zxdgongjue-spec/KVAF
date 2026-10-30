#pragma once
#include <cstdint>

namespace kvaf::ubmem {
constexpr std::uint32_t kMteBufferBytes = 32 * 1024;
constexpr std::uint64_t kProfileMagic = 0x55424d454d505246ULL;
enum class Operation : std::uint32_t { kGet = 0, kPut = 1 };
// Separate each core's status stores, as in the existing BatchGet status ABI.
// The host and kernel are built together.
struct alignas(64) CoreStatus {
  std::uint64_t result, failed_request, completed;
  std::uint64_t profile_magic, start_cycle, finish_cycle;
  std::uint64_t reserved[2];
};
struct BatchArgs {
  std::uint64_t remote_base, remote_bytes, mapped_base;
  std::uint64_t local_base, local_bytes;
  std::uint64_t addresses, valid, indices, offsets, output_indices, miss_count;
  std::uint64_t put_offsets, put_lengths, put_destinations;
  std::uint64_t statuses;
  Operation operation;
  std::uint32_t blocks, capacity, table_entries, token_bytes, tokens_per_block;
  std::uint32_t profile;
};
static_assert(sizeof(CoreStatus) == 64);
static_assert(sizeof(BatchArgs) == 152);
}  // namespace kvaf::ubmem
