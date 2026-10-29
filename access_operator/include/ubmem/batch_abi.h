#pragma once
#include <cstdint>

namespace kvaf::ubmem {
constexpr std::uint64_t kBatchMagic = 0x4b56414655424d31ULL;
enum class Operation : std::uint32_t { kGet = 0, kPut = 1 };
struct alignas(64) CoreStatus {
  std::uint64_t magic, result, failed_request, completed;
  std::uint64_t reserved[4];
};
struct BatchArgs {
  std::uint64_t magic;
  std::uint64_t remote_base, remote_bytes, mapped_base;
  std::uint64_t local_base, local_bytes;
  std::uint64_t addresses, valid, indices, offsets, output_indices, miss_count;
  std::uint64_t put_offsets, put_lengths, put_destinations;
  std::uint64_t statuses;
  Operation operation;
  std::uint32_t blocks, capacity, table_entries, token_bytes, tokens_per_block;
};
static_assert(sizeof(CoreStatus) == 64);
static_assert(sizeof(BatchArgs) == 152);
}  // namespace kvaf::ubmem
