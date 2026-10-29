#pragma once
#include <cstdint>

namespace kvaf::ubmem {
enum class Operation : std::uint32_t { kGet = 0, kPut = 1 };
// One scalar DCache line per core: publishing a report cannot write back
// another core's report. The host and kernel are built together.
struct alignas(64) CoreStatus {
  std::uint64_t result, failed_request, completed;
  std::uint64_t reserved[5];
};
struct BatchArgs {
  std::uint64_t remote_base, remote_bytes, mapped_base;
  std::uint64_t local_base, local_bytes;
  std::uint64_t addresses, valid, indices, offsets, output_indices, miss_count;
  std::uint64_t put_offsets, put_lengths, put_destinations;
  std::uint64_t statuses;
  Operation operation;
  std::uint32_t blocks, capacity, table_entries, token_bytes, tokens_per_block;
};
static_assert(sizeof(CoreStatus) == 64);
static_assert(sizeof(BatchArgs) == 144);
}  // namespace kvaf::ubmem
