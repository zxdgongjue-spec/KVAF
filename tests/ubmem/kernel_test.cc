#include "kernel_operator.h"
#include "ubmem/batch_abi.h"
#include "ubmem/test_pattern.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define __attribute__(...)
#include "../../access_operator/src/ubmem/batch_kernel.asc"
#undef __attribute__
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)

using namespace kvaf::ubmem;
template <typename T> std::uint64_t Ptr(T* value) { return reinterpret_cast<std::uintptr_t>(value); }
void Run(BatchArgs args) {
  auto* reports = reinterpret_cast<CoreStatus*>(args.statuses);
  std::fill_n(reinterpret_cast<unsigned char*>(reports), args.blocks * sizeof(CoreStatus), 0xFF);
  for (model_core = 0; model_core < args.blocks; ++model_core) {
    KvafUbmemBatchMteKernel(args);
    CHECK(!model_mte2 && !model_mte3);
    CHECK(reports[model_core].capacity == args.capacity);
    CHECK(reports[model_core].blocks == args.blocks);
    CHECK(reports[model_core].operation == static_cast<std::uint32_t>(args.operation));
  }
}
int main() {
  constexpr std::uint64_t remote_base = 0x100000000ULL;
  std::vector<std::uint8_t> remote(32768, 0xCF), local(5000, 0xD7);
  for (std::size_t i = 0; i < remote.size(); ++i) remote[i] = i * 37 + 3;
  std::array<CoreStatus, 3> statuses{};
  std::uint64_t addresses[]{remote_base, remote_base + 8192};
  std::uint8_t valid[]{1, 1};
  std::int32_t indices[]{1, 0, 1, -999}, offsets[]{0, 17, 656, -999}, outputs[]{2, 0, 1, -999};
  std::int32_t count = 3;
  BatchArgs a{};
  a.remote_base = remote_base; a.remote_bytes = remote.size();
  a.mapped_base = Ptr(remote.data()); a.local_base = Ptr(local.data() + 1); a.local_bytes = 3 * 656;
  a.addresses = Ptr(addresses); a.valid = Ptr(valid); a.indices = Ptr(indices); a.offsets = Ptr(offsets);
  a.output_indices = Ptr(outputs); a.miss_count = Ptr(&count); a.statuses = Ptr(statuses.data());
  a.operation = Operation::kGet; a.blocks = 3; a.capacity = 4; a.table_entries = 2;
  a.token_bytes = 656; a.tokens_per_block = 8;
  Run(a);
  for (unsigned r = 0; r < 3; ++r) {
    CHECK(statuses[r].result == 0 && statuses[r].completed == 1);
    for (unsigned b = 0; b < 656; ++b)
      CHECK(local[1 + outputs[r] * 656 + b] == remote[addresses[indices[r]] - remote_base + offsets[r] + b]);
  }
  CHECK(local.front() == 0xD7 && local[1 + 3 * 656] == 0xD7);
  const auto previous = local;
  count = 0; Run(a);
  CHECK(local == previous);
  for (auto& s : statuses) CHECK(s.result == 0 && s.completed == 0);
  count = -1; Run(a);
  for (auto& s : statuses) CHECK(s.result == 1);
  CHECK(local == previous);
  count = 5; Run(a);
  for (auto& s : statuses) CHECK(s.result == 1);
  count = 3; valid[1] = 0; Run(a); CHECK(statuses[0].result == 2 && statuses[2].result == 2);
  valid[1] = 1; offsets[0] = 5248; Run(a); CHECK(statuses[0].result == 2);
  offsets[0] = 0; outputs[0] = 3; Run(a); CHECK(statuses[0].result == 3);
  outputs[0] = 2; addresses[1] = remote_base + remote.size() - 16; Run(a); CHECK(statuses[0].result == 2);

  // Validation is per row, not transactional: a bad later row leaves the
  // previous rows copied, stops its core, and never writes the invalid row.
  addresses[1] = remote_base + 8192;
  a.blocks = 1;
  outputs[1] = 3;
  std::fill(local.begin(), local.end(), 0xD7);
  Run(a);
  CHECK(statuses[0].result == 3 && statuses[0].failed_request == 1 && statuses[0].completed == 1);
  for (unsigned b = 0; b < 656; ++b) CHECK(local[1 + 2 * 656 + b] == remote[8192 + b]);
  CHECK(local[1] == 0xD7 && local[1 + 656] == 0xD7 && local[1 + 3 * 656] == 0xD7);
  outputs[1] = 0;
  a.blocks = 3;

  std::uint64_t put_offsets[]{1, 20, 680}, lengths[]{1, 655, 4097};
  std::uint64_t destinations[]{remote_base + 1, remote_base + 8193, remote_base + 16384};
  a.operation = Operation::kPut; a.capacity = 3; a.local_base = Ptr(local.data()); a.local_bytes = local.size();
  a.put_offsets = Ptr(put_offsets); a.put_lengths = Ptr(lengths); a.put_destinations = Ptr(destinations);
  Run(a);
  for (unsigned r = 0; r < 3; ++r) {
    CHECK(statuses[r].result == 0);
    for (unsigned b = 0; b < lengths[r]; ++b)
      CHECK(remote[destinations[r] - remote_base + b] == local[put_offsets[r] + b]);
  }
  // An aligned source with a non-32-byte tail must still copy exact bytes.
  put_offsets[2] = 0; destinations[2] = remote_base + 16384;
  Run(a);
  for (unsigned b = 0; b < lengths[2]; ++b) CHECK(remote[16384 + b] == local[b]);

  // Hardware failure shape: 17 reversed slots, 2 cores, 83968 bytes per row,
  // and an unaligned input with 13-byte gaps. Row 0 targets offset 1343488.
  constexpr unsigned rows = 17;
  constexpr std::uint64_t slot = 656 * 128;
  std::vector<std::uint8_t> big_remote(rows * slot + 32, 0xCF);
  std::vector<std::uint8_t> big_local(rows * (slot + 13) + 3, 0xEF);
  std::array<std::uint64_t, rows> src_offsets{}, dst_addresses{}, row_lengths{};
  for (unsigned row = 0; row < rows; ++row) {
    const auto key = rows - 1 - row;
    src_offsets[row] = row * (slot + 13) + 3;
    dst_addresses[row] = remote_base + key * slot;
    row_lengths[row] = slot;
    for (std::uint64_t b = 0; b < slot; ++b)
      big_local[src_offsets[row] + b] = TestPattern(key * slot + b, 17);
  }
  a.remote_bytes = rows * slot; a.mapped_base = Ptr(big_remote.data());
  a.local_base = Ptr(big_local.data()); a.local_bytes = big_local.size();
  a.blocks = 2; a.capacity = rows;
  a.put_offsets = Ptr(src_offsets.data()); a.put_lengths = Ptr(row_lengths.data());
  a.put_destinations = Ptr(dst_addresses.data());
  Run(a);
  CHECK(statuses[0].result == 0 && statuses[0].completed == 9);
  CHECK(statuses[1].result == 0 && statuses[1].completed == 8);
  for (std::uint64_t i = 0; i < rows * slot; ++i) CHECK(big_remote[i] == TestPattern(i, 17));
  for (std::uint64_t i = rows * slot; i < big_remote.size(); ++i) CHECK(big_remote[i] == 0xCF);

  // The local-HBM control experiment uses the same kernel with offset-based
  // destinations. It must produce exactly the same bytes as mapped addressing.
  a.remote_base = 0;
  for (auto& address : dst_addresses) address -= remote_base;
  std::fill(big_remote.begin(), big_remote.end(), 0xCF);
  Run(a);
  for (std::uint64_t i = 0; i < rows * slot; ++i) CHECK(big_remote[i] == TestPattern(i, 17));
  for (std::uint64_t i = rows * slot; i < big_remote.size(); ++i) CHECK(big_remote[i] == 0xCF);
  // Exact MTE tile boundaries and tails: padding must never reach GM guards.
  for (const std::uint64_t bytes : {kMteBufferBytes - 1U, kMteBufferBytes,
                                   kMteBufferBytes + 1U, 2 * kMteBufferBytes + 17U}) {
    std::vector<std::uint8_t> source(bytes + 1), destination(bytes + 7, 0xCF);
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = TestPattern(i, 53);
    std::uint64_t source_offset = 1, destination_offset = 3, length = bytes;
    a.local_base = Ptr(source.data()); a.local_bytes = source.size();
    a.mapped_base = Ptr(destination.data()); a.remote_base = 0; a.remote_bytes = destination.size();
    a.blocks = 1; a.capacity = 1;
    a.put_offsets = Ptr(&source_offset); a.put_lengths = Ptr(&length); a.put_destinations = Ptr(&destination_offset);
    Run(a);
    CHECK(statuses[0].result == 0 && statuses[0].completed == 1);
    for (std::uint64_t i = 0; i < bytes; ++i) CHECK(destination[3 + i] == source[1 + i]);
    for (unsigned i = 0; i < 3; ++i) CHECK(destination[i] == 0xCF);
    for (std::size_t i = 3 + bytes; i < destination.size(); ++i) CHECK(destination[i] == 0xCF);
  }
  std::puts("UBmem actual kernel: mapped addressing, 656-byte reads, variable puts, inactive tail, guards, and invalid ranges passed");
}
