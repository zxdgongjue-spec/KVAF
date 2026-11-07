#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "../kernel/batch_get.asc"
#include "../kernel/batch_put.asc"

using namespace kvaf_ubmem;
uint64_t Address(const void* pointer) { return reinterpret_cast<uintptr_t>(pointer); }

void RunGet(GetArgs args, CoreStatus* statuses) {
  args.status = Address(statuses);
  std::fill(statuses, statuses + args.workers, CoreStatus{});
  for (uint32_t worker = 0; worker < args.workers; ++worker) {
    ubmem_model::core = worker;
    KvafUbmemBatchGet(args);
    ubmem_model::CheckDrained();
    assert(statuses[worker].magic == kStatusMagic);
  }
}

void TestGet(uint32_t width, uint32_t workers) {
  constexpr uint32_t rows = 513, block_rows = 3, blocks = 6;
  const uint64_t slot = uint64_t(width) * block_rows;
  std::vector<uint8_t> arena(slot * blocks);
  for (size_t i = 0; i < arena.size(); ++i) arena[i] = (i * 29 + i / 7) % 251;
  std::array<uint64_t, blocks> addresses{};
  std::array<uint8_t, blocks> valid{};
  for (uint32_t i = 0; i < blocks; ++i) { addresses[i] = Address(arena.data()) + i * slot; valid[i] = 1; }
  std::array<int32_t, rows + 2> indices{}, offsets{}, output_rows{};
  for (uint32_t i = 0; i < rows; ++i) {
    indices[i] = i % blocks;
    offsets[i] = (i % block_rows) * width;
    output_rows[i] = (i * 7) % rows;
  }
  // Repeated destinations must execute on one owner in prefix order.
  indices[rows] = 2; offsets[rows] = width; output_rows[rows] = 3;
  // Unused capacity must never be inspected.
  indices[rows + 1] = -999; offsets[rows + 1] = -999; output_rows[rows + 1] = -999;
  int32_t count = rows + 1;
  const size_t bytes = rows * width;
  const size_t allocation = (bytes + 64 + 63) / 64 * 64;
  std::unique_ptr<uint8_t, decltype(&std::free)> storage(
      static_cast<uint8_t*>(std::aligned_alloc(64, allocation)), &std::free);
  assert(storage);
  std::memset(storage.get(), 0xa7, allocation);
  auto* output = storage.get() + 32;
  std::vector<uint8_t> expected(bytes, 0xa7);
  for (int32_t i = 0; i < count; ++i) {
    std::memcpy(expected.data() + output_rows[i] * width,
        arena.data() + indices[i] * slot + offsets[i], width);
  }
  GetArgs args{Address(addresses.data()), Address(valid.data()), Address(indices.data()),
      Address(offsets.data()), Address(&count), Address(output_rows.data()), Address(output), 0,
      Address(arena.data()), arena.size(), slot, blocks, rows + 2, rows,
      width, block_rows, workers, kAbiVersion, 0};
  std::array<CoreStatus, kMaxWorkers> statuses{};
  ubmem_model::TrackWrites(output, bytes);
  RunGet(args, statuses.data());
  uint64_t total = 0;
  for (uint32_t i = 0; i < workers; ++i) { assert(statuses[i].result == kSuccess); total += statuses[i].copied_requests; }
  assert(total == static_cast<uint32_t>(count));
  assert(std::memcmp(output, expected.data(), bytes) == 0);
  for (size_t i = 0; i < 32; ++i) assert(storage.get()[i] == 0xa7 && output[bytes + i] == 0xa7);
  const auto before = expected;
  count = 0;
  RunGet(args, statuses.data());
  assert(std::memcmp(output, before.data(), bytes) == 0);
  count = -1;
  RunGet(args, statuses.data());
  for (uint32_t i = 0; i < workers; ++i) assert(statuses[i].result == kInvalidCount);
  count = rows + 3;
  RunGet(args, statuses.data());
  for (uint32_t i = 0; i < workers; ++i) assert(statuses[i].result == kInvalidCount);
  count = 1;
  output_rows[0] = rows;
  RunGet(args, statuses.data());
  for (uint32_t i = 0; i < workers; ++i) assert(statuses[i].result == kInvalidDestination);
  output_rows[0] = 0;
  indices[0] = blocks;
  RunGet(args, statuses.data()); assert(statuses[0].result == kInvalidLookup);
  indices[0] = 0; valid[0] = 0;
  RunGet(args, statuses.data()); assert(statuses[0].result == kInvalidLookup);
  valid[0] = 1; offsets[0] = width * block_rows;
  RunGet(args, statuses.data()); assert(statuses[0].result == kInvalidSource);
  offsets[0] = 1;
  RunGet(args, statuses.data()); assert(statuses[0].result == kInvalidSource);
  offsets[0] = 0; addresses[0] = Address(arena.data()) + arena.size() - 1;
  RunGet(args, statuses.data()); assert(statuses[0].result == kInvalidSource);
  ubmem_model::TrackWrites(nullptr, 0);
}

void TestPut(uint64_t slot, uint32_t workers) {
  const std::vector<uint64_t> sizes{1, 15, 31, 32, 33, 655, 656, 2624, 8191, 8192, 8193, 13003, 128 * 656, 128 * 2624, 10 * 8192 + 17};
  std::vector<uint8_t> arena(slot * sizes.size() + 128, 0xb9), source(1000000, 0);
  for (size_t i = 0; i < source.size(); ++i) source[i] = (i * 17 + i / 11) % 253;
  std::vector<PutEntry> entries;
  const uint64_t prefix = 32 + (32 - Address(arena.data()) % 32) % 32;
  auto expected = arena;
  uint64_t offset = 3;
  for (size_t i = 0; i < sizes.size(); ++i) {
    entries.push_back({offset, sizes[i], Address(arena.data() + prefix + i * slot)});
    std::memcpy(expected.data() + prefix + i * slot, source.data() + offset, sizes[i]);
    offset += sizes[i] + 5;
  }
  // Repeat a destination with different bytes/length. Every chunk must
  // remain on its original owner and preserve submission order.
  entries.push_back({13, slot - 17, Address(arena.data() + prefix)});
  std::memcpy(expected.data() + prefix, source.data() + 13, slot - 17);
  std::array<CoreStatus, kMaxWorkers> statuses{};
  auto& status = statuses[0];
  PutArgs args{Address(source.data()), source.size(), Address(entries.data()), Address(&status),
      Address(arena.data() + prefix), slot * sizes.size(), slot, static_cast<uint32_t>(entries.size()), kAbiVersion};
  ubmem_model::core = 0;
  ubmem_model::TrackWrites(arena.data() + prefix, slot * sizes.size());
  ubmem_model::workers = workers;
  uint64_t total = 0;
  for (uint32_t i = 0; i < workers; ++i) {
    ubmem_model::core = i;
    KvafUbmemBatchPut(args);
    ubmem_model::CheckDrained();
    assert(statuses[i].magic == kStatusMagic && statuses[i].result == kSuccess);
    total += statuses[i].copied_requests;
  }
  assert(total == entries.size());
  ubmem_model::core = 0;
  assert(arena == expected);
  args.count = 1;
  entries[0].length = 0;
  KvafUbmemBatchPut(args); assert(status.result == kInvalidSource);
  entries[0].length = 2; entries[0].input_offset = source.size() - 1;
  KvafUbmemBatchPut(args); assert(status.result == kInvalidSource);
  entries[0].input_offset = 0; entries[0].destination = args.region_base + args.region_bytes - 1;
  KvafUbmemBatchPut(args); assert(status.result == kInvalidDestination);
  assert(arena == expected);
  ubmem_model::CheckDrained();
  ubmem_model::TrackWrites(nullptr, 0);
}

int main() {
  assert(!Contains(UINT64_MAX - 8, 16, UINT64_MAX - 8, 1));
  assert(!Contains(100, 64, 99, 1));
  assert(Contains(100, 64, 163, 1));
  assert(!Contains(100, 64, 163, 2));
  for (uint32_t width : {656U, 2624U}) {
    for (uint32_t workers : {1U, 2U, 3U, 8U, 64U}) TestGet(width, workers);
  }
  TestPut(335885, 1);
  for (uint32_t workers : {1U, 2U, 8U, 32U}) TestPut(335904, workers);
  std::cout << "UBmem actual-kernel CPU model PASS (bytes, tails, guards, sector ownership, invalid prefixes, event credits)\n";
}
