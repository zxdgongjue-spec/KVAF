#include "cpu_urma/sgl_batch_plan.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using kvaf::CpuUrmaTransport;
using kvaf::cpu_urma_detail::PlanSglGroups;
using kvaf::cpu_urma_detail::PlanWriteScatterGroups;
using kvaf::cpu_urma_detail::SglGroup;

void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

void ContiguousAndTail() {
  std::uint8_t byte = 0;
  for (const auto token : {128U, 256U, 512U, 656U, 1024U, 2048U, 4096U}) {
    for (const auto count : {1U, 7U, 32U, 1024U, 8192U, 32769U}) {
      std::vector<CpuUrmaTransport::HostSge> buffers(count, {&byte, token});
      std::vector<std::uint64_t> offsets(count);
      for (std::size_t i = 0; i < count; ++i) offsets[i] = i * token;
      for (const auto limit : {1U, 2U, 3U, 8U, 16U}) {
        std::vector<SglGroup> groups;
        Check(PlanSglGroups(buffers.data(), offsets.data(), count, limit,
                            static_cast<std::uint64_t>(count) * token, &groups).ok(),
              "contiguous plan failed");
        Check(groups.size() == (count + limit - 1U) / limit, "incorrect WR count");
        std::size_t next = 0;
        for (const auto& group : groups) {
          Check(group.first == next, "dropped or duplicated rows");
          Check(group.count == std::min<std::size_t>(limit, count - next), "tail size");
          Check(group.bytes == group.count * token, "incorrect remote SGE length");
          next += group.count;
        }
        Check(next == count, "not all rows covered");
        if (count == 32769U && limit == 8U) {
          Check(groups.size() == 4097U && groups[4096].first == 32768U,
                "queue-capacity boundary must use WRs rather than rows");
        }
      }
    }
  }
}

void FragmentedAndScatterSources() {
  // Includes ascending runs, a gap, backward address movement, and variable
  // lengths. The extra invalid suffix must not be consumed by a prefix plan.
  const std::array<std::uint64_t, 8> offsets{16, 19, 40, 42, 0, 4, 7, 999};
  const std::array<std::size_t, 8> lengths{3, 5, 2, 6, 4, 3, 1, 0};
  std::array<std::array<std::uint8_t, 12>, 8> sources{};
  std::array<CpuUrmaTransport::HostSge, 8> buffers{};
  for (std::size_t i = 0; i < 7; ++i) {
    sources[i].fill(static_cast<std::uint8_t>(i + 1));
    buffers[i] = {sources[i].data() + 1, lengths[i]};
  }
  std::vector<SglGroup> groups;
  Check(PlanSglGroups(buffers.data(), offsets.data(), 7, 8, 64, &groups).ok(),
        "fragmented plan failed");
  Check(groups.size() == 3 && groups[0].count == 2 && groups[1].count == 2 &&
        groups[2].count == 3, "remote gaps must split groups");
  std::array<std::uint8_t, 64> expected, actual;
  expected.fill(0xEE);
  actual.fill(0xEE);
  for (std::size_t i = 0; i < 7; ++i)
    std::memcpy(expected.data() + offsets[i], buffers[i].data, lengths[i]);
  // Emulate the hardware's local-SGL -> single remote-range gather semantics.
  for (const auto& group : groups) {
    auto destination = offsets[group.first];
    for (std::size_t i = group.first; i < group.first + group.count; ++i) {
      std::memcpy(actual.data() + destination, buffers[i].data, buffers[i].bytes);
      destination += buffers[i].bytes;
    }
    Check(destination == offsets[group.first] + group.bytes, "remote range length");
  }
  Check(actual == expected, "gather changed data or overwrote a remote gap");
  // READ uses the same grouping with the local SGL as the destination.
  for (const auto& group : groups) {
    auto source = offsets[group.first];
    for (std::size_t i = group.first; i < group.first + group.count; ++i) {
      Check(std::memcmp(actual.data() + source, buffers[i].data, buffers[i].bytes) == 0,
            "READ scatter would use the wrong remote bytes");
      source += buffers[i].bytes;
    }
  }
  Check(!PlanSglGroups(buffers.data(), offsets.data(), 8, 8, 64, &groups).ok(),
        "invalid active row accepted");
  Check(groups.empty(), "failed plan left partial WRs");
}

void LengthAndRangeLimits() {
  std::uint8_t byte = 0;
  constexpr std::uint64_t max_length = std::numeric_limits<std::uint32_t>::max();
  // Planning never dereferences these synthetic large buffers.
  CpuUrmaTransport::HostSge buffers[]{{&byte, max_length - 1}, {&byte, 2}};
  std::uint64_t offsets[]{0, max_length - 1};
  std::vector<SglGroup> groups;
  Check(PlanSglGroups(buffers, offsets, 2, 8, max_length + 1, &groups).ok(), "large rows");
  Check(groups.size() == 2, "aggregate remote length overflow was not split");
  Check(!PlanSglGroups(buffers, offsets, 2, 8, max_length, &groups).ok(), "remote overflow");
  offsets[0] = std::numeric_limits<std::uint64_t>::max() - 1;
  Check(!PlanSglGroups(buffers, offsets, 1, 8,
                       std::numeric_limits<std::uint64_t>::max(), &groups).ok(),
        "remote address addition overflow");
  offsets[0] = 0;
  buffers[0].bytes = max_length + 1;
  Check(!PlanSglGroups(buffers, offsets, 1, 8, max_length + 1, &groups).ok(),
        "oversized individual row accepted");
  Check(!PlanSglGroups(buffers, offsets, 1, 0, max_length + 1, &groups).ok(), "zero SGL");
}

void PackedWriteScatter() {
  // More than 13 rows must fit in one WQE: only one local SGE is needed.
  for (const std::size_t count : {1U, 13U, 31U, 32U, 3000U, 6000U, 32769U}) {
    std::vector<std::uint8_t> source(count * 8);
    std::vector<CpuUrmaTransport::HostSge> buffers(count);
    std::vector<std::uint64_t> offsets(count);
    std::vector<std::uint8_t> expected(count * 16, 0xEE), actual = expected;
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < count; ++i) {
      const std::size_t bytes = 1 + i % 8;
      // Reversed destinations with gaps exercise independent remote SGEs.
      offsets[i] = (count - 1 - i) * 16;
      buffers[i] = {source.data() + cursor, bytes};
      std::fill_n(source.data() + cursor, bytes, static_cast<std::uint8_t>(i % 197));
      std::memcpy(expected.data() + offsets[i], buffers[i].data, bytes);
      cursor += bytes;
    }
    std::vector<SglGroup> groups;
    Check(PlanWriteScatterGroups(buffers.data(), offsets.data(), count, 31,
                                  actual.size(), &groups).ok(), "WRITE scatter plan failed");
    Check(groups.size() == (count + 30) / 31, "WRITE scatter WQE count");
    std::size_t next = 0;
    // Exercise group rebasing at a 1024-WQE queue boundary as transport does.
    for (std::size_t begin = 0; begin < groups.size(); begin += 1024) {
      const auto base_row = groups[begin].first;
      const auto* chunk_buffers = buffers.data() + base_row;
      const auto* chunk_offsets = offsets.data() + base_row;
      for (std::size_t g = begin; g < std::min(begin + 1024, groups.size()); ++g) {
        const auto& group = groups[g];
        Check(group.first == next && group.count == std::min<std::size_t>(31, count - next),
              "WRITE scatter rows missing or duplicated");
        const auto first = group.first - base_row;
        const auto* local = static_cast<const std::uint8_t*>(chunk_buffers[first].data);
        std::size_t consumed = 0;
        for (std::size_t j = 0; j < group.count; ++j) {
          const auto bytes = chunk_buffers[first + j].bytes;
          std::memcpy(actual.data() + chunk_offsets[first + j], local + consumed, bytes);
          consumed += bytes;
        }
        Check(consumed == group.bytes, "WRITE source/destination length mismatch");
        next += group.count;
      }
    }
    Check(next == count && actual == expected, "WRITE scatter data or remote gaps corrupted");
  }
}

void WriteScatterLimits() {
  std::array<std::uint8_t, 64> source{};
  CpuUrmaTransport::HostSge buffers[]{{source.data(), 3}, {source.data() + 3, 5},
                                     {source.data() + 20, 2}, {source.data() + 22, 6}};
  std::uint64_t offsets[]{40, 0, 30, 8};
  std::vector<SglGroup> groups;
  Check(PlanWriteScatterGroups(buffers, offsets, 4, 31, 64, &groups).ok(), "local gap plan");
  Check(groups.size() == 2 && groups[0].count == 2 && groups[1].count == 2,
        "noncontiguous local spans must split WRITE WQEs");
  Check(PlanWriteScatterGroups(buffers, offsets, 4, 1, 64, &groups).ok() && groups.size() == 4,
        "remote SGE limit not enforced");
  Check(!PlanWriteScatterGroups(buffers, offsets, 4, 0, 64, &groups).ok(), "zero remote SGL");
  offsets[3] = 60;
  Check(!PlanWriteScatterGroups(buffers, offsets, 4, 31, 64, &groups).ok() && groups.empty(),
        "remote overflow left a partial scatter plan");
  offsets[3] = 8;
  buffers[3].data = reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 2);
  Check(!PlanWriteScatterGroups(buffers, offsets, 4, 31, 64, &groups).ok(), "local overflow");

  // Synthetic addresses are used only for planning, never dereferenced.
  constexpr std::uint64_t max_length = std::numeric_limits<std::uint32_t>::max();
  CpuUrmaTransport::HostSge large[]{{reinterpret_cast<void*>(4096), max_length - 1},
      {reinterpret_cast<void*>(4096 + max_length - 1), 2}};
  std::uint64_t large_offsets[]{0, max_length - 1};
  Check(PlanWriteScatterGroups(large, large_offsets, 2, 31, max_length + 1, &groups).ok() &&
        groups.size() == 2, "single local SGE length overflow was not split");
  large[0].bytes = 0;
  Check(!PlanWriteScatterGroups(large, large_offsets, 2, 31, max_length + 1, &groups).ok(),
        "zero length WRITE row accepted");
}
}  // namespace

int main() {
  try {
    ContiguousAndTail();
    FragmentedAndScatterSources();
    LengthAndRangeLimits();
    PackedWriteScatter();
    WriteScatterLimits();
    std::cout << "CPU URMA SGL planning tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
