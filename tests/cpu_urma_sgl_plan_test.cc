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
}  // namespace

int main() {
  try {
    ContiguousAndTail();
    FragmentedAndScatterSources();
    LengthAndRangeLimits();
    std::cout << "CPU URMA SGL planning tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
