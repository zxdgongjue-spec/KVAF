#include "cpu_urma/sgl_batch_plan.h"
#include "cpu_urma/jetty_batch_dispatch.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <chrono>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using kvaf::CpuUrmaTransport;
using kvaf::cpu_urma_detail::PlanWriteScatterGroups;
using kvaf::cpu_urma_detail::SglGroup;

void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
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
    for (const std::size_t limit : {1U, 31U, 64U, 128U}) {
      std::fill(actual.begin(), actual.end(), 0xEE);
      std::vector<SglGroup> groups;
      Check(PlanWriteScatterGroups(buffers.data(), offsets.data(), count, limit,
                                    actual.size(), &groups).ok(), "WRITE scatter plan failed");
      Check(groups.size() == (count + limit - 1) / limit, "WRITE scatter WQE count");
      std::size_t next = 0;
      // Exercise group rebasing at a 1024-WQE queue boundary as transport does.
      for (std::size_t begin = 0; begin < groups.size(); begin += 1024) {
        const auto base_row = groups[begin].first;
        const auto* chunk_buffers = buffers.data() + base_row;
        const auto* chunk_offsets = offsets.data() + base_row;
        for (std::size_t g = begin; g < std::min(begin + 1024, groups.size()); ++g) {
          const auto& group = groups[g];
          Check(group.first == next && group.count == std::min<std::size_t>(limit, count - next),
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

void SingleSgeValidation() {
  std::array<std::uint8_t, 64> data{};
  std::vector<CpuUrmaTransport::HostSge> buffers(32769, {data.data(), 8});
  std::vector<std::uint64_t> offsets(buffers.size(), 16);
  const auto validate = [&] { return kvaf::cpu_urma_detail::ValidateBatchRows(
      buffers.data(), offsets.data(), buffers.size(), 64); };
  Check(validate().ok(), "valid single-SGE input");
  offsets.back() = 60;
  Check(!validate().ok(), "invalid row beyond first four-Jetty batch accepted");
  offsets.back() = 16;
  buffers.back().data = reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 2);
  Check(!validate().ok(), "local address overflow accepted");
  buffers.back().data = data.data();
  buffers.back().bytes = 0;
  Check(!validate().ok(), "zero row length accepted");
}

void JettyChunkSubmission() {
  for (const std::size_t count : {0U, 1U, 97U, 1024U}) {
    for (const std::size_t chunk : {0U, 1U, 32U, 1024U}) {
      std::vector<bool> ready(count, false);
      std::size_t submitted = 0;
      kvaf::cpu_urma_detail::DispatchJettyChunks(count, chunk,
          [&](std::size_t begin, std::size_t end) {
            Check(begin == submitted, "prepared next chunk before posting current chunk");
            for (auto i = begin; i < end; ++i) ready[i] = true;
          },
          [&](std::size_t begin, std::size_t end) {
            Check(begin == submitted, "submission order changed");
            for (auto i = begin; i < end; ++i) Check(ready[i], "posted unprepared WQE");
            if (end < count) Check(!ready[end], "prepared future chunks before current post");
            submitted = end;
            return true;
          });
      Check(submitted == count, "chunk tail lost");
    }
  }
  std::size_t prepared = 0, posts = 0;
  kvaf::cpu_urma_detail::DispatchJettyChunks(97, 32,
      [&](std::size_t, std::size_t end) { prepared = end; },
      [&](std::size_t, std::size_t) { return ++posts != 2; });
  Check(prepared == 64 && posts == 2, "continued after a failed submission");

  // Hold worker 0 in preparation until worker 1 has submitted. Each queue
  // must make progress without waiting for preparation of the other queue.
  std::promise<void> preparing, posted;
  auto preparing_future = preparing.get_future();
  auto posted_future = posted.get_future();
  auto first = std::async(std::launch::async, [&] {
    kvaf::cpu_urma_detail::DispatchJettyChunks(2, 1,
        [&](std::size_t begin, std::size_t) {
          if (begin == 0) {
            preparing.set_value();
            Check(posted_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                  "Jetty workers cannot progress independently");
          }
        }, [](std::size_t, std::size_t) { return true; });
  });
  auto second = std::async(std::launch::async, [&] {
    Check(preparing_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
          "first worker did not start");
    kvaf::cpu_urma_detail::DispatchJettyChunks(1, 1,
        [](std::size_t, std::size_t) {},
        [&](std::size_t, std::size_t) { posted.set_value(); return true; });
  });
  first.get();
  second.get();
}
}  // namespace

int main() {
  try {
    PackedWriteScatter();
    WriteScatterLimits();
    SingleSgeValidation();
    JettyChunkSubmission();
    std::cout << "CPU URMA SGL planning tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
