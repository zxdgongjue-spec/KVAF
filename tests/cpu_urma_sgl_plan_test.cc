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
using kvaf::cpu_urma_detail::GroupsFitRegisteredRegion;
using kvaf::cpu_urma_detail::MergeSglPlanSlices;
using kvaf::cpu_urma_detail::PlanWriteCoalescedGroups;
using kvaf::cpu_urma_detail::PlanWriteCoalescedSlice;
using kvaf::cpu_urma_detail::PrepareSglPlanSlices;
using kvaf::cpu_urma_detail::SglGroup;
using kvaf::cpu_urma_detail::SglPlanSlice;
using kvaf::cpu_urma_detail::ValidateBatchRows;

void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

kvaf::Status PlanInSlices(
    const CpuUrmaTransport::HostSge* buffers,
    const std::uint64_t* offsets, std::size_t count,
    std::size_t remote_sgl_size, std::uint64_t remote_bytes,
    std::uint64_t max_group_bytes, std::size_t requested_slices,
    std::vector<SglGroup>* groups,
    std::vector<SglPlanSlice>* planned_slices = nullptr) {
  std::vector<SglPlanSlice> local_slices;
  auto* slices = planned_slices == nullptr ? &local_slices : planned_slices;
  auto status = PrepareSglPlanSlices(count, requested_slices, slices);
  if (!status.ok()) return status;

  std::vector<std::future<kvaf::Status>> workers;
  workers.reserve(slices->size());
  for (std::size_t i = 0U; i < slices->size(); ++i) {
    workers.push_back(std::async(std::launch::async, [&, i] {
      return PlanWriteCoalescedSlice(
          buffers, offsets, count, remote_sgl_size, remote_bytes,
          &(*slices)[i], max_group_bytes);
    }));
  }
  for (auto& worker : workers) worker.get();
  return MergeSglPlanSlices(*slices, count, groups);
}

void CheckPlanShape(const std::vector<SglGroup>& groups,
                    const CpuUrmaTransport::HostSge* buffers,
                    const std::uint64_t* offsets, std::size_t count,
                    std::size_t remote_sgl_size,
                    std::uint64_t max_group_bytes) {
  std::size_t next = 0U;
  for (const auto& group : groups) {
    Check(group.first == next && group.count > 0U,
          "SGL plan does not cover rows in order");
    Check(group.count <= remote_sgl_size,
          "SGL plan exceeds its row limit");
    const auto local_base =
        reinterpret_cast<std::uintptr_t>(buffers[group.first].data);
    const auto remote_base = offsets[group.first];
    std::uint64_t bytes = 0U;
    for (std::size_t row = group.first;
         row < group.first + group.count; ++row) {
      Check(reinterpret_cast<std::uintptr_t>(buffers[row].data) ==
                local_base + bytes &&
                offsets[row] == remote_base + bytes,
            "SGL group contains a non-contiguous row");
      bytes += buffers[row].bytes;
    }
    Check(bytes == group.bytes && bytes <= max_group_bytes,
          "SGL group byte count is invalid");
    next += group.count;
  }
  Check(next == count, "SGL plan lost rows");
}

void CheckSerialAndSlicedPlans(
    const std::vector<CpuUrmaTransport::HostSge>& buffers,
    const std::vector<std::uint64_t>& offsets,
    std::size_t remote_sgl_size, std::uint64_t remote_bytes,
    std::uint64_t max_group_bytes, std::size_t requested_slices) {
  std::vector<SglGroup> serial;
  Check(PlanWriteCoalescedGroups(
            buffers.data(), offsets.data(), buffers.size(), remote_sgl_size,
            remote_bytes, &serial, max_group_bytes).ok(),
        "serial SGL plan failed");
  std::vector<SglGroup> sliced;
  std::vector<SglPlanSlice> slices;
  Check(PlanInSlices(buffers.data(), offsets.data(), buffers.size(),
                     remote_sgl_size, remote_bytes, max_group_bytes,
                     requested_slices, &sliced, &slices).ok(),
        "sliced SGL plan failed");
  CheckPlanShape(serial, buffers.data(), offsets.data(), buffers.size(),
                 remote_sgl_size, max_group_bytes);
  CheckPlanShape(sliced, buffers.data(), offsets.data(), buffers.size(),
                 remote_sgl_size, max_group_bytes);
  Check(sliced.size() >= serial.size() &&
            sliced.size() <= serial.size() + slices.size() - 1U,
        "slice boundaries added more than one WR each");
}

void ParallelWritePlanning() {
  // Variable row lengths remain equivalent when all ranges are contiguous.
  std::array<std::uint8_t, 512> packed{};
  std::vector<CpuUrmaTransport::HostSge> buffers(37);
  std::vector<std::uint64_t> offsets(37);
  std::size_t cursor = 0U;
  for (std::size_t i = 0U; i < buffers.size(); ++i) {
    const std::size_t bytes = 1U + i % 7U;
    buffers[i] = {packed.data() + cursor, bytes};
    offsets[i] = cursor;
    cursor += bytes;
  }
  CheckSerialAndSlicedPlans(buffers, offsets, 8U, cursor, 64U, 6U);

  // Independent local and remote gaps must split both serial and sliced plans.
  std::array<std::uint8_t, 1024> sparse{};
  cursor = 0U;
  std::uint64_t remote_cursor = 0U;
  for (std::size_t i = 0U; i < buffers.size(); ++i) {
    const std::size_t bytes = 2U + i % 5U;
    if (i == 5U || i == 19U) cursor += 11U;
    if (i == 12U || i == 27U) remote_cursor += 13U;
    buffers[i] = {sparse.data() + cursor, bytes};
    offsets[i] = remote_cursor;
    cursor += bytes;
    remote_cursor += bytes;
  }
  CheckSerialAndSlicedPlans(buffers, offsets, 9U, remote_cursor, 48U, 7U);

  // Limits imposed by row count and device message size remain enforced.
  cursor = 0U;
  for (std::size_t i = 0U; i < buffers.size(); ++i) {
    const std::size_t bytes = 3U + i % 4U;
    buffers[i] = {packed.data() + cursor, bytes};
    offsets[i] = cursor;
    cursor += bytes;
  }
  CheckSerialAndSlicedPlans(buffers, offsets, 3U, cursor, 11U, 5U);

  // More workers than rows creates one non-empty slice per row.
  std::vector<SglPlanSlice> tiny_slices;
  Check(PrepareSglPlanSlices(3U, 64U, &tiny_slices).ok() &&
            tiny_slices.size() == 3U,
        "rows smaller than planner count created empty slices");
  for (std::size_t i = 0U; i < tiny_slices.size(); ++i) {
    Check(tiny_slices[i].first == i && tiny_slices[i].count == 1U,
          "small batch slices are not balanced");
  }

  // A slice boundary may split one otherwise coalesced group, but no rows.
  std::array<std::uint8_t, 64> boundary_source{};
  std::vector<CpuUrmaTransport::HostSge> boundary_buffers(8);
  std::vector<std::uint64_t> boundary_offsets(8);
  for (std::size_t i = 0U; i < boundary_buffers.size(); ++i) {
    boundary_buffers[i] = {boundary_source.data() + i * 8U, 8U};
    boundary_offsets[i] = i * 8U;
  }
  std::vector<SglGroup> serial, sliced;
  Check(PlanWriteCoalescedGroups(
            boundary_buffers.data(), boundary_offsets.data(), 8U, 8U, 64U,
            &serial, 64U).ok() && serial.size() == 1U,
        "serial boundary plan did not coalesce");
  Check(PlanInSlices(boundary_buffers.data(), boundary_offsets.data(), 8U, 8U,
                     64U, 64U, 2U, &sliced).ok() && sliced.size() == 2U &&
            sliced[0].first == 0U && sliced[0].count == 4U &&
            sliced[1].first == 4U && sliced[1].count == 4U,
        "slice boundary was not merged in row order");

  // An invalid row in a later slice invalidates the whole plan before merge.
  boundary_offsets[6] = 63U;
  std::vector<SglPlanSlice> invalid_slices;
  sliced = {{99U, 99U, 99U}};
  Check(!PlanInSlices(boundary_buffers.data(), boundary_offsets.data(), 8U,
                      8U, 64U, 64U, 4U, &sliced, &invalid_slices).ok() &&
            sliced.empty(),
        "later slice error left a partial merged plan");
  Check(invalid_slices.size() == 4U && invalid_slices[0].status.ok() &&
            !invalid_slices[3].status.ok(),
        "later slice error was not retained");
  boundary_offsets[6] = 48U;
  Check(PlanInSlices(boundary_buffers.data(), boundary_offsets.data(), 8U,
                     8U, 64U, 64U, 4U, &sliced, &invalid_slices).ok() &&
            sliced.size() == 4U,
        "reused slice workspace retained a previous planning error");
}

void PackedWriteCoalesce() {
  // Adjacent local and remote rows can share one contiguous WRITE WQE.
  for (const std::size_t count : {1U, 13U, 31U, 32U, 3000U, 6000U, 32769U}) {
    std::vector<std::uint8_t> source(count * 8);
    std::vector<CpuUrmaTransport::HostSge> buffers(count);
    std::vector<std::uint64_t> offsets(count);
    std::vector<std::uint8_t> expected(count * 8, 0xEE), actual = expected;
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < count; ++i) {
      const std::size_t bytes = 1 + i % 8;
      offsets[i] = cursor;
      buffers[i] = {source.data() + cursor, bytes};
      std::fill_n(source.data() + cursor, bytes, static_cast<std::uint8_t>(i % 197));
      std::memcpy(expected.data() + offsets[i], buffers[i].data, bytes);
      cursor += bytes;
    }
    for (const std::size_t limit : {1U, 31U, 64U, 128U}) {
      std::fill(actual.begin(), actual.end(), 0xEE);
      std::vector<SglGroup> groups;
      Check(PlanWriteCoalescedGroups(buffers.data(), offsets.data(), count, limit,
                                    actual.size(), &groups).ok(), "WRITE coalesce plan failed");
      Check(groups.size() == (count + limit - 1) / limit, "WRITE coalesced WQE count");
      std::size_t next = 0;
      // Exercise group rebasing at a 1024-WQE queue boundary as transport does.
      for (std::size_t begin = 0; begin < groups.size(); begin += 1024) {
        const auto base_row = groups[begin].first;
        const auto* chunk_buffers = buffers.data() + base_row;
        const auto* chunk_offsets = offsets.data() + base_row;
        for (std::size_t g = begin; g < std::min(begin + 1024, groups.size()); ++g) {
          const auto& group = groups[g];
          Check(group.first == next && group.count == std::min<std::size_t>(limit, count - next),
                "WRITE coalesced rows missing or duplicated");
          const auto first = group.first - base_row;
          const auto* local = static_cast<const std::uint8_t*>(chunk_buffers[first].data);
          std::memcpy(actual.data() + chunk_offsets[first], local, group.bytes);
          next += group.count;
        }
      }
      Check(next == count && actual == expected, "WRITE coalesced data corrupted");
    }
  }
}

void WriteCoalesceLimits() {
  std::array<std::uint8_t, 64> source{};
  CpuUrmaTransport::HostSge buffers[]{{source.data(), 3}, {source.data() + 3, 5},
                                     {source.data() + 20, 2}, {source.data() + 22, 6}};
  std::uint64_t offsets[]{0, 3, 20, 22};
  std::vector<SglGroup> groups;
  Check(PlanWriteCoalescedGroups(buffers, offsets, 4, 31, 64, &groups).ok(), "local gap plan");
  Check(groups.size() == 2 && groups[0].count == 2 && groups[1].count == 2,
        "noncontiguous local spans must split WRITE WQEs");
  offsets[1] = 4;
  Check(PlanWriteCoalescedGroups(buffers, offsets, 4, 31, 64, &groups).ok() && groups.size() == 3,
        "noncontiguous remote spans must split WRITE WQEs");
  offsets[1] = 3;
  Check(PlanWriteCoalescedGroups(buffers, offsets, 4, 1, 64, &groups).ok() && groups.size() == 4,
        "row coalescing limit not enforced");
  Check(!PlanWriteCoalescedGroups(buffers, offsets, 4, 0, 64, &groups).ok(), "zero remote SGL");
  offsets[3] = 60;
  Check(!PlanWriteCoalescedGroups(buffers, offsets, 4, 31, 64, &groups).ok() && groups.empty(),
        "remote overflow left a partial coalescing plan");
  offsets[3] = 8;
  buffers[3].data = reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 2);
  Check(!PlanWriteCoalescedGroups(buffers, offsets, 4, 31, 64, &groups).ok(), "local overflow");

  // Synthetic addresses are used only for planning, never dereferenced.
  constexpr std::uint64_t max_length = std::numeric_limits<std::uint32_t>::max();
  CpuUrmaTransport::HostSge large[]{{reinterpret_cast<void*>(4096), max_length - 1},
      {reinterpret_cast<void*>(4096 + max_length - 1), 2}};
  std::uint64_t large_offsets[]{0, max_length - 1};
  Check(PlanWriteCoalescedGroups(large, large_offsets, 2, 31, max_length + 1, &groups).ok() &&
        groups.size() == 2, "single local SGE length overflow was not split");
  large[0].bytes = 0;
  Check(!PlanWriteCoalescedGroups(large, large_offsets, 2, 31, max_length + 1, &groups).ok(),
        "zero length WRITE row accepted");

  CpuUrmaTransport::HostSge capped[]{{source.data(), 8}, {source.data() + 8, 8},
                                     {source.data() + 16, 8}, {source.data() + 24, 8}};
  std::uint64_t capped_offsets[]{0, 8, 16, 24};
  Check(PlanWriteCoalescedGroups(capped, capped_offsets, 4, 128, 64, &groups, 16).ok() &&
            groups.size() == 2 && groups[0].count == 2 && groups[1].count == 2,
        "device message size did not split WRITE groups");
  capped[0].bytes = 17;
  Check(!PlanWriteCoalescedGroups(capped, capped_offsets, 4, 128, 64, &groups, 16).ok(),
        "row larger than device message size was accepted");

  capped[0].bytes = 8;
  Check(PlanWriteCoalescedGroups(capped, capped_offsets, 4, 128, 64, &groups, 16).ok(),
        "registered-region plan failed");
  Check(GroupsFitRegisteredRegion(capped, groups.data(), groups.size(), source.data(), 32),
        "valid grouped host region rejected");
  Check(!GroupsFitRegisteredRegion(capped, groups.data(), groups.size(), source.data(), 31),
        "group outside registered host region accepted");
}

void BatchRowValidation() {
  std::array<std::uint8_t, 16> source{};
  CpuUrmaTransport::HostSge buffers[]{{source.data(), 8},
                                      {source.data() + 8, 8}};
  std::uint64_t offsets[]{0, 8};
  Check(ValidateBatchRows(buffers, offsets, 2, 16).ok(),
        "valid one-to-one batch rejected");
  offsets[1] = 9;
  Check(!ValidateBatchRows(buffers, offsets, 2, 16).ok(),
        "remote overflow accepted");
  offsets[1] = 8;
  buffers[1].data = nullptr;
  Check(!ValidateBatchRows(buffers, offsets, 2, 16).ok(),
        "null local row accepted");
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
    ParallelWritePlanning();
    PackedWriteCoalesce();
    WriteCoalesceLimits();
    BatchRowValidation();
    JettyChunkSubmission();
    std::cout << "CPU URMA SGL planning tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
