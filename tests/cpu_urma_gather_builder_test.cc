#include "cpu_urma/batch_validation.h"
#include "cpu_urma/jetty_batch_dispatch.h"
#include "cpu_urma/sgl_batch_builder.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using kvaf::CpuUrmaTransport;
using kvaf::cpu_urma_detail::BatchValidationSlice;
using kvaf::cpu_urma_detail::BuildWriteGatherShard;
using kvaf::cpu_urma_detail::PrepareBatchValidationSlices;
using kvaf::cpu_urma_detail::ValidateBatchRows;
using kvaf::cpu_urma_detail::WriteGatherGroup;

void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

struct BuiltShard {
  std::vector<std::size_t> local_indices;
  std::vector<std::size_t> rows;
  std::vector<const void*> addresses;
  std::vector<std::size_t> lengths;
  std::vector<WriteGatherGroup> groups;
  std::size_t wr_count{0U};
};

kvaf::Status BuildShard(
    const std::vector<CpuUrmaTransport::HostSge>& buffers,
    const std::vector<std::uint64_t>& offsets,
    std::size_t first, std::size_t count, std::size_t local_sgl_size,
    std::uint64_t remote_bytes, std::uint64_t max_group_bytes,
    BuiltShard* result) {
  *result = {};
  return BuildWriteGatherShard(
      buffers.data(), offsets.data(), buffers.size(), first, count,
      local_sgl_size, remote_bytes, max_group_bytes,
      [&](std::size_t local_index, std::size_t row,
          const CpuUrmaTransport::HostSge& buffer) {
        result->local_indices.push_back(local_index);
        result->rows.push_back(row);
        result->addresses.push_back(buffer.data);
        result->lengths.push_back(buffer.bytes);
      },
      [&](std::size_t wr, const WriteGatherGroup& group) {
        Check(wr == result->groups.size(), "WRITE WR order changed");
        result->groups.push_back(group);
      },
      &result->wr_count);
}

bool SameGroups(const BuiltShard& left, const BuiltShard& right) {
  if (left.groups.size() != right.groups.size()) return false;
  for (std::size_t i = 0U; i < left.groups.size(); ++i) {
    if (left.groups[i].first != right.groups[i].first ||
        left.groups[i].count != right.groups[i].count ||
        left.groups[i].bytes != right.groups[i].bytes) {
      return false;
    }
  }
  return true;
}

void OnePassGatherSemantics() {
  std::array<std::uint8_t, 256> source{};
  std::vector<CpuUrmaTransport::HostSge> adjacent(8U);
  std::vector<CpuUrmaTransport::HostSge> scattered(8U);
  std::vector<std::uint64_t> offsets(8U);
  for (std::size_t row = 0U; row < offsets.size(); ++row) {
    adjacent[row] = {source.data() + row * 8U, 8U};
    scattered[row] = {source.data() + (15U - row) * 16U, 8U};
    offsets[row] = row * 8U;
  }

  BuiltShard packed;
  BuiltShard sparse;
  Check(BuildShard(adjacent, offsets, 0U, 8U, 3U, 64U, 64U,
                   &packed).ok(),
        "adjacent WRITE gather failed");
  Check(BuildShard(scattered, offsets, 0U, 8U, 3U, 64U, 64U,
                   &sparse).ok(),
        "scattered WRITE gather failed");
  Check(SameGroups(packed, sparse) && sparse.wr_count == 3U &&
            sparse.groups[0].count == 3U &&
            sparse.groups[1].count == 3U &&
            sparse.groups[2].count == 2U,
        "local layout changed WRITE grouping");
  for (std::size_t row = 0U; row < scattered.size(); ++row) {
    Check(sparse.local_indices[row] == row && sparse.rows[row] == row &&
              sparse.addresses[row] == scattered[row].data &&
              sparse.lengths[row] == scattered[row].bytes,
          "one input row did not produce one local SGE");
  }

  offsets[3] = 32U;
  offsets[4] = 40U;
  offsets[5] = 48U;
  offsets[6] = 56U;
  offsets[7] = 65U;
  BuiltShard split;
  Check(BuildShard(scattered, offsets, 0U, 8U, 8U, 80U, 24U,
                   &split).ok() &&
            split.wr_count == 4U &&
            split.groups[0].count == 3U &&
            split.groups[1].count == 3U &&
            split.groups[2].count == 1U &&
            split.groups[3].count == 1U,
        "remote gap or message limit did not split WRITE WRs");

  scattered.back().data = nullptr;
  Check(!BuildShard(scattered, offsets, 0U, 8U, 8U, 80U, 24U,
                    &split).ok() && split.wr_count == 0U,
        "late invalid row exposed a partial WR count");
}

void FourJettyPacketShape() {
  constexpr std::size_t kRows = 4096U;
  constexpr std::size_t kRowsPerJetty = kRows / 4U;
  constexpr std::size_t kRowBytes = 656U;
  std::vector<std::uint8_t> source(kRows * kRowBytes);
  std::vector<CpuUrmaTransport::HostSge> buffers(kRows);
  std::vector<std::uint64_t> offsets(kRows);
  for (std::size_t row = 0U; row < kRows; ++row) {
    buffers[row] = {source.data() + row * kRowBytes, kRowBytes};
    offsets[row] = row * kRowBytes;
  }

  std::size_t sgl12_wrs = 0U;
  std::size_t sgl13_wrs = 0U;
  for (std::size_t jetty = 0U; jetty < 4U; ++jetty) {
    const std::size_t first = jetty * kRowsPerJetty;
    BuiltShard sgl12;
    BuiltShard sgl13;
    Check(BuildShard(buffers, offsets, first, kRowsPerJetty, 12U,
                     source.size(), 65536U, &sgl12).ok() &&
              sgl12.groups.front().bytes == 7872U,
          "SGE=12 WRITE shape changed");
    Check(BuildShard(buffers, offsets, first, kRowsPerJetty, 13U,
                     source.size(), 65536U, &sgl13).ok() &&
              sgl13.groups.front().bytes == 8528U,
          "SGE=13 WRITE shape changed");
    sgl12_wrs += sgl12.wr_count;
    sgl13_wrs += sgl13.wr_count;
  }
  Check(sgl12_wrs == 344U && sgl13_wrs == 316U,
        "four-Jetty SGE=12/13 WR counts changed");
}

void JettyShardAndWaveCoverage() {
  constexpr std::size_t kJettys = 4U;
  constexpr std::size_t kRowsPerWave = 20U;
  constexpr std::size_t kRows = 27U;
  std::array<std::uint8_t, kRows * 8U> source{};
  std::vector<CpuUrmaTransport::HostSge> buffers(kRows);
  std::vector<std::uint64_t> offsets(kRows);
  for (std::size_t row = 0U; row < kRows; ++row) {
    buffers[row] = {source.data() + (kRows - 1U - row) * 8U, 8U};
    offsets[row] = row * 8U;
  }

  std::vector<std::size_t> visits(kRows, 0U);
  std::size_t total_wrs = 0U;
  for (std::size_t wave_first = 0U; wave_first < kRows;
       wave_first += kRowsPerWave) {
    const std::size_t wave_count =
        std::min(kRowsPerWave, kRows - wave_first);
    const std::size_t base = wave_count / kJettys;
    const std::size_t remainder = wave_count % kJettys;
    for (std::size_t jetty = 0U; jetty < kJettys; ++jetty) {
      const std::size_t count = base + (jetty < remainder ? 1U : 0U);
      if (count == 0U) continue;
      const std::size_t first = wave_first + jetty * base +
          std::min(jetty, remainder);
      BuiltShard shard;
      Check(BuildShard(buffers, offsets, first, count, 3U,
                       kRows * 8U, 64U, &shard).ok(),
            "Jetty shard build failed");
      for (const std::size_t row : shard.rows) ++visits[row];
      total_wrs += shard.wr_count;
    }
  }
  Check(std::all_of(visits.begin(), visits.end(),
                    [](std::size_t count) { return count == 1U; }) &&
            total_wrs == 12U,
        "Jetty waves lost or duplicated rows");
}

void BatchValidation() {
  std::array<std::uint8_t, 16> source{};
  CpuUrmaTransport::HostSge buffers[]{{source.data(), 8U},
                                      {source.data() + 8U, 8U}};
  std::uint64_t offsets[]{0U, 8U};
  Check(ValidateBatchRows(buffers, offsets, 2U, 16U).ok(),
        "valid batch rows rejected");
  offsets[1] = 9U;
  Check(!ValidateBatchRows(buffers, offsets, 2U, 16U).ok(),
        "remote overflow accepted");
  offsets[1] = 8U;
  Check(!ValidateBatchRows(buffers, offsets, 2U, 16U, 7U).ok(),
        "device message limit ignored");

  std::vector<BatchValidationSlice> slices;
  Check(PrepareBatchValidationSlices(10U, 3U, &slices).ok() &&
            slices.size() == 3U &&
            slices[0].first == 0U && slices[0].count == 4U &&
            slices[1].first == 4U && slices[1].count == 3U &&
            slices[2].first == 7U && slices[2].count == 3U,
        "batch validation slices are not balanced");
  Check(!PrepareBatchValidationSlices(0U, 3U, &slices).ok() &&
            slices.empty(),
        "empty validation input accepted");
}

void JettyChunkSubmission() {
  std::size_t prepared = 0U;
  std::size_t posts = 0U;
  kvaf::cpu_urma_detail::DispatchJettyChunks(
      97U, 32U,
      [&](std::size_t begin, std::size_t end) {
        Check(begin == prepared, "Jetty chunks were prepared out of order");
        prepared = end;
      },
      [&](std::size_t begin, std::size_t end) {
        Check(begin < end && end == prepared,
              "Jetty chunk posted before preparation");
        ++posts;
        return true;
      });
  Check(prepared == 97U && posts == 4U, "Jetty chunk tail was lost");

  prepared = 0U;
  posts = 0U;
  kvaf::cpu_urma_detail::DispatchJettyChunks(
      97U, 32U,
      [&](std::size_t, std::size_t end) { prepared = end; },
      [&](std::size_t, std::size_t) { return ++posts != 2U; });
  Check(prepared == 64U && posts == 2U,
        "Jetty dispatch continued after a failed post");
}
}  // namespace

int main() {
  try {
    OnePassGatherSemantics();
    FourJettyPacketShape();
    JettyShardAndWaveCoverage();
    BatchValidation();
    JettyChunkSubmission();
    std::cout << "CPU URMA gather builder tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
