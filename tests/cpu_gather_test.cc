#include "cpu_gather.h"
#include "cpu_gather_team.h"
#include "peer_protocol.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

void Require(bool condition) {
  if (!condition) throw std::runtime_error("parallel CPU gather mismatch");
}

void TestParallelGather() {
  constexpr std::size_t block_count = 4096U;
  constexpr std::size_t block_bytes = 656U;
  constexpr std::size_t payload_bytes = block_count * block_bytes;
  constexpr std::size_t source_offset = 64U;
  constexpr std::size_t staging_offset = source_offset + payload_bytes + 17U;
  std::vector<std::uint8_t> memory(staging_offset + payload_bytes + 512U);
  const auto base = reinterpret_cast<std::uint64_t>(memory.data());
  const auto staging = base + staging_offset;
  std::vector<kvaf::peer::GatherSpan> spans;
  spans.reserve(block_count);
  for (std::size_t block = block_count; block != 0U; --block)
    spans.push_back({base + source_offset + (block - 1U) * block_bytes,
                     block_bytes, 0U});
  std::uint32_t total_bytes = 0U;
  Require(kvaf::peer::ValidateGather(base, memory.size(), staging, spans, &total_bytes));
  Require(total_bytes == payload_bytes);

  kvaf::peer::CpuGatherTeam team(memory.data(), base, 16U);
  for (std::size_t round = 0U; round < 32U; ++round) {
    for (std::size_t index = 0U; index < payload_bytes; ++index)
      memory[source_offset + index] = static_cast<std::uint8_t>((index * 17U + round) & 0xffU);
    memory[staging_offset - 1U] = 0xa5U;
    memory[staging_offset + payload_bytes] = 0x5aU;
    Require(team.Prepare(staging, spans, total_bytes) > 1U);
    team.InvalidateSources();
    team.Copy();
    team.FlushStaging();
    for (std::size_t block = 0U; block < block_count; ++block) {
      Require(std::memcmp(memory.data() + staging_offset + block * block_bytes,
                          memory.data() + source_offset +
                              (block_count - 1U - block) * block_bytes,
                          block_bytes) == 0);
    }
    Require(memory[staging_offset - 1U] == 0xa5U);
    Require(memory[staging_offset + payload_bytes] == 0x5aU);
  }

  spans = {{base + source_offset, payload_bytes, 0U}};
  Require(team.Prepare(staging, spans, total_bytes) > 1U);
  team.InvalidateSources();
  team.Copy();
  team.FlushStaging();
  Require(std::memcmp(memory.data() + staging_offset,
                      memory.data() + source_offset, payload_bytes) == 0);

  spans = {{base + source_offset, 64U, 0U}};
  Require(team.Prepare(staging, spans, 64U) == 1U);
  team.InvalidateSources();
  team.Copy();
  team.FlushStaging();
  Require(std::memcmp(memory.data() + staging_offset,
                      memory.data() + source_offset, 64U) == 0);

  spans.clear();
  for (std::size_t block = block_count; block != 0U; --block)
    spans.push_back({base + source_offset + (block - 1U) * block_bytes,
                     block_bytes, 0U});
  kvaf::peer::CpuGatherTeam tuned_team(memory.data(), base, 16U, 64U * 1024U,
                                      kvaf::peer::GatherWaitMode::kActive);
  const std::vector<kvaf::peer::GatherSpan> small_spans{{base + source_offset, 64U, 0U}};
  for (std::size_t round = 0U; round < 16U; ++round) {
    const bool small = round == 8U;
    const auto& selected = small ? small_spans : spans;
    Require(tuned_team.Prepare(staging, selected,
                               small ? 64U : total_bytes) == (small ? 1U : 16U));
    tuned_team.InvalidateSources();
    tuned_team.Copy();
    tuned_team.FlushStaging();
    if (small) {
      Require(std::memcmp(memory.data() + staging_offset,
                          memory.data() + source_offset, 64U) == 0);
    } else {
      for (std::size_t block = 0U; block < block_count; ++block)
        Require(std::memcmp(memory.data() + staging_offset + block * block_bytes,
                            memory.data() + source_offset +
                                (block_count - 1U - block) * block_bytes,
                            block_bytes) == 0);
    }
  }
}

void TestIrregularGather() {
  constexpr std::size_t span_count = 2048U;
  constexpr std::size_t source_offset = 64U;
  constexpr std::size_t source_stride = 1024U;
  const std::size_t staging_offset = source_offset + span_count * source_stride + 17U;
  std::size_t payload_bytes = 0U;
  for (std::size_t index = 0U; index < span_count; ++index)
    payload_bytes += index % 2U == 0U ? 257U : 513U;
  std::vector<std::uint8_t> memory(staging_offset + payload_bytes + 512U);
  const auto base = reinterpret_cast<std::uint64_t>(memory.data());
  std::vector<kvaf::peer::GatherSpan> spans;
  spans.reserve(span_count);
  for (std::size_t index = 0U; index < span_count; ++index) {
    const std::size_t source_index = index * 37U % span_count;
    const std::size_t length = index % 2U == 0U ? 257U : 513U;
    const std::size_t offset = source_offset + source_index * source_stride;
    for (std::size_t byte = 0U; byte < length; ++byte)
      memory[offset + byte] = static_cast<std::uint8_t>((index + byte * 3U) & 0xffU);
    spans.push_back({base + offset, static_cast<std::uint32_t>(length), 0U});
  }
  std::uint32_t total_bytes = 0U;
  const auto staging = base + staging_offset;
  Require(kvaf::peer::ValidateGather(base, memory.size(), staging, spans, &total_bytes));
  Require(total_bytes == payload_bytes);
  kvaf::peer::CpuGatherTeam team(memory.data(), base, 8U);
  Require(team.Prepare(staging, spans, total_bytes) > 1U);
  team.InvalidateSources();
  team.Copy();
  team.FlushStaging();
  std::size_t destination_offset = staging_offset;
  for (const auto& span : spans) {
    Require(std::memcmp(memory.data() + destination_offset,
                        memory.data() + (span.address - base), span.length) == 0);
    destination_offset += span.length;
  }
}

int main() {
  std::array<std::uint8_t, 256> memory{};
  const auto base = reinterpret_cast<std::uint64_t>(memory.data());
  for (std::uint32_t index = 0U; index < 128U; ++index)
    memory[index] = static_cast<std::uint8_t>(index);
  const std::vector<kvaf::peer::GatherSpan> spans{
      {base + 80U, 16U, 0U}, {base + 0U, 16U, 0U}, {base + 48U, 16U, 0U}};
  std::uint32_t total = 0U;
  assert(kvaf::peer::ValidateGather(base, memory.size(), base + 128U, spans, &total));
  assert(total == 48U);
  kvaf::peer::GatherIntoStaging(memory.data(), base, base + 128U, spans);
  for (std::uint32_t index = 0U; index < 16U; ++index) {
    assert(memory[128U + index] == 80U + index);
    assert(memory[144U + index] == index);
    assert(memory[160U + index] == 48U + index);
  }
  assert(!kvaf::peer::ValidateGather(base, memory.size(), base + 64U, spans, &total));
  assert(!kvaf::peer::ValidateGather(base, memory.size(), base + 192U, spans, &total));
  const std::vector<kvaf::peer::GatherSpan> bad{{base + 200U, 8U, 0U}};
  assert(!kvaf::peer::ValidateGather(base, memory.size(), base + 128U, bad, &total));
  assert(kvaf::peer::GatherControlOffset(256U) == 192U);
  assert(sizeof(kvaf::peer::GatherSpan) == sizeof(kvaf::peer::GatherSpanWire));
  TestParallelGather();
  TestIrregularGather();
  std::cout << "CPU gather: PASS\n";
}
