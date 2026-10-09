#include "cpu_gather.h"
#include "cpu_gather_team.h"
#include "peer_protocol.h"

#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
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
    team.Copy();
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
  team.Copy();
  Require(std::memcmp(memory.data() + staging_offset,
                      memory.data() + source_offset, payload_bytes) == 0);

  spans = {{base + source_offset, 64U, 0U}};
  Require(team.Prepare(staging, spans, 64U) == 1U);
  team.Copy();
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
    tuned_team.Copy();
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

  kvaf::peer::CpuGatherTeam adaptive_team(memory.data(), base, 4U, 64U * 1024U,
                                         kvaf::peer::GatherWaitMode::kAdaptive);
  for (std::size_t round = 0U; round < 4U; ++round) {
    if (round != 0U) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    memory[source_offset + round] = static_cast<std::uint8_t>(round + 1U);
    Require(adaptive_team.Prepare(staging, spans, total_bytes) == 4U);
    adaptive_team.Copy();
    for (std::size_t block = 0U; block < block_count; ++block)
      Require(std::memcmp(memory.data() + staging_offset + block * block_bytes,
                          memory.data() + source_offset +
                              (block_count - 1U - block) * block_bytes,
                          block_bytes) == 0);
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
  team.Copy();
  std::size_t destination_offset = staging_offset;
  for (const auto& span : spans) {
    Require(std::memcmp(memory.data() + destination_offset,
                        memory.data() + (span.address - base), span.length) == 0);
    destination_offset += span.length;
  }
}

void TestDirectDescriptorLayout() {
  std::array<std::uint8_t, 512> memory{};
  const auto base = reinterpret_cast<std::uint64_t>(memory.data());
  std::vector<kvaf::peer::GatherSpan> spans{
      {base + 32U, 16U, 0U}, {base, 16U, 0U}};
  kvaf::peer::GatherDescriptorHeader header{
      kvaf::peer::kGatherDescriptorMagic, 1U, base + 128U, 2U, 32U};
  assert(kvaf::peer::ValidateDirectGather(base, memory.size(), base + 256U,
                                          header, spans));
  header.total_bytes = 31U;
  assert(!kvaf::peer::ValidateDirectGather(base, memory.size(), base + 256U,
                                           header, spans));
  header.total_bytes = 32U;
  header.staging_address = base + 256U;
  assert(!kvaf::peer::ValidateDirectGather(base, memory.size(), base + 256U,
                                           header, spans));
  header.staging_address = base + 128U;
  spans[0].address = base + 256U;
  assert(!kvaf::peer::ValidateDirectGather(base, memory.size(), base + 256U,
                                           header, spans));
  spans[0].address = base + 32U;
  assert(!kvaf::peer::ValidateDirectGather(base, memory.size(), base + 400U,
                                           header, spans));
}

void TestDirectPreparation() {
  constexpr std::size_t span_count = 4096U;
  constexpr std::size_t source_stride = 128U;
  constexpr std::size_t source_bytes = span_count * source_stride;
  constexpr std::size_t staging_offset = source_bytes + 64U;
  constexpr std::size_t descriptor_offset = staging_offset + source_bytes + 64U;
  std::vector<std::uint8_t> memory(descriptor_offset +
                                  kvaf::peer::GatherDescriptorBytes(span_count) + 256U);
  const auto base = reinterpret_cast<std::uint64_t>(memory.data());
  const auto address = (base + descriptor_offset + 63U) & ~std::uint64_t{63U};
  for (const auto mode : {kvaf::peer::GatherWaitMode::kPassive,
                          kvaf::peer::GatherWaitMode::kActive,
                          kvaf::peer::GatherWaitMode::kAdaptive}) {
    kvaf::peer::CpuGatherTeam team(memory.data(), base, 4U, 256U, mode);
    for (std::uint64_t sequence = 1U; sequence <= 24U; ++sequence) {
      const std::size_t count = sequence % 3U == 0U ? 1U : span_count - sequence;
      std::vector<kvaf::peer::GatherSpan> spans(count);
      std::uint32_t total = 0U;
      for (std::size_t index = 0U; index < count; ++index) {
        const auto offset = ((index * 37U + sequence) % span_count) * source_stride;
        const auto length = static_cast<std::uint32_t>(17U + (index + sequence) % 97U);
        spans[index] = {base + offset, length, 1U};
        for (std::size_t byte = 0U; byte < length; ++byte)
          memory[offset + byte] = static_cast<std::uint8_t>(index + byte + sequence);
        total += length;
      }
      kvaf::peer::GatherDescriptorHeader header{
          kvaf::peer::kGatherDescriptorMagic, sequence, base + staging_offset,
          static_cast<std::uint32_t>(count), total};
      auto* descriptor = memory.data() + (address - base);
      std::memcpy(descriptor, &header, sizeof(header));
      std::memcpy(descriptor + sizeof(header), spans.data(), spans.size() * sizeof(spans[0U]));
      Require(team.PrepareDirect(memory.size(), address, sequence + 1U, span_count) == 0U);
      Require(team.PrepareDirect(memory.size(), address, sequence, count - 1U) == 0U);
      if (sequence == 8U) {
        auto invalid = spans.front();
        invalid.address = base + staging_offset;
        std::memcpy(descriptor + sizeof(header), &invalid, sizeof(invalid));
        Require(team.PrepareDirect(memory.size(), address, sequence, span_count) == 0U);
        std::memcpy(descriptor + sizeof(header), spans.data(), sizeof(spans[0U]));
      }
      memory[staging_offset - 1U] = 0xa5U;
      memory[staging_offset + total] = 0x5aU;
      Require(team.PrepareDirect(memory.size(), address, sequence, span_count) ==
              (count == 1U ? 1U : 4U));
      Require(team.total_bytes() == total);
      team.Copy();
      std::size_t output = staging_offset;
      for (const auto& span : spans) {
        Require(std::memcmp(memory.data() + output,
                            memory.data() + (span.address - base), span.length) == 0);
        output += span.length;
      }
      Require(memory[staging_offset - 1U] == 0xa5U);
      Require(memory[staging_offset + total] == 0x5aU);
      if (mode == kvaf::peer::GatherWaitMode::kAdaptive)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
}

void TestDirectPreparationBounds() {
  alignas(64) std::array<std::uint8_t, 1024> memory{};
  const auto base = reinterpret_cast<std::uint64_t>(memory.data());
  const auto address = base + 512U;
  kvaf::peer::CpuGatherTeam team(memory.data(), base, 2U);
  const kvaf::peer::GatherDescriptorHeader valid_header{
      kvaf::peer::kGatherDescriptorMagic, 1U, base + 256U, 2U, 32U};
  const std::vector<kvaf::peer::GatherSpan> valid_spans{{base, 16U, 0U}, {base + 64U, 16U, 0U}};
  auto prepare = [&](const kvaf::peer::GatherDescriptorHeader& header,
                     const std::vector<kvaf::peer::GatherSpan>& spans) {
    std::memcpy(memory.data() + 512U, &header, sizeof(header));
    std::memcpy(memory.data() + 512U + sizeof(header), spans.data(),
                spans.size() * sizeof(spans[0U]));
    return team.PrepareDirect(memory.size(), address, 1U, 4096U);
  };
  for (const auto staging : {base + 8U, address, base + 960U, UINT64_MAX - 16U}) {
    auto header = valid_header;
    header.staging_address = staging;
    Require(prepare(header, valid_spans) == 0U);
  }
  for (const auto length : {0U, UINT32_MAX}) {
    auto spans = valid_spans;
    spans[0U].length = length;
    Require(prepare(valid_header, spans) == 0U);
  }
  for (const auto source : {base - 1U, address, base + 952U, UINT64_MAX - 8U}) {
    auto spans = valid_spans;
    spans[0U].address = source;
    Require(prepare(valid_header, spans) == 0U);
  }
  for (const auto count : {0U, 29U, UINT32_MAX}) {
    auto header = valid_header;
    header.span_count = count;
    Require(prepare(header, valid_spans) == 0U);
  }
  auto header = valid_header;
  header.total_bytes = 31U;
  Require(prepare(header, valid_spans) == 0U);
  header.total_bytes = 33U;
  Require(prepare(header, valid_spans) == 0U);
  header = valid_header;
  header.magic = 0U;
  Require(prepare(header, valid_spans) == 0U);
  for (const auto chunk_bytes : {1U, 63U, 65U}) {
    header = valid_header;
    header.pipeline_chunk_bytes = chunk_bytes;
    Require(prepare(header, valid_spans) == 0U);
  }
  header = valid_header;
  header.ready_bytes = 16U;
  Require(prepare(header, valid_spans) == 0U);
  Require(team.PrepareDirect(32U, address, 1U, 4096U) == 0U);
  Require(team.PrepareDirect(memory.size(), address + 1U, 1U, 4096U) == 0U);
  Require(team.PrepareDirect(memory.size(), base + 960U, 1U, 4096U) == 0U);
  Require(team.PrepareDirect(UINT64_MAX, address, 1U, 4096U) == 0U);
  Require(prepare(valid_header, valid_spans) == 1U);
  team.Copy();
  Require(kvaf::peer::ValidateDirectGather(base, memory.size(), address,
                                          valid_header, valid_spans));
  Require(!kvaf::peer::ValidateDirectGather(UINT64_MAX - 8U, 1024U, address,
                                           valid_header, valid_spans));
}

void TestPipelinedGather() {
  constexpr std::size_t span_count = 257U;
  constexpr std::size_t stride = 1024U;
  constexpr std::size_t staging_offset = span_count * stride + 128U;
  constexpr std::size_t descriptor_offset = 2U * staging_offset;
  std::vector<std::uint8_t> memory(descriptor_offset +
      kvaf::peer::GatherDescriptorBytes(span_count) + 256U);
  const auto base = reinterpret_cast<std::uint64_t>(memory.data());
  const auto descriptor = (base + descriptor_offset + 63U) & ~std::uint64_t{63U};
  auto* descriptor_memory = reinterpret_cast<std::uint8_t*>(descriptor);
  auto* progress = reinterpret_cast<std::uint64_t*>(descriptor_memory +
      offsetof(kvaf::peer::GatherDescriptorHeader, ready_bytes));
  std::vector<kvaf::peer::GatherSpan> spans(span_count);
  std::vector<std::uint8_t> expected;
  kvaf::peer::CpuGatherTeam team(memory.data(), base, 8U, 4096U,
                                kvaf::peer::GatherWaitMode::kActive);
  for (std::uint64_t sequence = 1U; sequence <= 9U; ++sequence) {
    expected.clear();
    for (std::size_t index = 0U; index < span_count; ++index) {
      const auto source_offset = ((index + sequence) % span_count) * stride;
      const auto length = static_cast<std::uint32_t>((index + sequence) % 2U == 0U ? 257U : 513U);
      spans[index] = {base + source_offset, length, 0U};
      for (std::uint32_t byte = 0U; byte < length; ++byte)
        memory[source_offset + byte] = static_cast<std::uint8_t>(index + byte + sequence);
      expected.insert(expected.end(), memory.data() + source_offset,
                      memory.data() + source_offset + length);
    }
    const auto chunk_bytes = sequence == 9U ? 0U : sequence % 2U == 0U ? 4096U : 8192U;
    const kvaf::peer::GatherDescriptorHeader header{
        kvaf::peer::kGatherDescriptorMagic, sequence, base + staging_offset,
        span_count, static_cast<std::uint32_t>(expected.size()), chunk_bytes};
    std::memcpy(descriptor_memory, &header, sizeof(header));
    std::memcpy(descriptor_memory + sizeof(header), spans.data(), spans.size() * sizeof(spans.front()));
    Require(team.PrepareDirect(memory.size(), descriptor, sequence, span_count) == 8U);
    std::atomic<bool> done{false};
    std::thread copier([&] {
      team.Copy();
      done.store(true, std::memory_order_release);
    });
    std::uint64_t observed = 0U;
    while (!done.load(std::memory_order_acquire)) {
      const auto ready = __atomic_load_n(progress, __ATOMIC_ACQUIRE);
      Require(ready >= observed && ready <= expected.size());
      Require(ready == expected.size() || ready % (chunk_bytes == 0U ? 1U : chunk_bytes) == 0U);
      Require(std::memcmp(memory.data() + staging_offset + observed,
                          expected.data() + observed, ready - observed) == 0);
      observed = ready;
      std::this_thread::yield();
    }
    copier.join();
    Require(__atomic_load_n(progress, __ATOMIC_ACQUIRE) ==
            (chunk_bytes == 0U ? 0U : expected.size()));
    Require(std::memcmp(memory.data() + staging_offset, expected.data(), expected.size()) == 0);
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
  assert(kvaf::peer::GatherStartedAck(1U) != 1U);
  assert(kvaf::peer::GatherStartedAck(1U) != (1U | kvaf::peer::kGatherErrorBit));
  assert(sizeof(kvaf::peer::GatherSpan) == sizeof(kvaf::peer::GatherSpanWire));
  TestParallelGather();
  TestIrregularGather();
  TestDirectDescriptorLayout();
  TestDirectPreparation();
  TestDirectPreparationBounds();
  TestPipelinedGather();
  std::cout << "CPU gather: PASS\n";
}
