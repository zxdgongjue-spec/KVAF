#include "cpu_gather.h"

#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cpu_gather_profile.h"
#include "cpu_gather_team.h"
#include "peer_protocol.h"

void RequireAt(bool condition, int line) {
  if (!condition)
    throw std::runtime_error("parallel CPU gather mismatch at line " + std::to_string(line));
}
#define Require(condition) RequireAt((condition), __LINE__)

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
    spans.push_back({base + source_offset + (block - 1U) * block_bytes, block_bytes, 0U});
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
                          memory.data() + source_offset + (block_count - 1U - block) * block_bytes,
                          block_bytes) == 0);
    }
    Require(memory[staging_offset - 1U] == 0xa5U);
    Require(memory[staging_offset + payload_bytes] == 0x5aU);
  }

  spans = {{base + source_offset, payload_bytes, 0U}};
  Require(team.Prepare(staging, spans, total_bytes) > 1U);
  team.Copy();
  Require(std::memcmp(memory.data() + staging_offset, memory.data() + source_offset,
                      payload_bytes) == 0);

  spans = {{base + source_offset, 64U, 0U}};
  Require(team.Prepare(staging, spans, 64U) == 1U);
  team.Copy();
  Require(std::memcmp(memory.data() + staging_offset, memory.data() + source_offset, 64U) == 0);

  spans.clear();
  for (std::size_t block = block_count; block != 0U; --block)
    spans.push_back({base + source_offset + (block - 1U) * block_bytes, block_bytes, 0U});
  kvaf::peer::CpuGatherTeam tuned_team(memory.data(), base, 16U, 64U * 1024U,
                                       kvaf::peer::GatherWaitMode::kActive);
  const std::vector<kvaf::peer::GatherSpan> small_spans{{base + source_offset, 64U, 0U}};
  for (std::size_t round = 0U; round < 16U; ++round) {
    const bool small = round == 8U;
    const auto& selected = small ? small_spans : spans;
    Require(tuned_team.Prepare(staging, selected, small ? 64U : total_bytes) == (small ? 1U : 16U));
    tuned_team.Copy();
    if (small) {
      Require(std::memcmp(memory.data() + staging_offset, memory.data() + source_offset, 64U) == 0);
    } else {
      for (std::size_t block = 0U; block < block_count; ++block)
        Require(
            std::memcmp(memory.data() + staging_offset + block * block_bytes,
                        memory.data() + source_offset + (block_count - 1U - block) * block_bytes,
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
                          memory.data() + source_offset + (block_count - 1U - block) * block_bytes,
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
    Require(std::memcmp(memory.data() + destination_offset, memory.data() + (span.address - base),
                        span.length) == 0);
    destination_offset += span.length;
  }
}

void TestDirectDescriptorLayout() {
  std::array<std::uint8_t, 512> memory{};
  const auto base = reinterpret_cast<std::uint64_t>(memory.data());
  std::vector<kvaf::peer::GatherSpan> spans{{base + 32U, 16U, 0U}, {base, 16U, 0U}};
  kvaf::peer::GatherDescriptorHeader header{kvaf::peer::kGatherDescriptorMagic, 1U, base + 128U, 2U,
                                            32U};
  assert(kvaf::peer::ValidateDirectGather(base, memory.size(), base + 256U, header, spans));
  header.total_bytes = 31U;
  assert(!kvaf::peer::ValidateDirectGather(base, memory.size(), base + 256U, header, spans));
  header.total_bytes = 32U;
  header.staging_address = base + 256U;
  assert(!kvaf::peer::ValidateDirectGather(base, memory.size(), base + 256U, header, spans));
  header.staging_address = base + 128U;
  spans[0].address = base + 256U;
  assert(!kvaf::peer::ValidateDirectGather(base, memory.size(), base + 256U, header, spans));
  spans[0].address = base + 32U;
  assert(!kvaf::peer::ValidateDirectGather(base, memory.size(), base + 400U, header, spans));
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
  for (const auto mode : {kvaf::peer::GatherWaitMode::kPassive, kvaf::peer::GatherWaitMode::kActive,
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
      kvaf::peer::GatherDescriptorHeader header{kvaf::peer::kGatherDescriptorMagic, sequence,
                                                base + staging_offset,
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
        Require(std::memcmp(memory.data() + output, memory.data() + (span.address - base),
                            span.length) == 0);
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
  const kvaf::peer::GatherDescriptorHeader valid_header{kvaf::peer::kGatherDescriptorMagic, 1U,
                                                        base + 256U, 2U, 32U};
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
  Require(
      kvaf::peer::ValidateDirectGather(base, memory.size(), address, valid_header, valid_spans));
  Require(!kvaf::peer::ValidateDirectGather(UINT64_MAX - 8U, 1024U, address, valid_header,
                                            valid_spans));
}

// Exercise new descriptor contents, repeated layouts, and independent API calls.
void TestChangingDirectLayouts() {
  constexpr std::size_t count = 257U;
  constexpr std::size_t stride = 1024U;
  constexpr std::size_t payload = count * 656U;
  constexpr std::size_t first_staging = count * stride + 128U;
  constexpr std::size_t second_staging = first_staging + payload + 128U;
  constexpr std::size_t descriptor_offset = second_staging + payload + 128U;
  const auto descriptor_bytes = kvaf::peer::GatherDescriptorBytes(count);
  std::vector<std::uint8_t> memory(descriptor_offset + 2U * descriptor_bytes + 512U);
  const auto base = reinterpret_cast<std::uint64_t>(memory.data());
  const auto first_descriptor = (base + descriptor_offset + 63U) & ~std::uint64_t{63U};
  const auto second_descriptor = (first_descriptor + descriptor_bytes + 63U) & ~std::uint64_t{63U};
  const auto original_spans = [&] {
    std::vector<kvaf::peer::GatherSpan> spans(count);
    for (std::size_t index = 0U; index < count; ++index)
      spans[index] = {base + ((index * 37U) % count) * stride, 656U, 0U};
    return spans;
  }();
  for (const auto threads : {1U, 4U}) {
    kvaf::peer::CpuGatherTeam team(memory.data(), base, threads, 1024U,
                                   kvaf::peer::GatherWaitMode::kActive);
    auto spans = original_spans;
    kvaf::peer::GatherDescriptorHeader header{
        kvaf::peer::kGatherDescriptorMagic, 1U, base + first_staging, count, payload, 8192U};
    auto descriptor = first_descriptor;
    auto region_bytes = memory.size();
    auto write_descriptor = [&] {
      auto* target = memory.data() + (descriptor - base);
      std::memcpy(target, &header, sizeof(header));
      std::memcpy(target + sizeof(header), spans.data(), spans.size() * sizeof(spans[0U]));
    };
    auto copy_and_verify = [&] {
      for (std::size_t index = 0U; index < count * stride; ++index)
        memory[index] = static_cast<std::uint8_t>(index * 17U + header.sequence);
      std::vector<std::uint8_t> expected;
      for (const auto& span : spans)
        expected.insert(expected.end(), memory.data() + (span.address - base),
                        memory.data() + (span.address - base) + span.length);
      Require(expected.size() == header.total_bytes);
      auto* output = memory.data() + (header.staging_address - base);
      std::memset(output, 0xa5, expected.size());
      output[-1] = 0x5aU;
      output[expected.size()] = 0x5aU;
      header.ready_bytes = 0U;
      write_descriptor();
      Require(team.PrepareDirect(region_bytes, descriptor, header.sequence, count) == threads);
      team.Copy();
      Require(std::memcmp(output, expected.data(), expected.size()) == 0);
      Require(output[-1] == 0x5aU && output[expected.size()] == 0x5aU);
      const auto* progress = reinterpret_cast<const std::uint64_t*>(
          memory.data() + (descriptor - base) +
          offsetof(kvaf::peer::GatherDescriptorHeader, ready_bytes));
      if (__atomic_load_n(progress, __ATOMIC_ACQUIRE) !=
          (header.pipeline_chunk_bytes == 0U ? 0U : header.total_bytes))
        std::cerr << "prefix mismatch threads=" << threads << " sequence=" << header.sequence
                  << " chunk=" << header.pipeline_chunk_bytes << " ready=" << *progress
                  << " bytes=" << header.total_bytes << '\n';
      Require(__atomic_load_n(progress, __ATOMIC_ACQUIRE) ==
              (header.pipeline_chunk_bytes == 0U ? 0U : header.total_bytes));
      ++header.sequence;
    };
    for (std::size_t change = 0U; change < 14U; ++change) {
      if (change == 1U) spans.back().address += 1U;
      if (change == 2U) {
        spans[0U].length -= 7U;
        spans[1U].length += 7U;
      }
      if (change == 3U) {
        header.total_bytes -= spans.back().length;
        spans.pop_back();
        --header.span_count;
      }
      if (change == 4U) header.staging_address = base + second_staging;
      if (change == 5U) header.pipeline_chunk_bytes = 4096U;
      if (change == 6U) header.pipeline_chunk_bytes = 0U;
      if (change == 7U) descriptor = second_descriptor;
      if (change == 8U) region_bytes -= 64U;
      if (change == 9U) {
        // Repeated invalid descriptors must fail before any gather copy is started.
        const auto saved = spans.back();
        spans.back().address = header.staging_address;
        for (int attempt = 0; attempt < 2; ++attempt) {
          write_descriptor();
          Require(team.PrepareDirect(region_bytes, descriptor, header.sequence, count) == 0U);
        }
        spans.back() = saved;
      }
      if (change == 10U) {
        // Switch through the legacy API, then prepare another direct descriptor.
        const std::vector<kvaf::peer::GatherSpan> legacy{{base, 128U, 0U}};
        Require(team.Prepare(base + first_staging, legacy, 128U) == 1U);
        team.Copy();
        Require(std::memcmp(memory.data(), memory.data() + first_staging, 128U) == 0);
        header.pipeline_chunk_bytes = 8192U;
      }
      if (change == 11U) {
        spans = original_spans;
        for (std::size_t index = 0U; index < count; ++index)
          spans[index].address = base + index * 656U;
        header.span_count = count;
        header.total_bytes = payload;
      }
      if (change == 12U) {
        // One large source must split at each ready boundary, including the tail.
        spans = {{base, payload, 0U}};
        header.span_count = 1U;
      }
      if (change == 13U) header.pipeline_chunk_bytes = 64U;
      copy_and_verify();
      copy_and_verify();
      // Every descriptor must reject a stale sequence or unreinitialized progress.
      write_descriptor();
      Require(team.PrepareDirect(region_bytes, descriptor, header.sequence + 1U, count) == 0U);
      header.ready_bytes = 64U;
      write_descriptor();
      Require(team.PrepareDirect(region_bytes, descriptor, header.sequence, count) == 0U);
      header.ready_bytes = 0U;
    }
  }
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
  auto* progress = reinterpret_cast<std::uint64_t*>(
      descriptor_memory + offsetof(kvaf::peer::GatherDescriptorHeader, ready_bytes));
  std::vector<kvaf::peer::GatherSpan> spans(span_count);
  std::vector<std::uint8_t> expected;
  kvaf::peer::CpuGatherTeam team(memory.data(), base, 8U, 4096U,
                                 kvaf::peer::GatherWaitMode::kActive);
  for (std::uint64_t sequence = 1U; sequence <= 12U; ++sequence) {
    expected.clear();
    const auto layout = (sequence - 1U) / 3U;
    for (std::size_t index = 0U; index < span_count; ++index) {
      const auto source_offset = ((index + layout) % span_count) * stride;
      const auto length = static_cast<std::uint32_t>((index + layout) % 2U == 0U ? 257U : 513U);
      spans[index] = {base + source_offset, length, 0U};
      for (std::uint32_t byte = 0U; byte < length; ++byte)
        memory[source_offset + byte] = static_cast<std::uint8_t>(index + byte + sequence);
      expected.insert(expected.end(), memory.data() + source_offset,
                      memory.data() + source_offset + length);
    }
    const auto chunk_bytes = layout == 3U ? 0U : layout % 2U == 0U ? 8192U : 4096U;
    const kvaf::peer::GatherDescriptorHeader header{
        kvaf::peer::kGatherDescriptorMagic,          sequence,   base + staging_offset, span_count,
        static_cast<std::uint32_t>(expected.size()), chunk_bytes};
    std::memcpy(descriptor_memory, &header, sizeof(header));
    std::memcpy(descriptor_memory + sizeof(header), spans.data(),
                spans.size() * sizeof(spans.front()));
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
      Require(std::memcmp(memory.data() + staging_offset + observed, expected.data() + observed,
                          ready - observed) == 0);
      observed = ready;
      std::this_thread::yield();
    }
    copier.join();
    Require(__atomic_load_n(progress, __ATOMIC_ACQUIRE) ==
            (chunk_bytes == 0U ? 0U : expected.size()));
    Require(std::memcmp(memory.data() + staging_offset, expected.data(), expected.size()) == 0);
  }
}

void TestCopyDiagnostics() {
  constexpr std::size_t blocks = 7U, block_bytes = 657U, stride = 1024U;
  constexpr std::size_t staging_offset = 8193U, descriptor_offset = 16384U;
  std::vector<std::uint8_t> memory(32768U);
  const auto base = reinterpret_cast<std::uint64_t>(memory.data());
  const auto descriptor = (base + descriptor_offset + 63U) & ~std::uint64_t{63U};
  for (const auto chunk : {0U, 512U})
    for (const bool enabled : {false, true}) {
      kvaf::peer::CpuGatherTeam team(memory.data(), base, 4U, 1024U,
                                     kvaf::peer::GatherWaitMode::kAdaptive, false, enabled);
      kvaf::peer::CpuGatherCopyStats stats(4U);
      for (std::uint64_t round = 1U; round <= 3U; ++round) {
        const auto count = round == 2U ? 1U : blocks;
        std::vector<kvaf::peer::GatherSpan> sources;
        std::vector<std::uint8_t> expected;
        for (std::size_t block = 0; block < count; ++block) {
          const auto offset = ((block + round) % blocks) * stride;
          for (std::size_t byte = 0; byte < block_bytes; ++byte)
            memory[offset + byte] = static_cast<std::uint8_t>(byte + round + block);
          sources.push_back({base + offset, block_bytes, 0U});
          expected.insert(expected.end(), memory.begin() + offset,
                          memory.begin() + offset + block_bytes);
        }
        const kvaf::peer::GatherDescriptorHeader header{kvaf::peer::kGatherDescriptorMagic,
                                                        round,
                                                        base + staging_offset,
                                                        static_cast<std::uint32_t>(count),
                                                        static_cast<std::uint32_t>(expected.size()),
                                                        chunk};
        std::memcpy(reinterpret_cast<void*>(descriptor), &header, sizeof(header));
        std::memcpy(reinterpret_cast<void*>(descriptor + sizeof(header)), sources.data(),
                    sources.size() * sizeof(sources[0]));
        const auto active = round == 2U ? 1U : 4U;
        Require(team.PrepareDirect(memory.size(), descriptor, round, blocks) == active);
        memory[staging_offset - 1U] = memory[staging_offset + expected.size()] = 0xa5;
        team.Copy();
        Require(std::memcmp(memory.data() + staging_offset, expected.data(), expected.size()) == 0);
        Require(memory[staging_offset - 1U] == 0xa5 &&
                memory[staging_offset + expected.size()] == 0xa5);
        const auto& copy = team.copy_profile();
        if (!enabled) {
          Require(copy.workers == 0U);
          continue;
        }
        Require(copy.workers == active && copy.end_ns >= copy.begin_ns);
        Require(copy.join_wait_ns <= copy.end_ns - copy.begin_ns);
        std::uint64_t copied = 0;
        for (std::size_t worker = 0; worker < copy.workers; ++worker) {
          const auto& sample = team.copy_worker_profile(worker);
          copied += sample.bytes;
          Require(sample.begin_ns >= copy.begin_ns && sample.end_ns >= sample.begin_ns);
          Require(sample.end_ns <= copy.end_ns);
          if (!sample.thread_cpu_available) Require(sample.thread_cpu_ns == 0U);
        }
        Require(copied == expected.size());
        stats.Accumulate(team);
      }
      auto* output = std::tmpfile();
      Require(output != nullptr);
      stats.Print(output, 123U, base);
      std::rewind(output);
      std::string text;
      std::array<char, 4096> line{};
      while (std::fgets(line.data(), line.size(), output)) text += line.data();
      std::fclose(output);
      if (!enabled) {
        Require(text.empty());
        continue;
      }
      Require(text.find("rounds=3 active_threads_min=1 active_threads_max=4") != std::string::npos);
      Require(
          text.find("expected_bytes_avg=3285.0 copied_bytes_avg=3285.0 byte_mismatch_rounds=0") !=
          std::string::npos);
      Require(text.find("session=123 worker=0 rounds=3") != std::string::npos);
      for (std::size_t worker = 1; worker < 4; ++worker)
        Require(text.find("worker=" + std::to_string(worker) + " rounds=2") != std::string::npos);
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
  assert(sizeof(kvaf::peer::GatherSpan) == 16U);
  TestParallelGather();
  TestIrregularGather();
  TestDirectDescriptorLayout();
  TestDirectPreparation();
  TestDirectPreparationBounds();
  TestChangingDirectLayouts();
  TestPipelinedGather();
  TestCopyDiagnostics();
  std::cout << "CPU gather: PASS\n";
}
