#include "cpu_gather_executor.h"

#include <cstdlib>
#include <iostream>

namespace {
void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::abort();
  }
}

void Run(kvaf::peer::GatherWaitMode mode, bool profile, bool variable, bool pipeline) {
  constexpr std::size_t region_bytes = 4U * 1024U * 1024U;
  void* allocation = nullptr;
  Check(posix_memalign(&allocation, 64U, region_bytes) == 0, "allocation failed");
  auto* memory = static_cast<std::uint8_t*>(allocation);
  const auto base = reinterpret_cast<std::uint64_t>(memory);
  {
    kvaf::peer::CpuGatherExecutor executor(memory, base, 4U, 4096U, mode);
    if (profile) executor.EnableProfiling();
    auto* header = reinterpret_cast<kvaf::peer::GatherDescriptorHeader*>(memory);
    auto* spans = reinterpret_cast<kvaf::peer::GatherSpan*>(memory + sizeof(*header));
    std::uint64_t sequence = 0U;
    // Reuse the executor across shrinking/growing generations, including a
    // single-thread generation while previously created helpers are inactive.
    for (const unsigned count : {512U, 1U, 128U, 512U}) {
      ++sequence;
      std::vector<std::uint8_t> expected;
      for (unsigned index = 0U; index < count; ++index) {
        const unsigned length = variable ? 128U + (index % 7U) * 37U : 656U;
        const auto offset = 65536U + (count - index - 1U) * 4096U;
        spans[index] = {base + offset, length, 0U};
        for (unsigned byte = 0U; byte < length; ++byte) {
          memory[offset + byte] = static_cast<std::uint8_t>(sequence + index * 17U + byte);
          expected.push_back(memory[offset + byte]);
        }
      }
      *header = {kvaf::peer::kGatherDescriptorMagic, sequence, base + 3U * 1024U * 1024U,
                 count, static_cast<std::uint32_t>(expected.size()), pipeline ? 4096U : 0U,
                 variable ? 0U : 656U, 0U};
      auto* staging = reinterpret_cast<std::uint8_t*>(header->staging_address);
      std::memset(staging, 0xAD, expected.size());
      const auto threads = executor.PrepareDirect(region_bytes, base, sequence, 512U);
      Check(threads == (count == 1U ? 1U : 4U), "unexpected active thread count");
      std::thread consumer;
      if (pipeline) {
        consumer = std::thread([&] {
          std::uint64_t previous = 0;
          do {
            const auto ready = __atomic_load_n(&header->ready_bytes, __ATOMIC_ACQUIRE);
            Check(ready >= previous && ready <= expected.size(), "invalid ready prefix");
            Check(ready == expected.size() || ready % header->pipeline_chunk_bytes == 0,
                  "ready prefix is not a complete chunk");
            Check(std::memcmp(staging + previous, expected.data() + previous, ready - previous) == 0,
                  "ready was published before its payload");
            previous = ready;
          } while (previous < expected.size());
        });
      }
      executor.Copy();
      if (consumer.joinable()) consumer.join();
      Check(std::memcmp(staging, expected.data(), expected.size()) == 0,
            "profiling changed gathered payload");
      Check(!pipeline || header->ready_bytes == expected.size(), "pipeline not complete");
      if (!profile) {
        Check(executor.ThreadProfiles().empty() && executor.CopyProfile().done_ns == 0U,
              "disabled executor populated profile data");
        continue;
      }
      const auto& copy = executor.CopyProfile();
      Check(copy.publish_ns > 0U && copy.publish_ns <= copy.begin_ns &&
            copy.begin_ns <= copy.coordinator_done_ns && copy.coordinator_done_ns <= copy.done_ns,
            "invalid coordinator timeline");
      std::uint64_t copied = 0U;
      for (std::size_t worker = 0U; worker < threads; ++worker) {
        const auto& thread = executor.ThreadProfiles()[worker];
        Check(copy.publish_ns <= thread.ready_ns && thread.ready_ns <= thread.start_ns &&
              copy.begin_ns <= thread.start_ns && thread.start_ns <= thread.copy_done_ns &&
              thread.copy_done_ns <= thread.done_ns && thread.done_ns <= copy.done_ns,
              "stale or unordered thread timeline");
        Check(thread.cpu_time_valid || thread.thread_cpu_ns == 0U,
              "unavailable thread CPU time reported as a measurement");
#if defined(__linux__)
        Check(thread.tid > 0U && thread.cpu >= 0, "missing Linux worker identity");
#endif
        copied += thread.bytes;
      }
      Check(copied == expected.size(), "profile byte counts omit or double-count a worker");
    }
  }
  std::free(allocation);
}

void BalancedCopies(unsigned count, unsigned token) {
  const std::uint64_t bytes = std::uint64_t(count) * token;
  const auto staging_offset = (131072U + bytes + 63U) & ~std::uint64_t{63U};
  const auto region_bytes = staging_offset + bytes + 64U;
  void* allocation = nullptr;
  Check(posix_memalign(&allocation, 64U, region_bytes) == 0, "allocation failed");
  auto* memory = static_cast<std::uint8_t*>(allocation);
  const auto base = reinterpret_cast<std::uint64_t>(memory);
  {
    kvaf::peer::CpuGatherExecutor executor(memory, base, 16U, 65536U,
                                         kvaf::peer::GatherWaitMode::kActive);
    executor.EnableProfiling();
    auto* header = reinterpret_cast<kvaf::peer::GatherDescriptorHeader*>(memory);
    auto* spans = reinterpret_cast<kvaf::peer::GatherSpan*>(header + 1);
    std::vector<std::uint8_t> expected(bytes);
    for (unsigned i = 0; i < count; ++i) {
      const auto offset = 131072U + std::uint64_t(count - i - 1) * token;
      spans[i] = {base + offset, token, 0};
      std::memset(memory + offset, i % 251, token);
      std::memset(expected.data() + std::uint64_t(i) * token, i % 251, token);
    }
    std::uint64_t sequence = 0;
    for (const auto chunk : {64U, 0U, 131072U, 524288U, 0U, 4194304U}) {
      *header = {kvaf::peer::kGatherDescriptorMagic, ++sequence, base + staging_offset,
                 count, static_cast<std::uint32_t>(bytes), chunk, token, 0};
      auto* staging = memory + staging_offset;
      std::memset(staging, 0xad, bytes);
      Check(executor.PrepareDirect(region_bytes, base, sequence, count) == 16,
            "chunk size reduced useful copy parallelism");
      std::thread consumer;
      if (chunk) consumer = std::thread([&] {
        std::uint64_t previous = 0;
        while (previous < bytes) {
          const auto ready = __atomic_load_n(&header->ready_bytes, __ATOMIC_ACQUIRE);
          Check(ready >= previous && ready <= bytes, "ready prefix regressed");
          Check(ready == bytes || ready % chunk == 0, "misaligned ready publication");
          Check(std::memcmp(staging + previous, expected.data() + previous, ready - previous) == 0,
                "published task prefix contains unfinished payload");
          previous = ready;
        }
      });
      executor.Copy();
      if (consumer.joinable()) consumer.join();
      Check(std::memcmp(staging, expected.data(), bytes) == 0, "balanced copy payload mismatch");
      for (const auto& profile : executor.ThreadProfiles())
        Check(profile.bytes == bytes / 16, "copy work still depends on ready chunk size");
    }
  }
  std::free(allocation);
}
}  // namespace

int main() {
  for (const auto mode : {kvaf::peer::GatherWaitMode::kActive,
                          kvaf::peer::GatherWaitMode::kAdaptive,
                          kvaf::peer::GatherWaitMode::kPassive})
    for (bool profile : {false, true})
      for (bool variable : {false, true})
        for (bool pipeline : {false, true}) Run(mode, profile, variable, pipeline);
  BalancedCopies(4096, 656);   // Previously 21 tasks for 16 threads at 128 KiB.
  BalancedCopies(4096, 4096);  // Multiple balanced rounds, bounded copy task size.
  std::cout << "CPU gather profile tests passed\n";
}
