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
      executor.Copy();
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
}  // namespace

int main() {
  for (const auto mode : {kvaf::peer::GatherWaitMode::kActive,
                          kvaf::peer::GatherWaitMode::kAdaptive,
                          kvaf::peer::GatherWaitMode::kPassive})
    for (bool profile : {false, true})
      for (bool variable : {false, true})
        for (bool pipeline : {false, true}) Run(mode, profile, variable, pipeline);
  std::cout << "CPU gather profile tests passed\n";
}
