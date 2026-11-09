#include "cpu_gather_executor.h"

#include <array>
#include <cstdlib>
#include <iostream>

#define CHECK(condition) do { if (!(condition)) { \
  std::cerr << "line " << __LINE__ << ": " #condition "\n"; std::exit(1); \
} } while (false)

int main() {
  using namespace kvaf::peer;
  std::size_t cases = 0;
  for (auto mode : {GatherWaitMode::kPassive, GatherWaitMode::kAdaptive, GatherWaitMode::kActive}) {
    for (bool profile : {false, true}) {
      for (bool variable : {false, true}) {
        for (std::uint32_t chunk : {0U, 512U}) {
          alignas(64) std::array<std::uint8_t, 131072> memory{};
          const auto base = reinterpret_cast<std::uintptr_t>(memory.data());
          for (std::size_t i = 0; i < 8192; ++i) memory[i] = (i * 19 + i / 256) % 251;
          auto* header = reinterpret_cast<GatherDescriptorHeader*>(memory.data() + 32768);
          auto* spans = reinterpret_cast<GatherSpan*>(header + 1);
          CpuGatherExecutor executor(memory.data(), base, 4, 512, mode, profile);
          std::uint64_t sequence = 0;
          // Grow helpers, shrink the active team, then reuse it. Each request
          // replaces the ready word, exercising per-request timing reset.
          for (std::uint32_t count : {1U, 8U, 3U, 8U}) {
            std::vector<std::uint8_t> expected;
            for (std::uint32_t i = 0; i < count; ++i) {
              const std::uint32_t length = 256U + (variable ? i * 13U : 0U);
              const auto source = (7U - i) * 512U;
              spans[i] = {base + source, length, 0U};
              expected.insert(expected.end(), memory.begin() + source, memory.begin() + source + length);
            }
            *header = {kGatherDescriptorMagic, ++sequence, base + 16384, count,
                       static_cast<std::uint32_t>(expected.size()), chunk, variable ? 0U : 256U};
            const auto threads = executor.PrepareDirect(memory.size(), base + 32768, sequence, 8);
            CHECK(threads == std::min<std::size_t>(4, std::max<std::size_t>(1, expected.size() / 512)));
            executor.Copy();
            CHECK(std::equal(expected.begin(), expected.end(), memory.begin() + 16384));
            if (chunk) CHECK(header->ready_bytes == expected.size());
            const auto& measured = executor.profile();
            if (profile) {
              CHECK(measured.bytes == expected.size());
              CHECK(measured.threads == threads);
              CHECK(measured.copy_ns > 0);
              CHECK(measured.first_ready_ns <= measured.copy_ns);
              CHECK((measured.first_ready_ns > 0) == (chunk > 0));
            } else {
              CHECK(measured.bytes == 0 && measured.copy_ns == 0 && measured.first_ready_ns == 0);
            }
            ++cases;
          }
        }
      }
    }
  }
  std::cout << "Gather executor PASS cases=" << cases << '\n';
}
