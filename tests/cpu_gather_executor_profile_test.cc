#include "cpu_gather_executor.h"

#include <cstdlib>
#include <iostream>

#if defined(KVAF_CPU_GATHER_COPY_TEST_HOOK)
namespace cpu_gather_test {
constexpr std::uint64_t kFirstTask = 164U * 1024U;
std::atomic<bool> hold_tail{false}, tail_entered{false}, release_tail{false};
std::atomic<std::uint64_t> held_begin{0U}, held_end{0U};
}  // namespace cpu_gather_test

namespace kvaf::peer {
void CpuGatherBeforeCopyRange(std::uint64_t begin, std::uint64_t end) {
  using namespace cpu_gather_test;
  if (!hold_tail.load(std::memory_order_acquire) ||
      begin != held_begin.load(std::memory_order_relaxed) ||
      end != held_end.load(std::memory_order_relaxed))
    return;
  tail_entered.store(true, std::memory_order_release);
  while (!release_tail.load(std::memory_order_acquire)) GatherSpinPause();
}
}  // namespace kvaf::peer
#endif

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
    for (const auto chunk : {64U, 0U, 65536U, 131072U, 524288U, 0U, 4194304U}) {
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

#if defined(KVAF_CPU_GATHER_COPY_TEST_HOOK)
void PublishBeforeTaskTail(bool profile, std::uint32_t chunk) {
  using namespace cpu_gather_test;
  constexpr unsigned count = 4096U, token = 656U;
  constexpr std::uint64_t bytes = std::uint64_t(count) * token;
  constexpr auto staging_offset = 131072U + bytes;
  constexpr auto region_bytes = staging_offset + bytes + 64U;
  void* allocation = nullptr;
  Check(posix_memalign(&allocation, 64U, region_bytes) == 0, "allocation failed");
  auto* memory = static_cast<std::uint8_t*>(allocation);
  const auto base = reinterpret_cast<std::uint64_t>(memory);
  {
    kvaf::peer::CpuGatherExecutor executor(memory, base, 16U, 65536U,
                                         kvaf::peer::GatherWaitMode::kActive);
    if (profile) executor.EnableProfiling();
    auto* header = reinterpret_cast<kvaf::peer::GatherDescriptorHeader*>(memory);
    auto* spans = reinterpret_cast<kvaf::peer::GatherSpan*>(header + 1);
    std::vector<std::uint8_t> expected(bytes);
    for (unsigned index = 0U; index < count; ++index) {
      const auto offset = 131072U + std::uint64_t(count - index - 1U) * token;
      spans[index] = {base + offset, token, 0U};
      std::memset(memory + offset, index % 251U, token);
      std::memset(expected.data() + std::uint64_t(index) * token, index % 251U, token);
    }
    *header = {kvaf::peer::kGatherDescriptorMagic, 1U, base + staging_offset,
               count, static_cast<std::uint32_t>(bytes), chunk, token, 0U};
    auto* staging = memory + staging_offset;
    std::memset(staging, 0xad, bytes);
    Check(executor.PrepareDirect(region_bytes, base, 1U, count) == 16U,
          "early publication changed copy parallelism");
    tail_entered.store(false, std::memory_order_relaxed);
    release_tail.store(false, std::memory_order_relaxed);
    held_begin.store(chunk, std::memory_order_relaxed);
    held_end.store(std::min(kFirstTask, std::uint64_t(chunk) * 2U), std::memory_order_relaxed);
    hold_tail.store(true, std::memory_order_release);
    std::atomic<bool> copy_done{false};
    std::thread consumer([&] {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (!tail_entered.load(std::memory_order_acquire) &&
             !copy_done.load(std::memory_order_acquire) &&
             std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
      Check(tail_entered.load(std::memory_order_acquire),
            "first task was not split at the ready boundary");
      Check(__atomic_load_n(&header->ready_bytes, __ATOMIC_ACQUIRE) == chunk,
            "first chunk was not published before copying the task tail");
      Check(std::memcmp(staging, expected.data(), chunk) == 0,
            "first ready prefix contains unfinished payload");
      Check(std::all_of(staging + chunk, staging + kFirstTask,
                        [](std::uint8_t byte) { return byte == 0xad; }),
            "task tail was copied before first prefix publication");
      release_tail.store(true, std::memory_order_release);
    });
    executor.Copy();
    copy_done.store(true, std::memory_order_release);
    consumer.join();
    hold_tail.store(false, std::memory_order_release);
    Check(header->ready_bytes == bytes, "early publication omitted the final prefix");
    Check(std::memcmp(staging, expected.data(), bytes) == 0, "early publication payload mismatch");
    if (profile)
      for (const auto& thread : executor.ThreadProfiles())
        Check(thread.bytes == kFirstTask, "partial publication changed balanced task ownership");
  }
  std::free(allocation);
}
#endif
void ShardedCopies(bool profile, unsigned token, unsigned sgl, unsigned chunk) {
  constexpr unsigned count = 4097;
  const auto bytes = std::uint64_t(count) * token;
  const auto staging_offset = (131072U + bytes + 63U) & ~std::uint64_t{63};
  const auto region_bytes = staging_offset + ((bytes + 63U) & ~std::uint64_t{63}) + 64;
  void* allocation = nullptr;
  Check(posix_memalign(&allocation, 64, region_bytes) == 0, "allocation failed");
  auto* memory = static_cast<std::uint8_t*>(allocation);
  const auto base = reinterpret_cast<std::uint64_t>(memory);
  {
    kvaf::peer::CpuGatherExecutor executor(memory, base, 8, 4096,
                                         kvaf::peer::GatherWaitMode::kActive);
    if (profile) executor.EnableProfiling();
    auto* h = reinterpret_cast<kvaf::peer::GatherDescriptorHeader*>(memory);
    auto* spans = reinterpret_cast<kvaf::peer::GatherSpan*>(h + 1);
    auto* progress = reinterpret_cast<std::uint64_t*>(memory + kvaf::GatherProgressOffset(count));
    const auto wrs = (count + sgl - 1) / sgl;
    const auto shards = kvaf::GatherShardCount(wrs);
    std::memset(progress, 0, kvaf::kGatherProgressBytes);
    std::vector<std::uint8_t> expected(bytes);
    for (unsigned sequence = 1; sequence <= 3; ++sequence) {
      for (unsigned i = 0; i < count; ++i) {
        const auto offset = 131072U + std::uint64_t(count - i - 1) * token;
        spans[i] = {base + offset, token, 0};
        std::memset(memory + offset, (i + sequence) % 251, token);
        std::memset(expected.data() + std::uint64_t(i) * token, (i + sequence) % 251, token);
      }
      *h = {kvaf::kGatherShardMagic, sequence, base + staging_offset, count,
            static_cast<std::uint32_t>(bytes), chunk, token, sgl};
      auto* staging = memory + staging_offset;
      std::memset(staging, 0xad, bytes);
      Check(executor.PrepareDirect(region_bytes, base, sequence, count) == 8,
            "shard preparation failed");
#if defined(KVAF_CPU_GATHER_COPY_TEST_HOOK)
      using namespace cpu_gather_test;
      const auto first_end = std::min<std::uint64_t>(
          kvaf::GatherShardBoundary(wrs, shards, 1), std::max<std::uint64_t>(1, chunk / (sgl * token))) * sgl * token;
      held_begin.store(0); held_end.store(first_end);
      tail_entered.store(false); release_tail.store(false); hold_tail.store(true);
#endif
      std::thread consumer([&] {
        std::vector<unsigned> observed(shards);
#if defined(KVAF_CPU_GATHER_COPY_TEST_HOOK)
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while ((__atomic_load_n(progress + shards - 1, __ATOMIC_ACQUIRE) >> 32) != sequence &&
               std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
        Check((__atomic_load_n(progress + shards - 1, __ATOMIC_ACQUIRE) >> 32) == sequence,
              "late shard cannot progress while first shard is blocked");
        Check((__atomic_load_n(progress, __ATOMIC_ACQUIRE) >> 32) != sequence,
              "blocked first shard advertised new progress");
        release_tail.store(true, std::memory_order_release);
#endif
        unsigned finished = 0;
        while (finished < shards) {
          finished = 0;
          for (unsigned shard = 0; shard < shards; ++shard) {
            const auto value = __atomic_load_n(progress + shard, __ATOMIC_ACQUIRE);
            if ((value >> 32) != sequence) continue;
            const auto first = kvaf::GatherShardBoundary(wrs, shards, shard);
            const auto size = kvaf::GatherShardBoundary(wrs, shards, shard + 1) - first;
            const auto ready = static_cast<unsigned>(value);
            Check(ready >= observed[shard] && ready <= size, "invalid per-shard progress");
            const auto begin = std::min(bytes, std::uint64_t(first + observed[shard]) * sgl * token);
            const auto end = std::min(bytes, std::uint64_t(first + ready) * sgl * token);
            Check(std::memcmp(staging + begin, expected.data() + begin, end - begin) == 0,
                  "shard publication preceded its payload");
            observed[shard] = ready;
            finished += ready == size;
          }
        }
      });
      executor.Copy(); consumer.join();
#if defined(KVAF_CPU_GATHER_COPY_TEST_HOOK)
      cpu_gather_test::hold_tail.store(false);
#endif
      Check(std::memcmp(staging, expected.data(), bytes) == 0, "shard output mismatch");
      if (profile) {
        std::uint64_t copied = 0;
        for (const auto& thread : executor.ThreadProfiles()) copied += thread.bytes;
        Check(copied == bytes, "shard profiles double-count or omit bytes");
      }
    }
    const auto header = *h;
    for (unsigned invalid : {0U, 14U}) {
      h->ready_bytes = invalid;
      Check(!executor.PrepareDirect(region_bytes, base, 3, count), "invalid READ group accepted");
    }
    *h = header;
    Check(!executor.PrepareDirect(kvaf::GatherProgressOffset(count) + 64, base, 3, count),
          "missing shard trailer accepted");
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
  for (bool profile : {false, true})
    for (unsigned token : {656U, 4096U})
      for (unsigned sgl : {1U, 6U, 13U})
        for (unsigned chunk : {64U, 4096U, 131072U}) ShardedCopies(profile, token, sgl, chunk);
  BalancedCopies(4096, 656);   // Previously 21 tasks for 16 threads at 128 KiB.
  BalancedCopies(4096, 4096);  // Multiple balanced rounds, bounded copy task size.
#if defined(KVAF_CPU_GATHER_COPY_TEST_HOOK)
  for (const auto chunk : {65536U, 131072U}) {
    PublishBeforeTaskTail(false, chunk);
    PublishBeforeTaskTail(true, chunk);
  }
#endif
  std::cout << "CPU gather profile tests passed\n";
}
