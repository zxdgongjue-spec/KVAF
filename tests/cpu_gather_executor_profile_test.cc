#include "cpu_gather_executor.h"

#include <cstdlib>
#include <iostream>

#if defined(KVAF_CPU_GATHER_COPY_TEST_HOOK)
namespace cpu_gather_test {
constexpr std::uint64_t kFirstTask = 164U * 1024U;
std::atomic<bool> hold_tail{false}, tail_entered{false}, release_tail{false};
std::atomic<std::uint64_t> held_begin{0U}, held_end{0U};
std::atomic<std::uint64_t> wait_generation{0U}, parked_helpers{0U}, spinning_helpers{0U};
std::atomic<bool> hold_finished_helper{false}, helper_finished{false}, release_helper{false};
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

void CpuGatherHelperCompleted(std::size_t worker, std::uint64_t generation) {
  using namespace cpu_gather_test;
  if (worker != 1U || generation != wait_generation.load(std::memory_order_acquire) ||
      !hold_finished_helper.load(std::memory_order_acquire))
    return;
  // Completion is already visible to Copy(). Hold this helper so the test can
  // deterministically exercise its idle decision AFTER the whole copy ends.
  helper_finished.store(true, std::memory_order_release);
  while (!release_helper.load(std::memory_order_acquire)) GatherSpinPause();
}

void CpuGatherHelperWait(std::size_t worker, std::uint64_t generation, bool spinning) {
  using namespace cpu_gather_test;
  if (generation == 0U || generation != wait_generation.load(std::memory_order_acquire)) return;
  (spinning ? spinning_helpers : parked_helpers).fetch_or(
      std::uint64_t{1U} << worker, std::memory_order_release);
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
template <class Predicate>
void Await(Predicate predicate, const char* message) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!predicate() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  Check(predicate(), message);
}

void HelperWaitPolicy(kvaf::peer::GatherWaitMode mode, bool hold_coordinator,
                      bool cooperative) {
  using namespace cpu_gather_test;
  constexpr unsigned count = 512U, token = 656U, threads = 4U;
  constexpr std::uint64_t bytes = std::uint64_t(count) * token;
  constexpr auto staging_offset = 65536U + bytes;
  constexpr auto region_bytes = staging_offset + bytes + 64U;
  void* allocation = nullptr;
  Check(posix_memalign(&allocation, 64U, region_bytes) == 0, "wait test allocation failed");
  auto* memory = static_cast<std::uint8_t*>(allocation);
  const auto base = reinterpret_cast<std::uint64_t>(memory);
  {
    kvaf::peer::CpuGatherExecutor executor(memory, base, threads, 4096U, mode);
    auto* header = reinterpret_cast<kvaf::peer::GatherDescriptorHeader*>(memory);
    auto* spans = reinterpret_cast<kvaf::peer::GatherSpan*>(header + 1);
    std::vector<std::uint8_t> expected(bytes);
    for (unsigned index = 0U; index < count; ++index) {
      const auto offset = 65536U + std::uint64_t(count - index - 1U) * token;
      spans[index] = {base + offset, token, 0U};
      std::memset(memory + offset, index % 251U, token);
      std::memset(expected.data() + std::uint64_t(index) * token, index % 251U, token);
    }
    const auto chunk = cooperative ? kvaf::GatherCooperativeChunkBytes(bytes, 4U) : 0U;
    *header = {cooperative ? kvaf::kGatherCooperativeMagic : kvaf::peer::kGatherDescriptorMagic,
               1U, base + staging_offset, count, static_cast<std::uint32_t>(bytes), chunk, token, 0U};
    std::memset(memory + staging_offset, 0xad, bytes);
    parked_helpers.store(0U); spinning_helpers.store(0U);
    helper_finished.store(false); release_helper.store(false);
    hold_finished_helper.store(!hold_coordinator);
    tail_entered.store(false); release_tail.store(false);
    held_begin.store(0U);
    held_end.store((cooperative ? chunk : bytes) / threads);
    hold_tail.store(hold_coordinator, std::memory_order_release);
    wait_generation.store(1U, std::memory_order_release);
    Check(executor.PrepareDirect(region_bytes, base, 1U, count) == threads,
          "wait test lost copy parallelism");
    constexpr std::uint64_t all_helpers = 14U;
    if (hold_coordinator) {
      std::thread driver([&] { executor.Copy(); });
      Await([&] { return tail_entered.load(std::memory_order_acquire) &&
                         parked_helpers.load(std::memory_order_acquire) == all_helpers; },
            "finished helpers did not park while coordinator was copying");
      Check(spinning_helpers.load() == 0U, "finished helper spun during unfinished copy");
      release_tail.store(true, std::memory_order_release);
      driver.join();
    } else {
      executor.Copy();
      Await([&] { return helper_finished.load(std::memory_order_acquire) &&
                         (parked_helpers.load(std::memory_order_acquire) & 12U) == 12U; },
            "non-warm helpers did not park after copy");
      release_helper.store(true, std::memory_order_release);
      Await([&] { return parked_helpers.load(std::memory_order_acquire) == all_helpers; },
            "warm helper did not exhaust its bounded spin budget");
      Check(spinning_helpers.load() ==
                (mode == kvaf::peer::GatherWaitMode::kPassive ? 0U : 2U),
            "idle spin was not limited to helper 1");
    }
    hold_tail.store(false); hold_finished_helper.store(false);
    wait_generation.store(0U, std::memory_order_release);
    Check(std::memcmp(memory + staging_offset, expected.data(), bytes) == 0,
          "helper wait policy changed gathered payload");
    // Repeated immediate handoffs exercise the spin-to-sleep race and helpers
    // becoming inactive for one generation, then participating again.
    for (std::uint64_t sequence = 2U; sequence <= 65U; ++sequence) {
      const auto spans_count = sequence % 2U == 0U ? 1U : count;
      const auto copy_bytes = spans_count * token;
      header->sequence = sequence;
      header->span_count = spans_count;
      header->total_bytes = copy_bytes;
      header->pipeline_chunk_bytes = cooperative
          ? kvaf::GatherCooperativeChunkBytes(copy_bytes, 4U) : 0U;
      header->ready_bytes = 0U;
      std::memset(memory + staging_offset, 0xad, copy_bytes);
      Check(executor.PrepareDirect(region_bytes, base, sequence, count) ==
                (spans_count == 1U ? 1U : threads), "wait test thread count did not recover");
      executor.Copy();
      Check(std::memcmp(memory + staging_offset, expected.data(), copy_bytes) == 0,
            "helper missed a generation after parking");
    }
    // Shutdown must also release helpers already dispatched to a generation
    // whose Copy() has not started yet.
    header->sequence = 66U; header->ready_bytes = 0U;
    Check(executor.PrepareDirect(region_bytes, base, 66U, count) == threads,
          "could not prepare shutdown generation");
  }
  std::free(allocation);
}

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
void CooperativeCopies(bool profile, unsigned token, unsigned chunks, bool pause_tail = false,
                       bool pause_first = false) {
  constexpr unsigned max_count = 4097, max_threads = 16;
  const auto max_bytes = std::uint64_t(max_count) * token;
  const auto staging_offset = (131072U + max_bytes + 63U) & ~std::uint64_t{63};
  const auto region_bytes = staging_offset + max_bytes + 128U;
  void* allocation = nullptr;
  Check(posix_memalign(&allocation, 64, region_bytes) == 0, "cooperative allocation failed");
  auto* memory = static_cast<std::uint8_t*>(allocation);
  const auto base = reinterpret_cast<std::uint64_t>(memory);
  {
    kvaf::peer::CpuGatherExecutor executor(memory, base, max_threads, 65536,
                                         kvaf::peer::GatherWaitMode::kActive);
    if (profile) executor.EnableProfiling();
    auto* h = reinterpret_cast<kvaf::peer::GatherDescriptorHeader*>(memory);
    auto* spans = reinterpret_cast<kvaf::peer::GatherSpan*>(h + 1);
    std::uint64_t sequence = 0;
    for (unsigned count : {4096U, 1U, 19U, 4097U}) {
      const auto bytes = std::uint64_t(count) * token;
      const auto chunk_bytes = kvaf::GatherCooperativeChunkBytes(bytes, chunks);
      std::vector<std::uint8_t> expected(bytes);
      for (unsigned i = 0; i < count; ++i) {
        const auto offset = 131072U + std::uint64_t(count - i - 1) * token;
        spans[i] = {base + offset, token, 0};
        std::memset(memory + offset, (i + sequence) % 251, token);
        std::memset(expected.data() + std::uint64_t(i) * token, (i + sequence) % 251, token);
      }
      *h = {kvaf::kGatherCooperativeMagic, ++sequence, base + staging_offset,
            count, static_cast<std::uint32_t>(bytes), chunk_bytes, token, 0};
      auto* staging = memory + staging_offset;
      std::memset(staging, 0xad, max_bytes + 64);
      const auto threads = executor.PrepareDirect(region_bytes, base, sequence, max_count);
      Check(threads == std::min<std::size_t>(max_threads, std::max<std::size_t>(1, bytes / 65536)),
            "cooperative blocks changed useful copy parallelism");
#if defined(KVAF_CPU_GATHER_COPY_TEST_HOOK)
      const bool hold = count == 4096 && (pause_tail || pause_first);
      if (hold) {
        using namespace cpu_gather_test;
        const auto share = chunk_bytes / threads;
        tail_entered.store(false); release_tail.store(false);
        held_begin.store(pause_first ? share : chunk_bytes);
        held_end.store(held_begin.load() + share);
        hold_tail.store(true, std::memory_order_release);
      }
#endif
      std::thread consumer([&] {
#if defined(KVAF_CPU_GATHER_COPY_TEST_HOOK)
        if (hold) {
          using namespace cpu_gather_test;
          const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
          while (!tail_entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
          Check(tail_entered.load(), "cooperative share was not copied in block order");
          if (pause_first) {
            Check(__atomic_load_n(&h->ready_bytes, __ATOMIC_ACQUIRE) == 0,
                  "cooperative prefix crossed an unfinished worker share");
          } else {
            while (__atomic_load_n(&h->ready_bytes, __ATOMIC_ACQUIRE) < chunk_bytes &&
                   std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
            Check(__atomic_load_n(&h->ready_bytes, __ATOMIC_ACQUIRE) == chunk_bytes,
                  "a copying coordinator blocked another worker's ready publication");
            Check(std::memcmp(staging, expected.data(), chunk_bytes) == 0,
                  "cooperative first block was published before payload");
          }
          Check(std::all_of(staging + held_begin.load(), staging + held_end.load(),
                            [](std::uint8_t b) { return b == 0xad; }), "held share was overwritten");
          release_tail.store(true, std::memory_order_release);
        }
#endif
        std::uint64_t previous = 0;
        while (previous < bytes) {
          const auto ready = __atomic_load_n(&h->ready_bytes, __ATOMIC_ACQUIRE);
          Check(ready >= previous && ready <= bytes && (ready == bytes || ready % chunk_bytes == 0),
                "invalid cooperative ready prefix");
          Check(std::memcmp(staging + previous, expected.data() + previous, ready - previous) == 0,
                "cooperative publication preceded its payload");
          previous = ready;
        }
      });
      executor.Copy(); consumer.join();
#if defined(KVAF_CPU_GATHER_COPY_TEST_HOOK)
      cpu_gather_test::hold_tail.store(false, std::memory_order_release);
#endif
      Check(std::memcmp(staging, expected.data(), bytes) == 0, "cooperative payload mismatch");
      Check(std::all_of(staging + bytes, staging + max_bytes + 64,
                        [](std::uint8_t b) { return b == 0xad; }), "cooperative tail guard overwritten");
      if (profile) {
        std::uint64_t copied = 0;
        for (std::size_t worker = 0; worker < threads; ++worker) {
          const auto& thread = executor.ThreadProfiles()[worker];
          copied += thread.bytes;
          Check(thread.progress_publications == (bytes + chunk_bytes - 1) / chunk_bytes,
                "cooperative worker did not visit each block exactly once");
        }
        Check(copied == bytes, "cooperative profiles omitted or duplicated bytes");
      }
    }
    h->magic = kvaf::peer::kGatherDescriptorMagic;
    h->pipeline_chunk_bytes = 0; h->ready_bytes = 0;
    Check(executor.PrepareDirect(region_bytes, base, sequence, max_count) != 0,
          "cooperative executor could not switch back to whole-batch");
    executor.Copy();
    h->magic = kvaf::kGatherCooperativeMagic;
    Check(!executor.PrepareDirect(region_bytes, base, sequence, max_count),
          "cooperative descriptor without chunks was accepted");
    h->pipeline_chunk_bytes = 64;
    Check(!executor.PrepareDirect(region_bytes, base, sequence, max_count),
          "unbounded cooperative chunk count was accepted");
    h->magic = 0x4b56414647445233ULL;  // Retired descriptor format.
    h->pipeline_chunk_bytes = 65536;
    h->ready_bytes = 6;
    Check(!executor.PrepareDirect(region_bytes, base, sequence, max_count),
          "retired descriptor format was accepted");
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
  for (bool profile : {false, true})
    for (unsigned token : {656U, 4096U})
      for (unsigned chunks : {2U, 4U, 8U}) CooperativeCopies(profile, token, chunks);
#if defined(KVAF_CPU_GATHER_COPY_TEST_HOOK)
  for (const auto mode : {kvaf::peer::GatherWaitMode::kActive,
                          kvaf::peer::GatherWaitMode::kAdaptive,
                          kvaf::peer::GatherWaitMode::kPassive}) {
    HelperWaitPolicy(mode, true, false);
    HelperWaitPolicy(mode, true, true);
    HelperWaitPolicy(mode, false, false);
  }
  CooperativeCopies(false, 656, 4, true);
  CooperativeCopies(true, 656, 4, true);
  CooperativeCopies(false, 656, 4, false, true);
  for (const auto chunk : {65536U, 131072U}) {
    PublishBeforeTaskTail(false, chunk);
    PublishBeforeTaskTail(true, chunk);
  }
#endif
  std::cout << "CPU gather profile tests passed\n";
}
