#include "sq_copy_benchmark.h"
#include "wqe.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <time.h>

#if !defined(__aarch64__) && !defined(KVAF_AICPU_URMA_HOST_TEST)
#error "The SQ copy benchmark must be compiled for aarch64"
#endif

namespace {

std::uint64_t NowNs() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool ThreadCpuNs(std::uint64_t* nanoseconds) {
  timespec value{};
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0) return false;
  *nanoseconds = static_cast<std::uint64_t>(value.tv_sec) * 1000000000ULL + value.tv_nsec;
  return true;
}

void Barrier() {
#if defined(KVAF_AICPU_URMA_HOST_TEST)
  std::atomic_thread_fence(std::memory_order_seq_cst);
#else
  asm volatile("dsb sy" ::: "memory");
#endif
}

}

extern "C" __attribute__((visibility("default")))
std::uint32_t KvafAicpuUrmaSqCopyBenchmark(kvaf::aicpu_urma::SqCopyBenchmarkArgs* args) {
  using namespace kvaf::aicpu_urma;
  const auto entry_ns = NowNs();
  if (args == nullptr || args->queues == 0U || args->statuses == 0U ||
      args->next_worker == 0U || args->workers == 0U ||
      args->workers > kSqCopyBenchmarkMaxWorkers || args->target_sq > 2U ||
      (args->batch_wqes != 8U && args->batch_wqes != 16U &&
       args->batch_wqes != 32U && args->batch_wqes != 64U) ||
      !kvaf::ValidGatherSqCopyMode(args->copy_mode)) return 1U;
  const auto worker = __atomic_fetch_add(reinterpret_cast<std::uint64_t*>(args->next_worker),
                                         1U, __ATOMIC_RELAXED);
  if (worker >= args->workers) return 1U;
  auto& status = reinterpret_cast<SqCopyBenchmarkStatus*>(args->statuses)[worker];
  status = {};
  status.magic = kSqCopyBenchmarkMagic;
  status.result = 1U;
  status.worker = worker;
  status.round = args->round;
  status.entry_ns = entry_ns;
  const auto& queue = reinterpret_cast<const SqCopyBenchmarkQueue*>(args->queues)[worker];
  const auto destination = args->target_sq == 1U ? queue.sq : queue.hbm;
  if (destination == 0U || destination % 64U != 0U || queue.bytes == 0U ||
      queue.bytes % 64U != 0U || queue.offset % 64U != 0U ||
      queue.capacity > kSqCopyBenchmarkQueueBytes || queue.offset > queue.capacity ||
      queue.bytes > queue.capacity - queue.offset ||
      destination > UINT64_MAX - queue.capacity ||
      (args->target_sq == 2U && (queue.source_hbm == 0U ||
       queue.source_hbm % 64U != 0U || queue.source_hbm > UINT64_MAX - queue.bytes ||
       (queue.source_hbm < destination + queue.offset + queue.bytes &&
        destination + queue.offset < queue.source_hbm + queue.bytes)))) {
    Barrier();
    return 1U;
  }
  alignas(64) std::uint32_t source[kScatterBatchWqes * kScatterWqeWords];
  const auto chunk_blocks = args->batch_wqes * kMaxScatterWqebbs;
  const auto chunk_bytes = chunk_blocks * kWqeBytes;
  auto* source_bytes = reinterpret_cast<std::uint8_t*>(source);
  if (args->target_sq != 2U) {
    for (std::uint32_t index = 0U; index < chunk_bytes; ++index)
      source_bytes[index] = SqCopyBenchmarkPattern(index, static_cast<std::uint32_t>(worker), args->round);
  }
  auto* output = reinterpret_cast<std::uint8_t*>(destination + queue.offset);
  std::uint64_t cpu_begin_ns = 0U;
  std::uint64_t cpu_end_ns = 0U;
  Barrier();
  const bool cpu_available = ThreadCpuNs(&cpu_begin_ns);
  status.copy_begin_ns = NowNs();
  for (std::uint64_t offset = 0U; offset < queue.bytes; offset += chunk_bytes) {
    const auto blocks = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(chunk_bytes, queue.bytes - offset) / kWqeBytes);
    const auto* input = args->target_sq == 2U
        ? reinterpret_cast<const std::uint8_t*>(queue.source_hbm) + offset : source_bytes;
    CopySqWqebbs(output + offset, input, blocks, args->copy_mode);
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "r"(output + offset), "r"(input) : "memory");
#endif
  }
  Barrier();
  status.copy_end_ns = NowNs();
  if (cpu_available && ThreadCpuNs(&cpu_end_ns) && cpu_end_ns >= cpu_begin_ns) {
    status.thread_cpu_available = 1U;
    status.thread_cpu_ns = cpu_end_ns - cpu_begin_ns;
  }
  status.bytes = queue.bytes;
  status.copy_calls = (queue.bytes + chunk_bytes - 1U) / chunk_bytes;
  status.result = 0U;
  Barrier();
  return 0U;
}

extern "C" __attribute__((visibility("default")))
std::uint32_t KvafAicpuUrmaSqCopyVerify(kvaf::aicpu_urma::SqCopyBenchmarkArgs* args) {
  using namespace kvaf::aicpu_urma;
  if (args == nullptr || args->queues == 0U || args->statuses == 0U ||
      args->next_verify_worker == 0U || args->workers == 0U ||
      args->workers > kSqCopyBenchmarkMaxWorkers || args->target_sq > 2U ||
      (args->batch_wqes != 8U && args->batch_wqes != 16U &&
       args->batch_wqes != 32U && args->batch_wqes != 64U)) return 1U;
  const auto worker = __atomic_fetch_add(reinterpret_cast<std::uint64_t*>(args->next_verify_worker),
                                         1U, __ATOMIC_RELAXED);
  if (worker >= args->workers) return 1U;
  auto& status = reinterpret_cast<SqCopyBenchmarkStatus*>(args->statuses)[worker];
  const auto& queue = reinterpret_cast<const SqCopyBenchmarkQueue*>(args->queues)[worker];
  const auto destination = args->target_sq == 1U ? queue.sq : queue.hbm;
  if (status.magic != kSqCopyBenchmarkMagic || status.worker != worker ||
      status.round != args->round || status.result != 0U || status.bytes != queue.bytes ||
      destination == 0U || destination % 64U != 0U || queue.bytes == 0U ||
      queue.bytes % 64U != 0U || queue.offset % 64U != 0U ||
      queue.capacity > kSqCopyBenchmarkQueueBytes || queue.offset > queue.capacity ||
      queue.bytes > queue.capacity - queue.offset || destination > UINT64_MAX - queue.capacity) {
    status.result = 3U;
    Barrier();
    return 1U;
  }
  const auto chunk_bytes = args->batch_wqes * kMaxScatterWqebbs * kWqeBytes;
  const auto* output = reinterpret_cast<const std::uint8_t*>(destination + queue.offset);
  const auto* words = reinterpret_cast<const volatile std::uint32_t*>(output);
  for (std::uint64_t offset = 0U; offset < queue.bytes; offset += sizeof(std::uint32_t)) {
    std::uint32_t expected = 0U;
    for (std::uint32_t byte = 0U; byte < sizeof(expected); ++byte)
      expected |= static_cast<std::uint32_t>(SqCopyBenchmarkPattern(
          args->target_sq == 2U ? offset + byte : (offset + byte) % chunk_bytes,
          static_cast<std::uint32_t>(worker), args->round)) << (byte * 8U);
    const auto actual = words[offset / sizeof(std::uint32_t)];
    if (actual != expected) {
      status.result = 2U;
      status.mismatch_offset = queue.offset + offset;
      status.expected = expected;
      status.actual = actual;
      Barrier();
      return 1U;
    }
  }
  status.result = 0U;
  Barrier();
  return 0U;
}
