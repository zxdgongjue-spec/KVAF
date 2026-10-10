#pragma once

#include "gather_output.h"

#include <cstdint>

namespace kvaf::aicpu_urma {

constexpr std::uint64_t kSqCopyBenchmarkMagic = 0x4b56414653514350ULL;
constexpr std::uint32_t kSqCopyBenchmarkMaxWorkers = 8U;
constexpr std::uint64_t kSqCopyBenchmarkQueueBytes = 8192ULL * 4U * 64U;

struct SqCopyBenchmarkQueue {
  std::uint64_t sq;
  std::uint64_t hbm;
  std::uint64_t capacity;
  std::uint64_t offset;
  std::uint64_t bytes;
};

struct alignas(64) SqCopyBenchmarkStatus {
  std::uint64_t magic;
  std::uint64_t result;
  std::uint64_t worker;
  std::uint64_t round;
  std::uint64_t entry_ns;
  std::uint64_t copy_begin_ns;
  std::uint64_t copy_end_ns;
  std::uint64_t thread_cpu_ns;
  std::uint64_t thread_cpu_available;
  std::uint64_t bytes;
  std::uint64_t copy_calls;
  std::uint64_t mismatch_offset;
  std::uint64_t expected;
  std::uint64_t actual;
  std::uint64_t reserved[2];
};

struct SqCopyBenchmarkArgs {
  std::uint64_t queues;
  std::uint64_t statuses;
  std::uint64_t next_worker;
  std::uint64_t round;
  std::uint32_t workers;
  std::uint32_t batch_wqes;
  std::uint32_t target_sq;
  GatherSqCopyMode copy_mode;
  std::uint64_t next_verify_worker;
};

inline std::uint8_t SqCopyBenchmarkPattern(std::uint64_t index,
                                          std::uint32_t worker, std::uint64_t round) {
  return static_cast<std::uint8_t>(index * 131U + index / 64U * 17U + worker * 29U + round * 7U);
}

static_assert(sizeof(SqCopyBenchmarkQueue) == 40U, "SQ copy queue ABI changed");
static_assert(sizeof(SqCopyBenchmarkStatus) == 128U, "SQ copy status ABI changed");
static_assert(sizeof(SqCopyBenchmarkArgs) == 56U, "SQ copy arguments ABI changed");

}
