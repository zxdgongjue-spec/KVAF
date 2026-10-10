#include "aicpu_urma/sq_copy_benchmark.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <thread>
#include <vector>

extern "C" std::uint32_t KvafAicpuUrmaSqCopyBenchmark(
    kvaf::aicpu_urma::SqCopyBenchmarkArgs* args);
extern "C" std::uint32_t KvafAicpuUrmaSqCopyVerify(
    kvaf::aicpu_urma::SqCopyBenchmarkArgs* args);

namespace {

using namespace kvaf::aicpu_urma;

struct alignas(64) Buffers {
  std::array<std::uint8_t, 4096> sq;
  std::array<std::uint8_t, 4096> hbm;
  std::array<std::uint8_t, 4096> source_hbm;
};

void TestCopy(std::uint32_t workers, std::uint32_t batch, kvaf::GatherSqCopyMode mode) {
  std::array<Buffers, 8> buffers{};
  std::array<SqCopyBenchmarkQueue, 8> queues{};
  std::array<SqCopyBenchmarkStatus, 8> statuses{};
  std::uint64_t counter = 0U;
  std::uint64_t verify_counter = 0U;
  SqCopyBenchmarkArgs args{reinterpret_cast<std::uint64_t>(queues.data()),
      reinterpret_cast<std::uint64_t>(statuses.data()), reinterpret_cast<std::uint64_t>(&counter),
      0U, workers, batch, 0U, mode, reinterpret_cast<std::uint64_t>(&verify_counter)};
  for (std::uint64_t round = 0U; round < 3U; ++round) {
    args.round = round;
    for (std::uint32_t target = 0U; target < 4U; ++target) {
      args.target_sq = target;
      counter = 0U;
      verify_counter = 0U;
      for (std::uint32_t worker = 0U; worker < workers; ++worker) {
        buffers[worker].sq.fill(0xa5U);
        buffers[worker].hbm.fill(0xa5U);
        queues[worker] = {reinterpret_cast<std::uint64_t>(buffers[worker].sq.data()),
            reinterpret_cast<std::uint64_t>(buffers[worker].hbm.data()), 4096U,
            (round + 1U) * 64U, 1664U + worker * 64U,
            reinterpret_cast<std::uint64_t>(buffers[worker].source_hbm.data())};
        for (std::uint64_t index = 0U; index < buffers[worker].source_hbm.size(); ++index)
          buffers[worker].source_hbm[index] = SqCopyBenchmarkPattern(index, worker, round);
      }
      std::vector<std::thread> threads;
      for (std::uint32_t thread = 0U; thread < workers; ++thread)
        threads.emplace_back([&] { assert(KvafAicpuUrmaSqCopyBenchmark(&args) == 0U); });
      for (auto& thread : threads) thread.join();
      assert(counter == workers);
      threads.clear();
      for (std::uint32_t thread = 0U; thread < workers; ++thread)
        threads.emplace_back([&] { assert(KvafAicpuUrmaSqCopyVerify(&args) == 0U); });
      for (auto& thread : threads) thread.join();
      assert(verify_counter == workers);
      for (std::uint32_t worker = 0U; worker < workers; ++worker) {
        const auto& status = statuses[worker];
        const auto& queue = queues[worker];
        assert(status.magic == kSqCopyBenchmarkMagic && status.worker == worker);
        assert(status.round == round && status.result == 0U && status.bytes == queue.bytes);
        assert(status.entry_ns <= status.copy_begin_ns && status.copy_begin_ns <= status.copy_end_ns);
        assert(status.copy_calls == (queue.bytes + batch * 192U - 1U) / (batch * 192U));
        const auto& destination = (target & 1U) != 0U ? buffers[worker].sq : buffers[worker].hbm;
        const auto& untouched = (target & 1U) != 0U ? buffers[worker].hbm : buffers[worker].sq;
        for (std::uint64_t index = 0U; index < destination.size(); ++index) {
          const auto expected = index >= queue.offset && index - queue.offset < queue.bytes
              ? SqCopyBenchmarkPattern(target >= 2U ? index - queue.offset
                  : (index - queue.offset) % (batch * 192U), worker, round)
              : 0xa5U;
          assert(destination[index] == expected);
          assert(untouched[index] == 0xa5U);
          assert(buffers[worker].source_hbm[index] == SqCopyBenchmarkPattern(index, worker, round));
        }
      }
      assert(KvafAicpuUrmaSqCopyBenchmark(&args) == 1U);
    }
  }
}

void TestInvalid() {
  assert(KvafAicpuUrmaSqCopyBenchmark(nullptr) == 1U);
  Buffers buffers{};
  SqCopyBenchmarkQueue queue{reinterpret_cast<std::uint64_t>(buffers.sq.data()),
      reinterpret_cast<std::uint64_t>(buffers.hbm.data()), 4096U, 0U, 64U, 0U};
  SqCopyBenchmarkStatus status{};
  std::uint64_t counter = 0U;
  std::uint64_t verify_counter = 0U;
  SqCopyBenchmarkArgs valid{reinterpret_cast<std::uint64_t>(&queue),
      reinterpret_cast<std::uint64_t>(&status), reinterpret_cast<std::uint64_t>(&counter),
      0U, 1U, 8U, 1U, kvaf::GatherSqCopyMode::kMemcpy,
      reinterpret_cast<std::uint64_t>(&verify_counter)};
  for (const auto bytes : {0ULL, 63ULL, 4160ULL}) {
    counter = 0U;
    queue.bytes = bytes;
    assert(KvafAicpuUrmaSqCopyBenchmark(&valid) == 1U && status.result == 1U);
  }
  queue.bytes = 64U;
  queue.offset = 4096U;
  counter = 0U;
  assert(KvafAicpuUrmaSqCopyBenchmark(&valid) == 1U);
  queue.offset = 4032U;
  counter = 0U;
  assert(KvafAicpuUrmaSqCopyBenchmark(&valid) == 0U);
  assert(KvafAicpuUrmaSqCopyVerify(&valid) == 0U);
  buffers.sq[queue.offset + 8U] ^= 1U;
  verify_counter = 0U;
  assert(KvafAicpuUrmaSqCopyVerify(&valid) == 1U && status.result == 2U);
  assert(status.mismatch_offset == queue.offset + 8U && status.actual != status.expected);
  auto invalid = valid;
  invalid.workers = 9U;
  assert(KvafAicpuUrmaSqCopyBenchmark(&invalid) == 1U);
  invalid = valid;
  invalid.target_sq = 4U;
  assert(KvafAicpuUrmaSqCopyBenchmark(&invalid) == 1U);
  invalid = valid;
  invalid.batch_wqes = 7U;
  assert(KvafAicpuUrmaSqCopyBenchmark(&invalid) == 1U);
  invalid = valid;
  invalid.copy_mode = static_cast<kvaf::GatherSqCopyMode>(2U);
  assert(KvafAicpuUrmaSqCopyBenchmark(&invalid) == 1U);
  queue.offset = 0U;
  for (const auto target : {2U, 3U}) {
    valid.target_sq = target;
    for (const auto source : {0ULL, static_cast<unsigned long long>(
         reinterpret_cast<std::uint64_t>(buffers.source_hbm.data()) + 1U),
         static_cast<unsigned long long>(target == 3U ? queue.sq : queue.hbm),
         UINT64_MAX - 63ULL}) {
      counter = 0U;
      queue.source_hbm = source;
      assert(KvafAicpuUrmaSqCopyBenchmark(&valid) == 1U);
    }
    queue.source_hbm = reinterpret_cast<std::uint64_t>(buffers.source_hbm.data());
    for (std::uint64_t index = 0U; index < queue.bytes; ++index)
      buffers.source_hbm[index] = SqCopyBenchmarkPattern(index, 0U, valid.round);
    counter = 0U;
    verify_counter = 0U;
    assert(KvafAicpuUrmaSqCopyBenchmark(&valid) == 0U);
    assert(KvafAicpuUrmaSqCopyVerify(&valid) == 0U);
    auto& destination = target == 3U ? buffers.sq : buffers.hbm;
    destination[8U] ^= 1U;
    verify_counter = 0U;
    assert(KvafAicpuUrmaSqCopyVerify(&valid) == 1U && status.result == 2U);
    assert(status.mismatch_offset == 8U && status.actual != status.expected);
  }
}

}

int main() {
  TestInvalid();
  for (const auto mode : {kvaf::GatherSqCopyMode::kMemcpy, kvaf::GatherSqCopyMode::kPaired}) {
    for (const auto workers : {1U, 5U, 8U}) {
      for (const auto batch : {8U, 16U, 32U, 64U}) TestCopy(workers, batch, mode);
    }
  }
}
