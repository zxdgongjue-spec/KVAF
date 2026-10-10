#include "../access_operator/src/aiv_sq_copy_benchmark/kernel.asc"

#include <array>
#include <cassert>
#include <thread>
#include <vector>

extern "C" std::uint32_t KvafAicpuUrmaSqCopyVerify(SqCopyBenchmarkArgs* args);

namespace {

struct alignas(64) Buffers {
  std::array<std::uint8_t, 4096> sq;
  std::array<std::uint8_t, 4096> hbm;
  std::array<std::uint8_t, 4096> source;
};

void TestCopy(std::uint32_t workers, std::uint32_t batch) {
  std::array<Buffers, 8> buffers{};
  std::array<SqCopyBenchmarkQueue, 8> queues{};
  std::array<SqCopyBenchmarkStatus, 8> statuses{};
  std::uint64_t verify_counter = 0U;
  SqCopyBenchmarkArgs args{reinterpret_cast<std::uint64_t>(queues.data()),
      reinterpret_cast<std::uint64_t>(statuses.data()), 0U, 0U, workers, batch,
      0U, kvaf::GatherSqCopyMode::kMemcpySp, reinterpret_cast<std::uint64_t>(&verify_counter)};
  for (std::uint64_t round = 0U; round < 3U; ++round) {
    args.round = round;
    for (std::uint32_t target = 0U; target < 4U; ++target) {
      args.target_sq = target;
      verify_counter = 0U;
      for (std::uint32_t worker = 0U; worker < workers; ++worker) {
        buffers[worker].sq.fill(0xa5U);
        buffers[worker].hbm.fill(0xa5U);
        queues[worker] = {reinterpret_cast<std::uint64_t>(buffers[worker].sq.data()),
            reinterpret_cast<std::uint64_t>(buffers[worker].hbm.data()), 4096U,
            (round + 1U) * 64U, 1664U + worker * 64U,
            reinterpret_cast<std::uint64_t>(buffers[worker].source.data())};
        for (std::uint64_t index = 0U; index < buffers[worker].source.size(); ++index)
          buffers[worker].source[index] = SqCopyBenchmarkPattern(index, worker, round);
      }
      std::vector<std::thread> threads;
      for (std::uint32_t worker = 0U; worker < workers; ++worker)
        threads.emplace_back([&, worker] {
          kvaf::aiv_sq_copy_stub::worker = worker;
          kvaf::aiv_sq_copy_stub::mte3_waits = 0U;
          KvafAivSqCopyBenchmark(args);
          assert(kvaf::aiv_sq_copy_stub::pending.empty());
          assert(kvaf::aiv_sq_copy_stub::mte3_waits >= 1U);
        });
      for (auto& thread : threads) thread.join();
      for (std::uint32_t worker = 0U; worker < workers; ++worker) {
        assert(KvafAicpuUrmaSqCopyVerify(&args) == 0U);
        assert(statuses[worker].copy_end_ns > statuses[worker].copy_begin_ns);
        assert(statuses[worker].thread_cpu_available == 0U);
        const auto& untouched = (target & 1U) != 0U ? buffers[worker].hbm : buffers[worker].sq;
        const auto& destination = (target & 1U) != 0U ? buffers[worker].sq : buffers[worker].hbm;
        for (std::uint64_t index = 0U; index < destination.size(); ++index) {
          assert(untouched[index] == 0xa5U);
          assert(buffers[worker].source[index] == SqCopyBenchmarkPattern(index, worker, round));
          if (index < queues[worker].offset || index >= queues[worker].offset + queues[worker].bytes)
            assert(destination[index] == 0xa5U);
        }
      }
      auto& destination = (target & 1U) != 0U ? buffers[0].sq : buffers[0].hbm;
      destination[queues[0].offset + 8U] ^= 1U;
      verify_counter = 0U;
      assert(KvafAicpuUrmaSqCopyVerify(&args) == 0U && statuses[0].result == 2U);
    }
  }
}

void TestInvalid() {
  Buffers buffers{};
  SqCopyBenchmarkQueue queue{reinterpret_cast<std::uint64_t>(buffers.sq.data()),
      reinterpret_cast<std::uint64_t>(buffers.hbm.data()), 4096U, 0U, 64U,
      reinterpret_cast<std::uint64_t>(buffers.source.data())};
  SqCopyBenchmarkStatus status{};
  SqCopyBenchmarkArgs args{reinterpret_cast<std::uint64_t>(&queue),
      reinterpret_cast<std::uint64_t>(&status), 0U, 0U, 1U, 8U,
      0U, kvaf::GatherSqCopyMode::kMemcpySp, 0U};
  kvaf::aiv_sq_copy_stub::worker = 0U;
  for (const auto bytes : {0ULL, 63ULL, 4160ULL}) {
    queue.bytes = bytes;
    KvafAivSqCopyBenchmark(args);
    assert(status.result == 1U && kvaf::aiv_sq_copy_stub::pending.empty());
  }
  queue.bytes = 64U;
  queue.offset = 4096U;
  KvafAivSqCopyBenchmark(args);
  assert(status.result == 1U);
  queue.offset = 4032U;
  KvafAivSqCopyBenchmark(args);
  assert(status.result == 0U);
  args.target_sq = 2U;
  queue.source_hbm = queue.hbm + queue.offset;
  KvafAivSqCopyBenchmark(args);
  assert(status.result == 1U);
  args.target_sq = 4U;
  KvafAivSqCopyBenchmark(args);
  assert(status.result == 1U);
  args.round = 64U;
  KvafAivSqCopyClock(args);
  assert(status.result == 0U && status.copy_end_ns - status.copy_begin_ns >= 64U);
}

}

int main() {
  TestInvalid();
  for (const auto workers : {1U, 5U, 8U}) {
    for (const auto batch : {8U, 16U, 32U, 64U}) TestCopy(workers, batch);
  }
}
