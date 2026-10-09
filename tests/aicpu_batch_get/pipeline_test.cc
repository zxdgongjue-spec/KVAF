#include "pipeline.h"
#include "scatter_copy.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using namespace kvaf::aicpu_test;
void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
std::uint64_t Now() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct ScriptIO {
  Result status{Result::kPending};
  std::uint64_t done{}, clock{}, reads{}, scatters{}, ready{}, scatter_cost{};
  Result Status() { return status; }
  std::uint64_t ReadRangeDone(std::uint32_t) { ++reads; return done; }
  std::uint64_t Now() { return ++clock; }
  void RangeReady(std::uint32_t, std::uint64_t tick) { ready = tick; }
  void Scatter(std::uint32_t, std::uint64_t) { ++scatters; clock += scatter_cost; }
  void Relax() {}
};
void WaitFailures() {
  ScriptIO stale; stale.done = 2;
  Check(RunPipelineRange(0, 3, 0, 5, stale) == Result::kTimeout && !stale.scatters,
        "previous-round ready allowed scatter");
  ScriptIO future; future.done = 4;
  Check(RunPipelineRange(0, 3, 0, 5, future) == Result::kSequenceError && !future.scatters,
        "future ready allowed scatter");
  ScriptIO late; late.done = 3; late.clock = 5;
  Check(RunPipelineRange(0, 3, 0, 5, late) == Result::kTimeout && !late.scatters,
        "late ready bypassed the round deadline");
  ScriptIO cancelled; cancelled.status = Result::kDataMismatch;
  Check(RunPipelineRange(0, 3, 0, 5, cancelled) == Result::kDataMismatch && !cancelled.reads,
        "range ignored another block's failure");
  ScriptIO overrun; overrun.done = 3; overrun.scatter_cost = 5;
  Check(RunPipelineRange(0, 3, 0, 5, overrun) == Result::kTimeout && overrun.scatters == 1,
        "scatter overran the batch deadline without failing");
  ScriptIO valid; valid.done = 3;
  Check(RunPipelineRange(0, 3, 0, 5, valid) == Result::kOk && valid.scatters == 1 && valid.ready == 1,
        "valid range was not consumed exactly once");
  Check(RunPipelineRange(kMaxScatterThreads, 3, 0, 5, valid) == Result::kInvalidArgs,
        "unbounded worker accepted");
  Check(RunPipelineRange(0, 0, 0, 5, valid) == Result::kInvalidArgs &&
        RunPipelineRange(0, 3, 0, 0, valid) == Result::kInvalidArgs,
        "invalid sequence/deadline accepted");
}

struct ConcurrentIO {
  Config config;
  Layout layout{};
  std::uint32_t workers;
  std::uint64_t sequence{};
  std::vector<std::uint8_t> source, output;
  std::array<std::atomic<std::uint64_t>, kMaxScatterThreads> ready;
  std::array<std::atomic<bool>, kMaxScatterThreads> started;
  std::atomic<Result> status{Result::kPending};
  std::atomic<bool> producer_finished{false}, consumed_early{false};
  std::atomic<std::uint64_t> first_ready{UINT64_MAX}, calls{0};
  explicit ConcurrentIO(Config c) : config(c), workers(ActiveThreads(c)) {
    Check(MakeLayout(c, &layout), "invalid pipeline test layout");
    source.resize(layout.payload_bytes); output.assign(layout.output_bytes, 0xA5);
    for (std::uint32_t w = 0; w < kMaxScatterThreads; ++w) {
      ready[w].store(0); started[w].store(false);
    }
  }
  Result Status() { return status.load(std::memory_order_acquire); }
  std::uint64_t ReadRangeDone(std::uint32_t w) { return ready[w].load(std::memory_order_acquire); }
  std::uint64_t Now() { return ::Now(); }
  void RangeReady(std::uint32_t, std::uint64_t observed) {
    auto old = first_ready.load();
    while (observed < old && !first_ready.compare_exchange_weak(old, observed)) {}
  }
  void Relax() { std::this_thread::yield(); }
  void Scatter(std::uint32_t w, std::uint64_t) {
    if (w == 0 && workers > 1 && ready[1].load(std::memory_order_acquire) < sequence)
      consumed_early.store(true);
    started[w].store(true, std::memory_order_release);
    // Hold worker 0's scatter in flight while the producer writes later
    // ranges. This deterministic gate tests overlap without timing guesses.
    while (w == 0 && !producer_finished.load(std::memory_order_acquire) &&
           Status() == Result::kPending) Relax();
    const auto range = ScatterRange(config, w, workers);
    CopyScatterRows(output.data() + range.begin * layout.stride,
                    source.data() + range.begin * config.row_bytes,
                    range.end - range.begin, config.row_bytes, layout.stride);
    ++calls;
  }
  void Produce(std::uint32_t w) {
    const auto range = ScatterRange(config, w, workers);
    for (auto slot = range.begin; slot < range.end; ++slot) {
      const auto row = LogicalRowForOutputSlot(config, slot);
      const auto tag = RowTag(sequence, row);
      auto* data = source.data() + slot * config.row_bytes;
      std::memcpy(data, &tag, sizeof(tag));
      for (std::uint64_t b = 8; b < config.row_bytes; ++b)
        data[b] = PayloadPattern(config, sequence, row, b);
    }
    ready[w].store(sequence, std::memory_order_release);
  }
  void Verify() {
    for (std::uint64_t slot = 0; slot < config.rows; ++slot) {
      const auto* data = output.data() + slot * layout.stride;
      Check(std::memcmp(data, source.data() + slot * config.row_bytes, config.row_bytes) == 0,
            "pipeline copied stale or incomplete payload");
      for (auto b = config.row_bytes; b < layout.stride; ++b)
        Check(data[b] == 0xA5, "pipeline overwrote output padding");
    }
  }
};
void ConcurrentRanges() {
  for (auto bytes : {8, 65, 656}) for (auto rows : {3, 67, 4096})
    for (auto requested : {1, 4, 8}) {
      Config c; c.rows = rows; c.row_bytes = bytes; c.scatter_threads = requested;
      c.transfer_mode = TransferMode::kScatterPipeline; c.verify_payload = 1;
      ConcurrentIO io(c);
      std::uint64_t previous_end = 0;
      for (std::uint32_t w = 0; w < io.workers; ++w) {
        const auto range = ScatterRange(c, w, io.workers);
        Check(range.begin == previous_end && range.end > range.begin &&
            range.begin * c.row_bytes % 64 == 0, "pipeline ranges overlap/share an input line");
        previous_end = range.end;
        Check(PipelineReadyOffset(w) % 64 == 0 && PipelineReadyOffset(w) >= kDoneOffset + sizeof(Control) &&
            PipelineReadyOffset(w) + sizeof(Control) <= kStagingOffset, "ready overlaps controls/data");
      }
      Check(previous_end == c.rows, "pipeline omitted tail rows");
      for (std::uint64_t seq = 1; seq <= 3; ++seq) {
        io.sequence = seq; io.producer_finished.store(false); io.consumed_early.store(false);
        io.first_ready.store(UINT64_MAX);
        for (std::uint32_t w = 0; w < io.workers; ++w) io.started[w].store(false);
        const auto start = Now();
        std::vector<std::thread> threads;
        std::array<Result, kMaxScatterThreads> outcomes{};
        for (std::uint32_t w = 0; w < io.workers; ++w)
          threads.emplace_back([&, w] {
            outcomes[w] = RunPipelineRange(w, seq, start, 2000000000ULL, io);
          });
        io.Produce(0);
        while (!io.started[0].load(std::memory_order_acquire) && Now() - start < 2000000000ULL)
          std::this_thread::yield();
        if (!io.started[0].load()) io.status.store(Result::kTimeout);
        for (std::uint32_t w = 1; w < io.workers; ++w) io.Produce(w);
        io.producer_finished.store(true, std::memory_order_release);
        for (auto& thread : threads) thread.join();
        Check(io.Status() == Result::kPending, "producer/consumer did not make progress");
        for (std::uint32_t w = 0; w < io.workers; ++w)
          Check(outcomes[w] == Result::kOk, "pipeline range failed");
        Check(io.workers == 1 || io.consumed_early.load(), "scatter waited for the whole batch");
        Check(io.calls.load() == seq * io.workers && io.first_ready.load() >= start &&
              io.first_ready.load() <= Now(), "range reused/completion timing mismatch");
        io.Verify();
      }
    }
}
void PayloadValidation() {
  Config c; c.verify_payload = 1;
  for (auto seq : {1ULL, 255ULL, 256ULL, 65535ULL, 65536ULL})
    for (auto byte : {8ULL, 64ULL, 655ULL})
      Check(PayloadPattern(c, seq, 3, byte) != PayloadPattern(c, seq + 1, 3, byte),
            "payload did not change between adjacent sequences");
  c.verify_payload = 0;
  Check(PayloadPattern(c, 99, 3, 655) == Pattern(3, 655), "default bandwidth pattern changed");
}
}  // namespace
int main() {
  try {
    WaitFailures(); ConcurrentRanges(); PayloadValidation();
    std::puts("aicpu pipeline: PASS"); return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what()); return 1;
  }
}
