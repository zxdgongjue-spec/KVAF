#include "device_loop.h"
#include "metrics.h"
#include "scatter_pool.h"
#include "scatter_copy.h"

#include <cstdio>
#include <cstring>
#include <functional>

namespace {
using namespace kvaf::aicpu_test;
struct Simulator {
  Config config;
  Layout layout{};
  std::vector<std::uint8_t> staging, output;
  std::uint64_t clock{}, sequence{}, done{}, publications{};
  std::uint64_t prepared_rounds{}, verified_sequence{};
  std::uint64_t preparation_ticks{}, verification_ticks{};
  bool output_cached{true};  // Initial fill, then scatter/verification refills it.
  int reads{};
  bool timeout{}, future{}, corrupt{};
  explicit Simulator(Config c) : config(c) {
    Require(MakeLayout(c, &layout), "simulator config");
    staging.resize(layout.payload_bytes); output.assign(layout.output_bytes, 0xA5);
  }
  std::uint64_t Frequency() { return 1000; }
  std::uint64_t Now() { return ++clock; }
  void PrepareRound(const Config&, const Layout&) {
    Require(prepared_rounds == publications && verified_sequence == sequence,
            "cache preparation must follow verification, once per round");
    output_cached = false;
    ++prepared_rounds;
    clock += preparation_ticks;
  }
  void PublishRequest(std::uint64_t seq) {
    Require(seq == sequence + 1, "sequence must increase after scatter");
    Require(prepared_rounds == publications + 1 && !output_cached,
            "request published before cache preparation completed");
    sequence = seq; reads = 0; ++publications;
  }
  std::uint64_t ReadDone() {
    if (future) return sequence + 1;
    if (timeout || ++reads < 3) return done;
    // Model the CPU WRITE and delayed completion, with a new tag every round.
    for (std::uint64_t slot = 0; slot < config.rows; ++slot) {
      const auto row = LogicalRowForOutputSlot(config, slot);
      auto* dst = staging.data() + slot * config.row_bytes;
      auto tag = RowTag(sequence, row);
      if (corrupt && row == 1) --tag;
      std::memcpy(dst, &tag, 8);
      for (std::uint64_t b = 8; b < config.row_bytes; ++b) dst[b] = Pattern(row, b);
    }
    return done = sequence;
  }
  void Scatter(const Config& c, const Layout& l) {
    Require(done == sequence, "scatter ran before completion");
    Require(!output_cached, "scatter inherited initialization/verification cache state");
    CopyScatterRows(output.data(), staging.data(), c.rows, c.row_bytes, l.stride);
    output_cached = true;
    clock += 5;
  }
  bool VerifyTags(std::uint64_t seq, const Config& c, const Layout& l) {
    output_cached = true;
    clock += verification_ticks;
    for (std::uint64_t row = 0; row < c.rows; ++row) {
      std::uint64_t tag;
      std::memcpy(&tag, output.data() + OutputOffset(c, l, row), 8);
      if (tag != RowTag(seq, row)) return false;
    }
    verified_sequence = seq;
    return true;
  }
};
void SuccessAndMetrics() {
  Config c; c.rows = 3; c.row_bytes = 128; c.warmup = 2; c.iterations = 4;
  Simulator io(c);
  std::vector<Sample> samples(c.iterations); KernelResult result{};
  Require(RunDevice(c, io, samples.data(), &result) == Result::kOk, "device loop failed");
  Require(io.publications == 6 && result.completed_rounds == 6, "warmup/round count mismatch");
  for (std::uint64_t slot = 0; slot < c.rows; ++slot) {
    const auto row = c.rows - 1 - slot;
    const auto* bytes = io.output.data() + slot * io.layout.stride;
    for (std::uint64_t b = 8; b < c.row_bytes; ++b)
      Require(bytes[b] == Pattern(row, b), "incorrect output-slot scatter payload");
    for (std::uint64_t b = c.row_bytes; b < io.layout.stride; ++b)
      Require(bytes[b] == 0xA5, "scatter overwrote guard");
  }
  const auto report = Summarize(c, result, samples);
  Require(report.status == 0 && report.mean_ns == report.wait_ns + report.scatter_ns &&
          report.p50_ns > 0 && report.p99_ns >= report.p50_ns && report.gbps > 0, "metrics mismatch");
  samples[0].scatter_ticks++;
  bool rejected = false;
  try { (void)Summarize(c, result, samples); } catch (...) { rejected = true; }
  Require(rejected, "invalid timing accepted");
}
void Failures() {
  Config c; c.rows = 3; c.row_bytes = 64; c.iterations = 1; c.warmup = 0; c.timeout_ms = 5;
  Sample sample{}; KernelResult result{};
  Simulator missing(c); missing.timeout = true;
  Require(RunDevice(c, missing, &sample, &result) == Result::kTimeout &&
          result.completed_rounds == 0 && result.failed_sequence == 1, "missing completion must timeout");
  Simulator future(c); future.future = true;
  Require(RunDevice(c, future, &sample, &result) == Result::kSequenceError, "future sequence accepted");
  Simulator corrupt(c); corrupt.corrupt = true;
  Require(RunDevice(c, corrupt, &sample, &result) == Result::kDataMismatch, "stale row accepted");
}
void CachePreparationAndTiming() {
  for (auto warmup : {0, 2}) {
    Config c; c.rows = 67; c.row_bytes = 656; c.warmup = warmup; c.iterations = 4;
    c.timeout_ms = 10;
    Simulator baseline(c), delayed(c);
    // Both costs exceed the per-round timeout; neither belongs to a sample.
    delayed.preparation_ticks = 1000;
    delayed.verification_ticks = 2000;
    std::vector<Sample> baseline_samples(c.iterations), delayed_samples(c.iterations);
    KernelResult baseline_result{}, delayed_result{};
    Require(RunDevice(c, baseline, baseline_samples.data(), &baseline_result) == Result::kOk,
            "baseline cache preparation loop failed");
    Require(RunDevice(c, delayed, delayed_samples.data(), &delayed_result) == Result::kOk,
            "cache preparation/verification leaked into round timeout");
    Require(delayed.prepared_rounds == c.warmup + c.iterations &&
            delayed.verified_sequence == delayed.prepared_rounds,
            "cache preparation skipped first/warmup/measured round or verification");
    Require(delayed.clock == baseline.clock + delayed.prepared_rounds * 3000,
            "simulator did not apply preparation/verification cost");
    for (std::size_t i = 0; i < baseline_samples.size(); ++i) {
      const auto& a = baseline_samples[i]; const auto& b = delayed_samples[i];
      Require(a.total_ticks == b.total_ticks && a.wait_ticks == b.wait_ticks &&
              a.scatter_ticks == b.scatter_ticks,
              "cache preparation/verification changed reported timing");
    }
  }
}
void Bounds() {
  Config c; Layout l{};
  Require(MakeLayout(c, &l), "default layout invalid");
  c.rows = UINT64_MAX; Require(!MakeLayout(c, &l), "overflow accepted");
  c = {}; c.row_bytes = 7; Require(!MakeLayout(c, &l), "undersized tag accepted");
  c = {}; c.warmup = UINT64_MAX; Require(!MakeLayout(c, &l), "round overflow accepted");
  c = {}; c.iterations = 0; Require(!MakeLayout(c, &l), "zero samples accepted");
  c = {}; c.timeout_ms = 0; Require(!MakeLayout(c, &l), "unbounded wait accepted");
  c = {}; c.magic = 0; Require(!MakeLayout(c, &l), "invalid ABI accepted");
}
void ParallelScatter() {
  for (auto bytes : {8, 65, 656, 4096}) for (auto rows : {3, 67, 4096})
    for (auto threads : {1, 2, 4, 8}) {
      Config c; c.rows = rows; c.row_bytes = bytes; c.scatter_threads = threads;
      Simulator io(c);
      struct Context { Simulator* io; } context{&io};
      ScatterPool pool;
      Require(pool.Initialize(ActiveThreads(c), [](void* opaque, std::uint32_t worker) {
        auto& io = *static_cast<Context*>(opaque)->io;
        auto range = ScatterRange(io.config, worker, ActiveThreads(io.config));
        CopyScatterRows(io.output.data() + OutputSlotOffset(io.layout, range.begin),
                        io.staging.data() + range.begin * io.config.row_bytes,
                        range.end - range.begin, io.config.row_bytes, io.layout.stride);
      }, &context) == 0, "pthread initialization failed");
      std::uint64_t previous = 0;
      for (std::uint32_t worker = 0; worker < ActiveThreads(c); ++worker) {
        auto range = ScatterRange(c, worker, ActiveThreads(c));
        Require(range.begin == previous && range.end > range.begin &&
                (range.begin * c.row_bytes) % 64 == 0, "overlapping/unaligned partition");
        previous = range.end;
      }
      Require(previous == c.rows, "missing rows");
      for (unsigned round = 0; round < 4; ++round) {
        for (std::size_t b = 0; b < io.staging.size(); ++b)
          io.staging[b] = static_cast<std::uint8_t>(b * 17 + round);
        pool.Run();
        for (std::uint64_t slot = 0; slot < c.rows; ++slot) {
          const auto* dst = io.output.data() + OutputSlotOffset(io.layout, slot);
          Require(std::memcmp(dst, io.staging.data() + slot * c.row_bytes, c.row_bytes) == 0,
                  "parallel scatter data mismatch");
          for (auto b = c.row_bytes; b < io.layout.stride; ++b)
            Require(dst[b] == 0xA5, "parallel scatter guard overwritten");
        }
      }
    }
  Config c; Layout l{};
  c.rows = 4096; c.row_bytes = 656;
  Require(MakeLayout(c, &l) && l.payload_bytes == 2686976 && l.stride == 768,
          "4096x656 layout mismatch");
  c.scatter_threads = 0; Require(!MakeLayout(c, &l), "zero threads accepted");
  c.scatter_threads = 33; Require(!MakeLayout(c, &l), "too many threads accepted");
}
void DeviceSelection() {
  Require(SelectDevice(4, false, 0) == 4, "CPU device-id was ignored");
  Require(SelectDevice(4, true, 4) == 4, "matching pinned device rejected");
  bool rejected = false;
  try { (void)SelectDevice(4, true, 0); } catch (...) { rejected = true; }
  Require(rejected, "mismatched NPU devices accepted");
  rejected = false;
  try { (void)SelectDevice(UINT32_MAX, false, 0); } catch (...) { rejected = true; }
  Require(rejected, "overflowing device-id accepted");

}
}  // namespace
int main() {
  try { ParallelScatter(); Bounds(); DeviceSelection(); SuccessAndMetrics(); Failures();
        CachePreparationAndTiming();
        std::puts("aicpu get protocol: PASS"); return 0; }
  catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
